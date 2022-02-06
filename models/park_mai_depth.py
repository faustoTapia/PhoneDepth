import sys

from tensorflow._api.v2 import math
from tensorflow.python.keras.engine.sequential import Sequential
sys.path.append('./')
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras import initializers
import math
from os import path

from .mobilenetV3 import MobileNetV3Large
from depth_utils import gradient_loss_multiscale_func, md_loss_total, si_rmse, rmse_nonlog, error_ordinal_wrap
from .compileStrategies import compile_md, compile_md_doubleDepth
from .operators import OutputActivation, group_convolution, output_activations_dic, limited_logsigmoid_activation
from .operators import BuildingBlocks, building_blocks_dic, block_RRDB, block_IMDB
from depth_utils import parkmai_total_loss_func, ssi_loss_func
from schedulers import LearningRateSearchingScheduleLinear, LearningRateSearchingSchedulePow
from schedulers import OneCycleLR, CyclicLR

# Encoder based on mobilenetV3
def mobilenetV3encoder(input_shape = (224,224,3), alpha=0.75, minimalistic = False,  name='mobilenetv3_encoder'):
    base_model = MobileNetV3Large(input_shape=input_shape, alpha=alpha, minimalistic=minimalistic,
                                                        include_top=False, weights='imagenet')
    features = ['expanded_conv/depthwise',               # Take input of this one as named previous operation is named dynamically
                        'expanded_conv_1/project/BatchNorm',
                        'expanded_conv_3/project/BatchNorm',
                        'expanded_conv_6/project/BatchNorm',
                        'expanded_conv_14/Add'] 
    # Input placeholder
    inp = base_model.input
    # Layer outputs
    outputs = [base_model.get_layer(feats).output for feats in features[1:]]
    outputs = [base_model.get_layer(features[0]).input] + outputs
    #adding input into the outputs
    outputs = [inp] + outputs
    encoder = tf.keras.models.Model(inputs = inp, outputs=outputs, name=name)
    
    feat_out_channels = [outputs[i].shape[-1] for i in range(1, len(outputs))]
    return encoder, feat_out_channels


# Returns prelu with torch default configurations
def torch_prelu():
    return PReLU(alpha_initializer=initializers.Constant(0.25),
                            shared_axes=[1,2,3])
def standard_bn():
    return BatchNormalization(axis=-1, momentum=0.01, epsilon=1.1e-5)


################### DECOUPLED ENC-DEC FUNCTIONAL API ##################
#########################################################################################

def upConv_f(out_channels, kernel_size=3, ratio=2, kernel_initializer = 'he_normal', building_block: BuildingBlocks=BuildingBlocks.DEPTHWISE_SEPARABLE,
            depthwise_separable=True, n_blocks=1, n_groups=8, name=''):

    upconv = lambda x: upConv_inp(x, out_channels, kernel_size=kernel_size, ratio=ratio, kernel_initializer=kernel_initializer, building_block=building_block,
                                  depthwise_separable=depthwise_separable, n_blocks=n_blocks, n_groups=n_groups, name=name)

    return upconv

def upConv_inp(x, out_channels, kernel_size=3, ratio=2, kernel_initializer = 'he_normal', building_block: BuildingBlocks=BuildingBlocks.DEPTHWISE_SEPARABLE, depthwise_separable=True, n_blocks=1, n_groups=8, name=None):

    out = x

    if building_block == BuildingBlocks.DEPTHWISE_SEPARABLE:
        seq_layers = []
        for i in range(n_blocks):
            seq_layers += [SeparableConv2D(out_channels, kernel_size=kernel_size, strides=1, padding='same', use_bias=False, depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer)]
        
        seq_layers += [UpSampling2D(size=(ratio, ratio), interpolation='nearest')]
        # seq_layers += [UpSampling2D(size=(ratio, ratio), interpolation='bilinear')]
        out = Sequential(seq_layers, name=name)(out)

    elif building_block == BuildingBlocks.GROUP_CONVOLUTIONS:
        for i in range(n_blocks):
            out = group_convolution(out, filter_size=kernel_size, strides=1, num_filters=out_channels, groups=n_groups, kernel_initializer=kernel_initializer,
                                    name=name+'/group_conv{:02d}'.format(i) if name is not None else name)

    elif building_block == BuildingBlocks.RRDB:
        for i in range(n_blocks):
            out= block_RRDB(out, kernel_size=kernel_size, depthwise_separable=depthwise_separable, weights_initializer=kernel_initializer,
                            name=name + '/group_conv{:02d}'.format(i) if name is not None else name)
        
    elif building_block == BuildingBlocks.IMDB:
        for i in range(n_blocks):
            out = block_IMDB(out, kernel_size=kernel_size, depthwise_separable=depthwise_separable, weights_initializer=kernel_initializer,
                            name=name + '/IMDB{:02d}'.format(i) if name is not None else name)

    # Output layer for channel size flexibility
    if building_block in [BuildingBlocks.RRDB, BuildingBlocks.IMDB]:
        if depthwise_separable:
            out = SeparableConv2D(out_channels, kernel_size=kernel_size, padding='same', depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer,
                                  name=name + '/conv_depth_sep_out' if name is not None else name)(out)
        else:
            out = Conv2D(out_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_initializer,
                         name=name + '/conv_out' if name is not None else name)(out)

    # Sinde Depthwise separable is set in Sequential mode for compatibility with previously trained models
    if building_block != BuildingBlocks.DEPTHWISE_SEPARABLE:
        out = UpSampling2D(size=(ratio, ratio), interpolation='nearest', name=name +'/upsamp' if name is not None else name)(out)
        # out = UpSampling2D(size=(ratio, ratio), interpolation='bilinear', name=name +'/upsamp' if name is not None else name)(out)

    return out


def conv_f(out_channels, kernel_size=3, strides=1, kernel_initializer = 'he_normal', building_block: BuildingBlocks=BuildingBlocks.DEPTHWISE_SEPARABLE, depthwise_separable=True, n_blocks=1, n_groups=8 , name=''):

    conv = lambda x: conv_inp(x, out_channels, kernel_size=kernel_size, strides=strides, kernel_initializer=kernel_initializer,
                                building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=n_blocks, n_groups=n_groups, name=name)

    return conv

def conv_inp(x, out_channels, kernel_size=3, strides=1, kernel_initializer = 'he_normal', building_block: BuildingBlocks=BuildingBlocks.DEPTHWISE_SEPARABLE, depthwise_separable=True, n_blocks=1, n_groups=8 , name=''):
    out = x

    if building_block == BuildingBlocks.DEPTHWISE_SEPARABLE:
        seq_layers = []
        for i in range(n_blocks):
            seq_layers += [SeparableConv2D(out_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer), torch_prelu()]
        out = Sequential(seq_layers, name=name)(out)

    if building_block == BuildingBlocks.GROUP_CONVOLUTIONS:
        for i in range(n_blocks):
            out = group_convolution(out, filter_size=kernel_size, strides=strides, num_filters=out_channels, groups=n_groups, kernel_initializer=kernel_initializer, name=name+'/group_conv{:02d}'.format(i) if name is not None else name)
            out = torch_prelu()(out)
    
    if building_block == BuildingBlocks.RRDB:
        for i in range(n_blocks):
            out = block_RRDB(out, kernel_size=kernel_size, depthwise_separable=depthwise_separable, weights_initializer=kernel_initializer, name=name+'/RRDB{:02d}'.format(i) if name is not None else name)
        
        if depthwise_separable:
            out = SeparableConv2D(out_channels, kernel_size=kernel_size, padding='same', depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer, name=name+'/conv_sep_out' if name is not None else name)(out)
        else:
            out = Conv2D(out_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_initializer, name=name+'/conv_out' if name is not None else name)(out)
        
    elif building_block == BuildingBlocks.IMDB:
        for i in range(n_blocks):
            out = block_IMDB(out, kernel_size=kernel_size, depthwise_separable=depthwise_separable, weights_initializer=kernel_initializer, name=name+'/IMDB{:02d}'.format(i) if name is not None else name)
        
        if depthwise_separable:
            out = SeparableConv2D(out_channels, kernel_size=kernel_size, padding='same', depthwise_initializer=kernel_initializer, pointwise_initializer=kernel_initializer, name=name+'/conv_sep_out' if name is not None else name)(out)
        else:
            out = Conv2D(out_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_initializer, name=name+'/conv_out' if name is not None else name)(out)
    
    blocks_needing_activation = [BuildingBlocks.RRDB, BuildingBlocks.IMDB]
    if building_block in blocks_needing_activation:
        out = torch_prelu()(out)

    return out

def reduction1x1_f(num_out_filters, max_depth, is_final=False, name=''):
    def geometric_transform(self, tensor):
        theta = self.sigmoid(tensor[:,:,:,0]) * tf.constant(math.pi /3.0)
        phi = self.sigmoid(tensor[:, :, :, 1]) * tf.constant(math.pi * 2.0)
        dist = self.sigmoid(tensor[:, :, :, 2]) * max_depth
        n1 = tf.expand_dims(tf.multiply(tf.math.sin(theta), tf.math.cos(phi)), axis = -1)
        n2 = tf.expand_dims(tf.multiply(tf.math.sin(theta), tf.math.sin(phi)), axis = -1)
        n3 = tf.expand_dims(tf.math.cos(theta), axis = -1)
        n4 = tf.expand_dims(dist, axis = -1)
        tensor = tf.concat([n1, n2, n3, n4], axis = -1)
        return tensor

    reduc = tf.keras.Sequential(name=name)

    while num_out_filters >= 4:
        if num_out_filters < 8:
            if is_final:
                reduc.add(tf.keras.Sequential([Conv2D(filters=1, kernel_size=1,use_bias=False, kernel_initializer='he_normal'),
                                                            torch_prelu()]))
            else:
                reduc.add(tf.keras.Sequential([Conv2D(filters=3, kernel_size=1, use_bias=False),
                                                    tf.keras.layers.Lambda(geometric_transform)]))
            break
        else:
            reduc.add(tf.keras.Sequential(
                [Conv2D(filters=num_out_filters, kernel_size=1, use_bias=False, kernel_initializer='he_normal'),
                torch_prelu()]
            ))

        num_in_filters = num_out_filters
        num_out_filters = num_out_filters // 2
    
    return reduc

def bts_functional(features, num_features = 128, max_depth=1.0, building_block='depthwise_sep', depthwise_separable=True, conv_block_chain_length=1, n_groups=8, output_activation=None, is_depth=True, as_model=True, name='BTS_decoder'):
    if building_block not in building_blocks_dic.keys():
        ValueError("Invalid building block {}, choose one of: {}".format(building_block, building_blocks_dic.keys()))
    building_block = building_blocks_dic[building_block]
    if output_activation not in output_activations_dic.keys():
        ValueError("Invalid output activation {}, choose from {}".format(output_activation, output_activations_dic.keys()))
    output_act = output_activations_dic[output_activation]
    
    if as_model:
        inputs = [tf.keras.Input(feature.shape[1:]) for feature in features]
    else:
        inputs = features

    #### DECODER SECTION
    ## Define network building blocks
    act = torch_prelu()

    upconv5 = upConv_f(num_features, kernel_size=3, building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='upconv_5')
    bn5 = standard_bn()
    
    conv5 = conv_f(num_features, kernel_size=3,  building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='conv_5')
    
    upconv4 = upConv_f(num_features // 2, kernel_size=3,  building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='upconv_4')
    bn4 = standard_bn()

    conv4 = conv_f(num_features//2, kernel_size=3, building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='conv_4')
    bn4_2 = standard_bn()
    # self.reduc8x8 = reduction1x1(num_features // 4, num_features // 4, self.max_depth)

    upconv3 = upConv_f(num_features // 4, kernel_size=3,  building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='upconv_3')
    bn3 = standard_bn()
    conv3 = conv_f(num_features // 4, kernel_size=3,  building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='conv_3')
    # self.reduc4x4 = reduction1x1(num_features // 4, num_features // 8, self.max_depth)

    upconv2 = upConv_f(num_features // 8, building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='upconv_2')
    bn2 = standard_bn()

    conv2 = conv_f(num_features // 8, kernel_size=3,  building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='conv_2')
    # self.reduc2x2 = reduction1x1(num_features // 8, num_features // 16, self.max_depth)

    upconv1 = upConv_f(num_features // 16, building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='upconv_1')
    reduc1x1 = reduction1x1_f(num_features // 32, max_depth, is_final=True, name='reduc1x1')

    conv1 = conv_f(num_features // 16, kernel_size=3, building_block=building_block, depthwise_separable=depthwise_separable, n_blocks=conv_block_chain_length, n_groups=n_groups, name='conv_1')

    # conv1X1 = Conv2D(num_features // 4, kernel_size=1, use_bias=False, name='conv1x1')
    conv1X1 = tf.keras.Sequential([
        Conv2D(num_features // 4, kernel_size=1, use_bias=False, name='conv1x1_conv', kernel_initializer='he_normal'),
        torch_prelu()], 
        name='conv1x1')

    if is_depth:
        get_output = Conv2D(1, kernel_size=1, padding='same', use_bias=False,
                                            name='depth_extractor')
    else:
        # Not used, but left for completeness
        get_output = Conv2D(11, kernel_size=3, padding='same', use_bias=False, name='segmentation_layer')


    ## Building network
    skip0, skip1, skip2, skip3 = inputs[1], inputs[2], inputs[3], inputs[4]
    dense_features = act(inputs[5])
    upconv5_o = upconv5(dense_features)  # H/16
    upconv5_o = bn5(upconv5_o)
    concat5_o = tf.concat([upconv5_o, skip3], axis=-1)
    iconv5_o = conv5(concat5_o)

    upconv4_o = upconv4(iconv5_o)  # H/8
    upconv4_o = bn4(upconv4_o)
    concat4_o = tf.concat([upconv4_o, skip2], axis=-1)
    iconv4_o = conv4(concat4_o)
    iconv4_o = bn4_2(iconv4_o)

    iconv4_o = conv1X1(iconv4_o)

    upconv3_o = upconv3(iconv4_o)  # H/4
    upconv3_o = bn3(upconv3_o)
    concat3_o = tf.concat([upconv3_o, skip1], axis=-1)

    iconv3_o = conv3(concat3_o)

    upconv2_o = upconv2(iconv3_o)  # H/2
    upconv2_o = bn2(upconv2_o)
    concat2_o = tf.concat([upconv2_o, skip0], axis=-1)
    iconv2_o = conv2(concat2_o)

    upconv1_o = upconv1(iconv2_o)
    reduc1x1_o = reduc1x1(upconv1_o)
    concat1_o = tf.concat([upconv1_o, reduc1x1_o], axis=-1)
    iconv1_o = conv1(concat1_o)

    output = get_output(iconv1_o)

    if output_act == OutputActivation.SIGMOID:
        output = tf.keras.layers.Lambda(lambda x: limited_logsigmoid_activation(x, max_depth), name="output_limited_logsigmoid") (output)

    if as_model:
        return tf.keras.Model(inputs, output, name=name)
    else:
        return output

def bts_model_f(input_shape=(224,224,3), alpha=0.75, num_features = 128, max_depth=1.0, minimalistic=False, output_activation=None,
                compile_strategy = None):
    encoder,_ = mobilenetV3encoder(input_shape=input_shape, alpha=alpha, minimalistic=minimalistic, name='mb3-encoder')
    decoder = bts_functional(encoder.outputs, num_features=num_features, max_depth=max_depth, output_activation=output_activation, name= 'bts-decoder')
    # decoder_out = bts_functional(encoder.outputs, num_features=num_features, max_depth=max_depth, name= 'bts-decoder')
    
    input_tensor = tf.keras.Input(input_shape, name="input_imgs")
    features = encoder(input_tensor)
    output = decoder(features)

    model = tf.keras.Model(inputs=input_tensor, outputs=output, name="bts_depth_predictor")

    # model = tf.keras.Model(inputs=encoder.input, outputs=decoder_out, name="mobile_bts_depth_predictor")

    if compile_strategy == 'md':
        compile_md(model)
    if compile_strategy == 'md_dd':
        compile_md_doubleDepth(model)
    elif compile_strategy == 'md_1cycle':
        compile_md_1cycle_lr(model)
    elif compile_strategy == 'md_cyclic':
        compile_md_cyclic_lr(model)
    elif compile_strategy == 'md_cyclic_halved':
        compile_md_cyclic_lr_halved(model)
    elif compile_strategy == 'md_poly_decay':
        compile_md_polynomial_decay(model)

    return model

def get_depthnet_with_downsizing(model, model_in_size=(128,160), io_size = (480,640)):
    input_tensor = tf.keras.Input([*io_size]+[3])
    x = tf.keras.layers.experimental.preprocessing.Resizing(
            model_in_size[0], model_in_size[1])(input_tensor)
    # x = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, -1.0)(x)
    x = model(x)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(2**16, 0.0)(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(io_size[0], io_size[1])(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x, name="bts_depth_predictor_downsized")

def compile_md_1cycle_lr(model, n_samples=96490, batch_size=64, epochs=30):
    # 96490 images
    # with 64 batch iterations: 96490 / 64 : 1508
    lr = OneCycleLR(tot_iter= math.ceil(n_samples / batch_size) * epochs,
                    lr_range=(1.5e-5, 5.0e-2),
                    step_size=20,
                    tail = 0.05,
                    annealing_lr_fraction=0.1)
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_cyclic_lr(model, n_samples=96490, batch_size = 64, epochs = 30):
    # 96490 images
    # with batch iterations: 96490 / bach_size 
    lr = CyclicLR(tot_iter= math.ceil(n_samples / batch_size) * epochs,
                lr_range=(1.5e-5, 5.0e-2),
                step_size=20,
                num_cycles=4
                )
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_cyclic_lr_halved(model, n_samples=96490, batch_size=64, epochs = 30):
    lr = CyclicLR(tot_iter=math.ceil(n_samples / batch_size) * epochs,
                lr_range=(1.5e-5, 5.0e-2),
                step_size=20,
                num_cycles=4,
                decay_func=lambda x, it: x * 2**(-it)
                )
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_polynomial_decay(model, n_samples=96490, batch_size = 64, epochs = 30):
    # 96490 images
    # with batch iterations: 96490 / bach_size 
    lr = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=5.0e-2,
                                                        decay_steps=math.ceil(n_samples / batch_size) * epochs,
                                                        end_learning_rate=1.5e-5,
                                                        power=1)
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# NOT FOR TRAINING. Only for learning rate search
def compile_md_lr_test_pow(model):
    lr = LearningRateSearchingSchedulePow(init_learning_rate=1.0e-6, end_learning_rate=0.5, steps=1500*4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# NOT FOR TRAINING. Only for learning rate search
def compile_md_lr_test_linear(model):
    lr = LearningRateSearchingScheduleLinear(init_learning_rate=1.0e-6, end_learning_rate=0.5, steps=1500*4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, rmse_nonlog]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_general(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = si_rmse
    metrics = [si_rmse, rmse_nonlog]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_parkmai(model, si_balance=0.85,
                    ssi_alpha=0.5, ssi_scales=4,
                    si_weight=2.0, ssi_weight=1.0, mse_weight=10.0,
                    lr_init=1.0e-3, num_steps=100000, lr_end=1.0e-6, lr_decay_pow=0.9,
                    adam_beta1=0.9, adam_beta2=0.999, adam_eps=1.0e-7):
    
    loss = parkmai_total_loss_func(si_balance=si_balance,
                                    ssi_alpha=ssi_alpha, ssi_scales=ssi_scales,
                                    si_weight=si_weight,
                                    ssi_weight=ssi_weight,
                                    mse_weight=mse_weight)
    metrics = [si_rmse, rmse_nonlog, ssi_loss_func(alpha=ssi_alpha, scales=ssi_scales)]

    # Optimizer
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                            initial_learning_rate= lr_init,
                            decay_steps=num_steps,
                            end_learning_rate= lr_end,
                            power= lr_decay_pow)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                        beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_eps)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
