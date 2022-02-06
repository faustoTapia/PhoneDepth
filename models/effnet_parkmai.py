import sys

# from tensorflow.python.util.nest import _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE
# sys.path.append('./')
from re import S
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.ops.gen_dataset_ops import skip_dataset_eager_fallback
from tensorflow.python.types.core import Value
import tensorflow_addons as tfa

from depth_utils import md_loss_total, si_rmse, rmse_nonlog, gradient_loss_multiscale_func, error_ordinal_wrap, parkmai_total_loss_func, ssi_loss_func
from .compileStrategies import compile_md, compile_md_doubleDepth
from .efficientNetEncoders import efficientNet_encoder
from .park_mai_depth import bts_functional
from .depthEncoder import DepthEncoder
from .operators import double_attention_module, global_non_local

def effiparkmai_net(version, input_shape=None, compile_strategy=None, enc_weights='imagenet', decoder_features=128,
                    dec_building_block='depthwise_sep', dec_depthwise_separable=True, dec_conv_block_chain_length=1, dec_n_groups=8,
                    output_activation=None,
                    separated_submodules=True, trainable_encoder=True):
    default_shapes=[224, 256, 256, 288, 384, 448, 512, 608]
    
    if input_shape is None:
        in_size = default_shapes[version]
        input_shape = (in_size, in_size, 3)

    print("Input shape: {}".format(input_shape))
    encoder = efficientNet_encoder(version, input_shape, weights=enc_weights, trainable=trainable_encoder)

    if separated_submodules:
        decoder = bts_functional(encoder.outputs, num_features=decoder_features,
                                building_block=dec_building_block, depthwise_separable=dec_depthwise_separable,
                                conv_block_chain_length=dec_conv_block_chain_length, n_groups=dec_n_groups,
                                output_activation=output_activation,
                                as_model=True)

        input_tensor = tf.keras.Input(shape=input_shape, name='input_img')
        features = encoder(input_tensor)
        output = decoder(features)

        model = tf.keras.Model(inputs=input_tensor, outputs=output, name="effiB{:d}parkmai_depth_predictor".format(version))
    else:
        decoder_outs = bts_functional(encoder.outputs, num_features=decoder_features,
                                      building_block=dec_building_block, depthwise_separable=dec_depthwise_separable,
                                      conv_block_chain_length=dec_conv_block_chain_length, n_groups=dec_n_groups,
                                      output_activation=output_activation,
                                      as_model=False)
        model = tf.keras.Model(inputs=encoder.inputs, outputs=decoder_outs, name="effiB{:d}parkmai_depth_predictor".format(version))

    if compile_strategy == 'md':
        compile_md(model)
    elif compile_strategy == 'md_dd':
        compile_md_doubleDepth(model)
    
    return model

def effi_dualatt_parkmai_net(version, input_shape=None, compile_strategy=None, enc_weights='imagenet', decoder_features=128, n_dualatt=5, max_depth=1.0,
                            dec_building_block='depthwise_sep', dec_depthwise_separable=True, dec_conv_block_chain_length=1, dec_n_groups=8,
                            output_activation=None, trainable_encoder=True):
    default_shapes=[224, 256, 256, 288, 384, 448, 512, 608]
    
    if input_shape is None:
        in_size = default_shapes[version]
        input_shape = (in_size, in_size, 3)

    print("Input shape: {}".format(input_shape))
    encoder = efficientNet_encoder(version, input_shape, weights=enc_weights, trainable=trainable_encoder)

    skip_connections = []
    for i, feature in enumerate(encoder.outputs):
        if i < len(encoder.outputs) - n_dualatt:
            skip_connections.append(feature)
        else:
            skip_connetion = double_attention_module(feature, residual=True, name="DAM_{}".format(i))
            skip_connections.append(skip_connetion)

    decoder_out = bts_functional(skip_connections, num_features=decoder_features,
                                building_block=dec_building_block, depthwise_separable=dec_depthwise_separable,
                                conv_block_chain_length=dec_conv_block_chain_length, n_groups=dec_n_groups,
                                output_activation=output_activation,
                                as_model=False)
    model = tf.keras.Model(inputs=encoder.inputs, outputs=decoder_out, name="effiB{:d}_dualatt_parkmai_depth_predictor".format(version))

    if compile_strategy == 'md':
        compile_md(model)
    elif compile_strategy == 'md_dd':
        compile_md_doubleDepth(model)
    
    return model

def effiDepth_dualatt_parkmai_net(version, input_size=None, compile_strategy=None, enc_weights='imagenet', decoder_features=128, n_dualatt=5, max_depth=1.0,
                            dec_building_block='depthwise_sep', dec_depthwise_separable=True, dec_conv_block_chain_length=1, dec_n_groups=8,
                            output_activation=None, trainable_encoder=True):
    default_shapes=[224, 256, 256, 288, 384, 448, 512, 608]
    
    if input_size is None:
        inp_size = default_shapes[version]
        input_size = (inp_size, inp_size)
        input_shape = input_size + (4, )
    elif len(input_size) != 2:
        raise ValueError('Provide input size of length 2')
    else:
        input_shape = input_size + (4,)

    inp_tensor = tf.keras.Input(shape=input_shape)
    img_tensor = inp_tensor[:,:,:,:-1]
    depth_tensor = tf.expand_dims(inp_tensor[:,:,:,-1], -1)

    print("Input shape: {}".format(input_shape))
    img_encoder_outs = efficientNet_encoder(version, input_shape=input_size+(3,), input_tensor=img_tensor, weights=enc_weights, trainable=trainable_encoder)
    depth_encoder_out = DepthEncoder(input_shape=input_size+(1,), input_tensor=depth_tensor, num_features=128, depthwise_separable=dec_depthwise_separable, name='DepthEncoder')

    skip_connections = []
    for i, (img_feature, depth_feature) in enumerate(zip(img_encoder_outs, depth_encoder_out)):
        if i < len(img_encoder_outs) - n_dualatt:
            skip_connection = tf.concat([img_feature, depth_feature], axis=-1)
            skip_connections.append(skip_connection)
        else:
            skip_connection = tf.concat([img_feature, depth_feature], axis=-1)
            skip_connetion = double_attention_module(skip_connection, residual=True, name="DAM_{}".format(i))
            skip_connections.append(skip_connetion)

    decoder_out = bts_functional(skip_connections, num_features=decoder_features,
                                building_block=dec_building_block, depthwise_separable=dec_depthwise_separable,
                                conv_block_chain_length=dec_conv_block_chain_length, n_groups=dec_n_groups,
                                output_activation=output_activation,
                                as_model=False)

    model = tf.keras.Model(inputs=inp_tensor, outputs=decoder_out, name="effiB{:d}_dualatt_parkmai_depth_predictor".format(version))

    if compile_strategy == 'md':
        compile_md(model)
    elif compile_strategy == 'md_dd':
        compile_md_doubleDepth(model)

    return model

def effi_globalNonLoc_parkmai_net(version, input_shape=None, compile_strategy=None, enc_weights='imagenet', decoder_features=128, n_glblNonLoc=2, trainable_encoder=True):
    default_shapes=[224, 256, 256, 288, 384, 448, 512, 608]
    
    if input_shape is None:
        in_size = default_shapes[version]
        input_shape = (in_size, in_size, 3)

    print("Input shape: {}".format(input_shape))
    encoder = efficientNet_encoder(version, input_shape, weights=enc_weights, trainable=trainable_encoder)

    skip_connections = []
    for i, feature in enumerate(encoder.outputs):
        if i < len(encoder.outputs) - n_glblNonLoc:
            skip_connections.append(feature)
        else:
            skip_connection = global_non_local(feature, c_neck_divisor=2, name="GNLoc_l{}".format(i))
            skip_connections.append(skip_connection)

    decoder_out = bts_functional(skip_connections, num_features=decoder_features, as_model=False)
    model = tf.keras.Model(inputs=encoder.inputs, outputs=decoder_out, name="effiB{:d}_dualatt_parkmai_depth_predictor".format(version))

    if compile_strategy == 'md':
        compile_md(model)
    if compile_strategy == 'md_dd':
        compile_md_doubleDepth(model)
    
    return model
