from enum import Enum
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.activations import sigmoid
from tensorflow import image
import numpy as np

def layer_name_from_parent(parent_name: str, name: str)-> str:
    return parent_name + "/" + name if parent_name is not None else None

def global_non_local(X, c_neck_divisor=2, name=None):
    tensor_shape = list(X.shape)
    h, w , c = tensor_shape[1], tensor_shape[2], tensor_shape[3]
    c_neck = c // c_neck_divisor

    theta = tf.keras.layers.Conv2D(c_neck, kernel_size=(1,1), padding='same', name=layer_name_from_parent(name, 'ConvTheta'))(X)
    theta_rsh =  tf.keras.layers.Reshape((h*w, c_neck), name=layer_name_from_parent(name, 'ReshapeTheta'))(theta)

    phi = tf.keras.layers.Conv2D(c_neck, kernel_size=(1,1), padding='same', name=layer_name_from_parent(name, 'ConvPhi'))(X)
    phi_rsh = tf.keras.layers.Reshape((c_neck, h*w), name=layer_name_from_parent(name, 'ReshapePhi'))(phi)

    g = tf.keras.layers.Conv2D(c_neck, kernel_size=(1,1), padding='same', name=layer_name_from_parent(name, 'ConvG'))(X)
    g_rsh = tf.keras.layers.Reshape((h*w, c_neck), name=layer_name_from_parent(name, 'ReshapeG'))(g)

    theta_phi = tf.matmul(theta_rsh, phi_rsh)
    theta_phi = tf.keras.layers.Softmax()(theta_phi)

    theta_phi_g = tf.matmul(theta_phi, g_rsh)
    theta_phi_g = tf.keras.layers.Reshape((h, w, c_neck))(theta_phi_g)

    theta_phi_g = tf.keras.layers.Conv2D(c, kernel_size=(1,1), padding='same', name=layer_name_from_parent(name, 'ConvThetaPhi'))(theta_phi_g)

    out = tf.keras.layers.Add(name=layer_name_from_parent(name, "outAdd"))([theta_phi_g, X])
    return out


def gated_attention_module(X, gate, hid_filts, kernel_size=1, single_channel_attention=True, depthwise_separable=False, resize_method='bilinear', name=None):
    inp_shape = tf.shape(X)
    gate_shape = tf.shape(gate)

    ConvLayer = layers.Conv2D if not depthwise_separable else layers.SeparableConv2D
    
    Xp = ConvLayer(filters=hid_filts, kernel_size=kernel_size, padding='same', name=layer_name_from_parent(name, 'convX'))(X)
    Xp = image.resize(Xp, gate_shape[1:3], method=resize_method)

    gp = ConvLayer(filters=hid_filts, kernel_size=kernel_size, padding='same', name=layer_name_from_parent(name, 'convG'))(gate)
    # gp = image.resize(gp, inp_shape[1:3], method='bilinear')

    psi = layers.LeakyReLU(alpha=0.2)(layers.Add()([Xp, gp]))
    psi_filts = 1 if single_channel_attention else inp_shape[3]
    psi = ConvLayer(filters=psi_filts, kernel_size=kernel_size, padding='same', name=layer_name_from_parent(name, 'convSig'))(psi)
    psi = sigmoid(psi)
    
    alpha = image.resize(psi, inp_shape[1:3], method=resize_method)
    out = tf.math.multiply(X, alpha)

    return out

def spatial_attention_module(x, kernel_size=5, dilation_rate=2, name=None):
    out = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=1, padding='same', dilation_rate=dilation_rate, name=layer_name_from_parent(name, 'DepthwiseConv'))(x)
    out = layers.Lambda(sigmoid, name=layer_name_from_parent(name, "/Sigmoid")) (out)
    
    out = out * x

    return out

def channel_attention_module(x, bottle_neck_reduct=1, name=None):
    channels = x.shape[3]
    
    out = tf.reduce_mean(x, axis=[1,2], keepdims=True, name=layer_name_from_parent(name, "AvgPool"))
    out = layers.Conv2D(filters=channels//bottle_neck_reduct, kernel_size=1, padding="same", name=layer_name_from_parent(name, "Conv1x1_1"))(out)
    out = layers.ReLU(name=name+"/Relu_1" if name is not None else None)(out)
    out = layers.Conv2D(filters=channels, kernel_size=1, padding='same', name=layer_name_from_parent(name, "Conv1x1_2"))(out)
    out = layers.Lambda(sigmoid, name=layer_name_from_parent(name, "Sigmoid")) (out)

    out = out * x

    return out

def double_attention_module(x, channel_reduction=4, residual=True, name=None):
    channels = x.shape[3]

    out = layers.Conv2D(filters=channels*2, kernel_size=3, padding='same', name=layer_name_from_parent(name, "Conv_0"))(x)
    out = layers.ReLU(name=name+"/Relu_0") (out)
    out = layers.Conv2D(filters=channels*2, kernel_size=1, padding='same', name=layer_name_from_parent(name, "Conv1x1_0"))(out)
    
    channel_att = channel_attention_module(out, bottle_neck_reduct=channel_reduction, name=layer_name_from_parent(name, "CAM"))
    spatial_att = spatial_attention_module(out, name=layer_name_from_parent(name, "SAM"))

    out = tf.concat([spatial_att, channel_att], axis=3, name=layer_name_from_parent(name, "concat"))
    out = layers.Conv2D(filters=channels, kernel_size=1, padding='same', name=layer_name_from_parent(name, "Conv1x1_out"))(out)
    
    if residual:
        out = out + x

    return out


def group_convolution(inp, num_filters, filter_size, strides, dilation_rate=1, groups=8, padding='same', kernel_initializer='glorot_uniform', name=None):
    # splits = tf.split(inp, groups, axis=3, name=name+"/split" if name is not None else None)
    _, _, _, c = inp.shape
    
    inp_split_size = c // groups
    group_filters = num_filters // groups

    if c < groups:
        raise ValueError("Number of input channels less than number of groups: {} < {}".format(c, groups))
    remainder_channels = c % groups
    if remainder_channels != 0:
        print("WARNING: Input channels not divisible by grups, mod({}, {}) != 0, so groups are not all of the same number of channels".format(c, groups))
        # This makes last group to be incomplete but the same number of groups are used
    if num_filters % groups != 0:
        raise ValueError("Number of filters must be divisible by number of groups, mod({}, {}) !=  0".format(num_filters, groups))

    # Dealing with cases when input channels are not divisible by number of groups
    inp_group_channels = [inp_split_size] * groups
    for i in range(remainder_channels):
        inp_group_channels[i] += 1
    inp_indeces = [0] + list(np.cumsum(inp_group_channels))

    splits = [inp[:, :, :, inp_indeces[group] : inp_indeces[group + 1]] for group in range(groups)]

    splits = [layers.Conv2D(group_filters, kernel_size=filter_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
                kernel_initializer=kernel_initializer, name=name+"/Conv{:02d}".format(i) if name is not None else None)(x) for i, x in enumerate(splits)]
    # out = tf.concat(splits, axis=3, name=name+"/concat" if name is not None else None)
    out = layers.Concatenate(axis=-1, name=name+"/concat" if name is not None else None)(splits)
    out = layers.Conv2D(num_filters, kernel_size=1, kernel_initializer=kernel_initializer, name=name+"/pointwise_conv" if name is not None else None)(out)

    return out


def block_DENSE(x, filters, bottle_neck_factor=2, kernel_size=3, depthwise_separable=False, growth_rate=4, residual_type='add', weights_initializer='he_normal', name=None):
    c = x.shape[3]
    bottle_neck_filters = int(c // bottle_neck_factor)
    if bottle_neck_filters <=0:
        bottle_neck_filters = c

    if depthwise_separable:
        internal_inp = layers.SeparableConv2D(bottle_neck_filters, kernel_size, padding='same', depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                            name=name+"/depthsepConv_inp" if name is not None else name)(x)
    else:
        internal_inp = layers.Conv2D(bottle_neck_filters, kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/Conv_inp" if name is not None else name)(x)
    internal_inp = layers.LeakyReLU(name=name + "/activation_inp" if name is not None else name)(internal_inp)

    if residual_type == 'concat':
        jump_connections = [internal_inp]
    elif residual_type == 'add':
        jump_connections = internal_inp
    else:
        ValueError("Residual type can only be either of ['add', 'concat'], given {}".format(residual_type))

    for i in range(growth_rate):
        if residual_type == 'concat':
            if len(jump_connections) > 1:
                curr_inp = layers.Concatenate(axis=-1, name=name+"/concat{}".format(i) if name is not None else name)(jump_connections)
                # curr_inp = tf.concat(jump_connections, axis=-1, name=name+"/concat{}".format(i) if name is not None else name)
            else:
                curr_inp = jump_connections[0]
        else:
            curr_inp = jump_connections
        if depthwise_separable:
            curr_out = layers.SeparableConv2D(bottle_neck_filters, kernel_size, padding='same', depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                              name=name+"/depthsepConv{}".format(i) if name is not None else name)(curr_inp)
        else:
            curr_out = layers.Conv2D(bottle_neck_filters, kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/Conv{}".format(i) if name is not None else name)(curr_inp)
        curr_out = layers.LeakyReLU(name=name + "/activation{}".format(i) if name is not None else name)(curr_out)

        if residual_type == 'concat':
            jump_connections += [curr_out]
        else:
            jump_connections += curr_out
    if residual_type == 'concat':
        out = layers.Concatenate(axis=-1, name=name + "/out_concat" if name is not None else name)(jump_connections) 
        # out = tf.concat(jump_connections, axis=-1, name=name + "/out_concat" if name is not None else name)
    else:
        out = jump_connections
    
    if depthwise_separable:
        out = layers.SeparableConv2D(filters, kernel_size, padding='same', depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                     name=name+"/depthsepConv_out" if name is not None else name) (out)
    else:
        out = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/Conv_out" if name is not None else name) (out)

    return out

def block_RRDB(x, kernel_size=3, beta=0.3, depthwise_separable=False, n_dense_blocks=2, dense_bottleneck_factor=8, dense_growth_rate=3, dense_residual_type='add', weights_initializer='he_normal', name=None):
    c = x.shape[3]
    curr_out = x
    for i in range(n_dense_blocks):
        dense_out = block_DENSE(curr_out, c, bottle_neck_factor=dense_bottleneck_factor, kernel_size=kernel_size, depthwise_separable=depthwise_separable,
                                growth_rate=dense_growth_rate, residual_type=dense_residual_type, weights_initializer=weights_initializer, name=name + "/Denseblock{}".format(i))
        curr_out = dense_out * beta + curr_out
    
    curr_out = curr_out * beta + x

    return curr_out

def block_IMDB(x, kernel_size=3, depthwise_separable=False, weights_initializer='he_normal', num_splits=3, name=None):
    tot_channels = x.shape[-1]
    if tot_channels < 2**num_splits:
        raise ValueError("tot_channels must be greater or equal to 2**num_splits, {}!>={}".format(tot_channels, 2**num_splits))
    if tot_channels % 2**num_splits != 0:
        print("WARNING: Number of input channels odd, so Uneven splits. {}%{}!=0".format(tot_channels, 2**num_splits))
    if not depthwise_separable:
        curr_inp = layers.Conv2D(tot_channels, kernel_size=kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/initConv" if name is not None else name)(x)
    else:
        curr_inp = layers.SeparableConv2D(tot_channels, kernel_size=kernel_size, padding='same',
                                          depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                          name=name+"/initSepConv" if name is not None else name)(x)
    curr_inp = layers.LeakyReLU(name=name+"/initActivation" if name is not None else name)(curr_inp)

    cum_concats = []
    for i in range(num_splits):
        curr_channels = curr_inp.shape[-1]
        middle = curr_channels // 2 
        cum_concats += [curr_inp[:,:,:,:middle]]
        new_branch = curr_inp[:,:,:, middle:]
        if not depthwise_separable:
            curr_inp = layers.Conv2D(middle, kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/splitConv{}".format(i) if name is not None else name)(new_branch)
            # curr_inp = layers.Conv2D(curr_channels - middle, kernel_size, padding='same', kernel_initializer=weights_initializer, name=name+"/splitConv{}".format(i) if name is not None else name)(new_branch)
        else:
            curr_inp = layers.SeparableConv2D(curr_channels - middle, kernel_size, padding='same',
                                              depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                              name=name+"/splitSepConv{}".format(i) if name is not None else name)(new_branch)
        curr_inp = layers.LeakyReLU(name=name+"/activation{}".format(i) if name is not None else name)(curr_inp)
    
    cum_concats += [curr_inp]
    concat_branches = layers.Concatenate(axis=-1, name = name+"/JoinConcat" if name is not None else name)(cum_concats)
    # concat_branches = tf.concat(cum_concats, axis=-1, name = name+"/JoinConcat" if name is not None else name)

    if not depthwise_separable:
        out = layers.Conv2D(tot_channels, kernel_size,  kernel_initializer=weights_initializer, padding='same', name=name+'/outConv' if name is not None else name)(concat_branches)
    else:
        out = layers.SeparableConv2D(tot_channels, kernel_size, depthwise_initializer=weights_initializer, pointwise_initializer=weights_initializer,
                                     padding='same', name=name+'/outSepConv' if name is not None else name)(concat_branches)

    out = out + x

    return out


def limited_logsigmoid_activation(x, max_val, min_cutoff=1e-10):
    x = tf.keras.activations.sigmoid(x)
    x = - tf.math.divide(1.0, x) + 1.0 + tf.math.log(max_val)
    x = tf.clip_by_value(x, tf.math.log(min_cutoff), float('inf'))
    return x

class BuildingBlocks(Enum):
    DEPTHWISE_SEPARABLE = 0
    GROUP_CONVOLUTIONS = 1
    RRDB = 2
    IMDB = 3

building_blocks_dic = {'depthwise_sep': BuildingBlocks.DEPTHWISE_SEPARABLE,
                    'group_convs': BuildingBlocks.GROUP_CONVOLUTIONS,
                    'rrdb': BuildingBlocks.RRDB,
                    'imdb': BuildingBlocks.IMDB}

class OutputActivation(Enum):
    LINEAR = 0
    SIGMOID = 1
    HTAN = 2

output_activations_dic = {None: OutputActivation.LINEAR,
                            "sigmoid": OutputActivation.SIGMOID,
                            "htan": OutputActivation.HTAN}


if __name__=="__main__":
    input = keras.Input((20,20,320))
    depthwise_sep = True
    residual_type = 'concat'

    out_dense = block_DENSE(input, 20,  name="test_dense")

    model = tf.keras.Model(inputs=input, outputs=out_dense)

    out_rrdb = block_RRDB(input,depthwise_separable=depthwise_sep, dense_residual_type=residual_type, name="test_rrdb")
    model2 = tf.keras.Model(inputs=input, outputs=out_rrdb)

    out_imdb = block_IMDB(input, depthwise_separable=depthwise_sep, name='IMDB_test')
    model3 = tf.keras.Model(inputs=input, outputs=out_imdb)
    

    print("Done")
