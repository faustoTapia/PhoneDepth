import tensorflow as tf
from tensorflow.keras import layers
from .operators import block_IMDB

def name_from_parent(parent: str, current: str):
    out = parent + '/' + current if parent is not None else None
    return out


def deepening_block(x, filters, depthwise_separable=True, name=None):

    if depthwise_separable:
        out = layers.SeparableConv2D(filters, kernel_size=3, strides=1, padding='same', name=name_from_parent(name, 'sepConv'))(x)
    else:
        out = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', name=name_from_parent(name, 'conv'))(x)

    out = layers.AveragePooling2D(pool_size=(2,2), strides=2, padding='same', name = name_from_parent(name, 'pooling'))(out)
    out = block_IMDB(out, depthwise_separable=depthwise_separable, name=name_from_parent(name, 'imdb'))

    out = layers.BatchNormalization(name=name_from_parent(name, 'bn'))(out)
    
    return out


def DepthEncoder(input_shape=(320, 320, 1), input_tensor=None, num_features=128, depthwise_separable=True, name='DepthEncoder'):
    as_model = True

    if input_tensor is not None:
        if input_tensor.shape[1:] != list(input_shape):
            raise ValueError("Incoherent input shape and tensor provided")
        as_model = False

    else:
        input_tensor = tf.keras.Input(shape=input_shape)

    # Level1 out
    out0 = input_tensor

    out1 = deepening_block(out0, filters=num_features // (2**4), depthwise_separable=depthwise_separable, name=name_from_parent(name, 'DeepBlock1'))
    out2 = deepening_block(out1, filters=num_features // (2**3), depthwise_separable=depthwise_separable, name=name_from_parent(name, 'DeepBlock2'))
    out3 = deepening_block(out2, filters=num_features // (2**2), depthwise_separable=depthwise_separable, name=name_from_parent(name, 'DeepBlock3'))
    out4 = deepening_block(out3, filters=num_features // (2**1), depthwise_separable=depthwise_separable, name=name_from_parent(name, 'DeepBlock4'))
    out5 = deepening_block(out4, filters=num_features // (2**0), depthwise_separable=depthwise_separable, name=name_from_parent(name, 'DeepBlock5'))

    out_list = [out0, out1, out2, out3, out4, out5]

    if as_model:
        model = tf.keras.Model(input_tensor, out_list, name=name)
        return model
    else:
        return out_list
