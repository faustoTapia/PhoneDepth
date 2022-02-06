import os
import math
from tensorflow.keras.applications import mobilenet
from tensorflow.keras import layers, Model, Sequential
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import collections
import math

#import tensorflow_probability as tfp
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        exit(-1)

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, layers.Conv2D):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, layers.DepthwiseConv2D):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, layers.BatchNormalization):
        m.weight.data.fill_(1)
        m.bias.data.zero_()        
        
        
def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return  tf.keras.Sequential([
          layers.DepthwiseConv2D(kernel_size=kernel_size,strides=1, padding='SAME', use_bias=False),
          layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
          layers.ReLU()]
        )

def pointwise(in_channels, out_channels):
    return  tf.keras.Sequential([
          layers.Conv2D(filters=out_channels, kernel_size=1,strides=1, padding='SAME', use_bias=False),
          layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
          layers.ReLU()]
        )



class FastDepth(tf.keras.Model):
    def __init__(self, input_size=(480, 640), trainable_encoder=True, include_features=False):

        super(FastDepth, self).__init__()
        #self._set_inputs(tf.TensorSpec([None,480,640,3],tf.float32,name='inputs'))
        self.input_size=input_size
        self._include_features = include_features
        mymobilenet = mobilenet.MobileNet(input_shape=input_size + (3,), weights='imagenet',include_top=False)
        #mobilenet.load_weights('model_best.pth.tar')
        for i in range(86):
            layer = mymobilenet.get_layer(index=i)
            layer.trainable = trainable_encoder
            setattr( self, 'layer{}'.format(i), layer)

        kernel_size = 5
        self.decode_conv1 = tf.keras.Sequential([
            depthwise(1024, kernel_size),
            pointwise(1024, 512)])
        self.decode_conv2 = tf.keras.Sequential([
            depthwise(512, kernel_size),
            pointwise(512, 256)])
        self.decode_conv3 = tf.keras.Sequential([
            depthwise(256, kernel_size),
            pointwise(256, 128)])
        self.decode_conv4 = tf.keras.Sequential([
            depthwise(128, kernel_size),
            pointwise(128, 64)])
        self.decode_conv5 = tf.keras.Sequential([
            depthwise(64, kernel_size),
            pointwise(64, 32)])
        self.decode_conv6 = pointwise(32, 1)
        # self.out=tf.keras.layers.ReLU()
        self.out=tf.keras.activations.linear
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def call(self, x):
        feat=[]
        # Normalize in the tensorflow manner
        uu = x / 127.5 - 1.0
        x = uu

        for i in range(86):
            layer = getattr(self, 'layer{}'.format(i))
            x = layer(x)
            #print("{}: {}".format(i, x.shape))
            if i==9:

                x1 = x
                #print(x1.shape)
            elif i==22:
                x2 = x
                feat.append(x)
            elif i==35:
                x3 = x
                feat.append(x)
            elif i==72:
                feat.append(x)
            elif i==85:
                feat.append(x)

        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            #x = layers.UpSampling2D(size=(2, 2))(x)
            if i==1:
                x=tf.image.resize(x, tuple(dim//16 for dim in self.input_size), method=tf.image.ResizeMethod.BILINEAR)
            if i==2:
                x=tf.image.resize(x, tuple(dim//8 for dim in self.input_size),method=tf.image.ResizeMethod.BILINEAR)
                x = x + x3
            if i==3:
                x=tf.image.resize(x, tuple(dim//4 for dim in self.input_size), method=tf.image.ResizeMethod.BILINEAR)
                x = x + x2
            if i==4:
                x=tf.image.resize(x, tuple(dim//2 for dim in self.input_size),method=tf.image.ResizeMethod.BILINEAR)
                x = x + x1
            if i==5:
                x=tf.image.resize(x, self.input_size,method=tf.image.ResizeMethod.BILINEAR)

        x = self.decode_conv6(x)
        x=self.out(x)

        if self._include_features:
            return x, feat
        else:
            return x

if __name__=="__main__":
    input_size = (224, 224)
    model = FastDepth(input_size=input_size, trainable_encoder=False)
    inp = tf.keras.Input(shape=input_size + (3,))
    out = model(inp)
    print("Finished")