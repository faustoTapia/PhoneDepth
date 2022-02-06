import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from math import e, pi

# Jitter in image color properties given probability
def color_jitter(prob = 1.0, brightness=0.0, contrast=0.0, saturation=0.0, hue = 0.0, max_val=255.0):
    def col_jitter(img):
        if np.random.rand() < prob:
            out = tf.image.random_brightness(img, max_val * brightness)
            out = tf.image.random_contrast(out,
                max(0.0, 1.0-contrast), 1.0+contrast) if contrast>0.0 else out
            out = tf.image.random_saturation(out, 
                max(0.0, 1.0-saturation), 1.0+saturation) if saturation>0.0 else out
            out = tf.image.random_hue(out, hue)
            out = tf.clip_by_value(out, 0.0, max_val)
        else:
            out = img
        return out
    return col_jitter

# Appyl salt and pepper noise with probability prob and to px_fraction praction of pixels
def salty_noise(prob, px_fraction = 0.01):
    def salt_noise(x):
        if np.random.rand() < prob:
            tensor_shape = x.shape
            x = tf.cast(x, tf.float32)
            rnd = tf.random.uniform(x.shape)
            mask = rnd < px_fraction
            x = tf.where(mask, tf.random.uniform(tensor_shape)*255.0, x)
        return x
    return salt_noise

# Random rotation with probability prob and max angle angle_deg in degrees
def random_rotation(prob, angle_deg):
    def rand_rot(data_tuple):
        elems = list(data_tuple)
        if np.random.rand() < prob:
            angle = (tf.random.uniform([1]) - 0.5)*2.0 * angle_deg / 360.0 * 2 * pi
            for i in range(len(elems)):
                elems[i] = tfa.image.rotate(elems[i], angle)
        elems = tuple(elems)
        return elems

    return rand_rot

# Needs testing
def random_crop_and_resize(prob, min_size= 1.0, max_size=1.0, img_shape= (128,160), depth_shape=None, center_crop=False,  n_inps=1, n_imgs=1, conf_indx=-1):
    if depth_shape is None:
        depth_shape= img_shape
    if min_size > max_size:
        raise ValueError(f"min_size={min_size} !<= max_size={max_size}")
    if not (0.0<min_size<=1.0):
        raise ValueError(f"min_size={min_size} not in range (0,1]")
    if not(0.0 < max_size <=1.0):
        raise ValueError(f"max_size={max_size} not in range (0,1]")

    def rand_crop_resize(data_tuple):
        elems = list(data_tuple)
        width = elems[0].shape[1]
        height = elems[0].shape[0]
        scale = 1.0
        if np.random.rand() < prob:        
            scale = np.random.rand() * (max_size - min_size) + min_size
            crop_w = scale * width
            crop_h = scale * height
            if center_crop:
                crop_x = int((width - crop_w) // 2.0)
                crop_y = int((height - crop_h) // 2.0)
            else:
                crop_x = int(np.random.rand() * (width - crop_w))
                crop_y = int(np.random.rand() * (height - crop_h))
            crop_w  = int(crop_w)
            crop_h = int(crop_h)

            for i in range(len(elems)):
                elems[i] = elems[i][crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        for i in range(len(elems)):
            if i < n_inps:
                elems[i] = tf.image.resize(elems[i], img_shape)
            else:
                elems[i] = tf.image.resize(elems[i], depth_shape)
            if i > n_imgs - 1 and i != conf_indx:
                elems[i] = elems[i] * scale
        
        elems = tuple(elems)
        return elems

    return rand_crop_resize


def cascade_functions(funcs):
    def cascaded_func(x):
        for func in funcs:
            x = func(x)
        return x
    return cascaded_func

