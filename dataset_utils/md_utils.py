import random
import h5py
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
from os import path

default_md_dir = '/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1'

def load_data_list_md(list_dir, md_dir= 'C:/ml/megadeep/MegaDepth_v1', depth_file_type='h5'):
    base_dir = md_dir + '/phoenix/S6/zl548/MegaDepth_v1/'

    Images_file_name = list_dir + "/imgs_MD.p"

    file = open(Images_file_name, "rb" )
    image_list = pickle.load(file)
    image_list = [base_dir + x for x in image_list]

    if depth_file_type == 'h5':
        depth_list_file = '/targets_MD.p'
    elif depth_file_type == 'npy':
        depth_list_file = '/targets_MD_npy.p'
    elif depth_file_type == 'png':
        depth_list_file = '/targets_MD_png.p'
    else:
        raise ValueError("Invalid depth_file_type, must be either of: 'h5', 'npy', 'png'")

    targets_file_name = list_dir + depth_list_file
    file = open(targets_file_name, "rb" )
    target_list = pickle.load(file)
    target_list = [base_dir + x for x in target_list]
    file.close()
    return (image_list, target_list)

# @ tf.py_function
def load_depth(file, size=None):
    # Taking care of tensorflow string
    if not isinstance(file, str):
        file = file.numpy().decode('utf-8')
    
    hdf5_file_read = h5py.File(file, 'r')
    gt = hdf5_file_read.get('/depth')
    gt = np.array(gt, dtype=np.float32)

    # Keep ordinal images unscaled (scale important in thresholds when determining ordinal loss)
    if np.min(gt) > -0.1:           # Should be -1 for ordinal images
        # Setting invalid values to 0
        gt[gt > np.percentile(gt[gt > 1e-8], 98)] = 0
        gt[gt < np.percentile(gt[gt>1e-8], 1)] = 0

        # max_depth = np.max(gt)
        # gt = gt/max_depth

    gt = tf.convert_to_tensor(gt, tf.float32)
    gt = tf.expand_dims(gt, -1)
    if size is not None:
        gt = tf.image.resize(gt, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    hdf5_file_read.close()

    return gt

def load_depth_npy(file, size=None):
    # Taking care of tensorflow string
    if not isinstance(file, str):
        file = file.numpy().decode('utf-8')

    gt = np.load(file, allow_pickle=True)

    # Keep ordinal images unscaled (scale important in thresholds when determining ordinal loss)
    if np.min(gt) > -0.1:           # Should be -1 for ordinal images
        # Setting invalid values to 0
        gt[gt > np.percentile(gt[gt > 1e-8], 98)] = 0
        gt[gt < np.percentile(gt[gt>1e-8], 1)] = 0

    gt = tf.convert_to_tensor(gt, tf.float32)
    gt = tf.expand_dims(gt, -1)
    if size is not None:
        gt = tf.image.resize(gt, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return gt

def load_depth_png(file, size=None):

    depth = tf.io.read_file(file)
    depth = tf.io.decode_image(depth, channels=1, dtype=tf.uint16, expand_animations=False)
    depth = tf.cast(depth, tf.float32)

    # Encoding convention
    max_val = tf.math.reduce_max(depth)
    if tf.reduce_max(depth) > 2**16-2:
        # Ordinal case
        value_mask = depth < 2**16 - 1
        depth = tf.where(value_mask, depth, -1.0)
    else:
        mask_upper = depth < tfp.stats.percentile(depth, 98)
        mask_lower = depth > tfp.stats.percentile(depth, 1)
        mask = tf.logical_and(mask_upper, mask_lower)

        depth = tf.where(mask, depth, 0)
        
        max_val = tf.math.reduce_max(depth)
        depth = tf.divide(depth, max_val)

    depth = tf.image.resize(depth, size, method='nearest')

    return depth

def load_depth_wrapped(file, size):
    gt_tf = tf.py_function(load_depth, [file, size], tf.float32)
    gt_tf.set_shape((size[0], size[1],1))
    return gt_tf

def load_img(file, size=None):
    image = tf.io.read_file(file)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    if size is not None:
        image = tf.image.resize(image, tf.convert_to_tensor(size, tf.int32))
    return image

def prep_data_train(load_img_func, load_depth_func, in_size, out_size=None, py_func=True, random_flip=False,
                    in_transform=None, out_transform=None, io_transform = None):
    if out_size is None:
        out_size = in_size
    def func(img_file, depth_file):
        img = load_img_func(img_file, in_size)

        # Using py_function when necesary for reading files
        if py_func:
            [depth,] = tf.py_function(load_depth_func, [depth_file, out_size], [tf.float32])
            depth.set_shape((out_size[0], out_size[1], 1))
        else:
            depth = load_depth_func(depth_file, in_size)

        if io_transform is not None:
            img, depth = io_transform((img, depth))
        if in_transform is not None:
            img = in_transform(img)
        if out_transform is not None:
            depth = out_transform(depth)

        # Flipping
        if random_flip:
            flip = tf.random.uniform([1]) < 0.5
            img = tf.cond(flip[0],
                    true_fn=lambda: tf.image.flip_left_right(img),
                    false_fn=lambda: img)
            depth = tf.cond(flip[0],
                        true_fn = lambda: tf.image.flip_left_right(depth),
                        false_fn = lambda: depth)

        return (img,depth)
    return func


def prep_data_eval(load_img_func, load_depth_func, in_size, py_func=True,
                    in_transform=None, out_transform=None, io_transform = None):
    
    def func(img_file, depth_file):
        img = load_img_func(img_file, in_size)
        
        if py_func:
            [depth,] = tf.py_function(load_depth_func, [depth_file], [tf.float32])
            depth.set_shape((None, None, 1))
        else:
            depth = load_depth_func(depth_file, None)

        if io_transform is not None:
            img, depth = io_transform((img, depth))
        if in_transform is not None:
            img = in_transform(img)
        if out_transform is not None:
            depth = out_transform(depth)

        return (img, depth)
    
    return func


def merge_md_data_lists(data_lists):
    inputs = []
    targets = []
    for data_list in data_lists:
        inputs+=data_list[0]
        targets+=data_list[1]
    return (inputs, targets)

def md_dataset(dataset_dir, partition_list='final_list', mode='train', img_type='all', depth_type='h5', input_shape=(224,224), output_shape=None, evaluation=False,
                batch_size=8, random_flip=False, n_images=None, start_img=0,
                in_transform=None, out_transform=None, io_transform=None, shuffle=True, num_parallel_calls=tf.data.AUTOTUNE):
    'Note that transform must be only a topological transform for the input image, do not change geometry as target will not be changed'
    output_shape = input_shape if output_shape is None else output_shape
    buffer_size = 100000

    if mode == 'train':
        landscape_list = load_data_list_md(dataset_dir + '/' + partition_list + '/train_list/landscape', md_dir=dataset_dir, depth_file_type=depth_type)
        portrait_list = load_data_list_md(dataset_dir + '/' + partition_list + '/train_list/portrait/', md_dir=dataset_dir, depth_file_type=depth_type)

    elif mode == 'val':
        landscape_list = load_data_list_md(dataset_dir + '/' + partition_list + '/val_list/landscape', md_dir=dataset_dir, depth_file_type=depth_type)
        portrait_list = load_data_list_md(dataset_dir + '/' + partition_list + '/val_list/portrait/', md_dir=dataset_dir, depth_file_type=depth_type)
        
    elif mode == 'test':
        landscape_list = load_data_list_md(dataset_dir + '/' + partition_list + '/test_shuffled/landscape', md_dir=dataset_dir, depth_file_type=depth_type)
        portrait_list = load_data_list_md(dataset_dir + '/' + partition_list + '/test_shuffled/portrait/', md_dir=dataset_dir, depth_file_type=depth_type)
 
    else:
        raise(ValueError("MAI Generator mode can only be: 'train', 'val' or 'test'"))
        
    if img_type == 'landscape':
        list_data_files = landscape_list
    elif img_type == 'portrait':
        list_data_files = portrait_list
    elif img_type == 'all':
        list_data_files = merge_md_data_lists([landscape_list, portrait_list])
    else:
        raise ValueError("Invalid img_type for megadepth dataset. Can only be: 'landscape', 'portrait', 'all'")

    if n_images is not None and n_images + start_img < len(list_data_files[0]):
        data_lists = []
        for i in range(len(list_data_files)):
            data_lists.append(list_data_files[i][start_img: start_img + n_images])
        data_lists = tuple(data_lists)
    else:
        data_lists = list_data_files

    dataset = tf.data.Dataset.from_tensor_slices(data_lists)
    if shuffle:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    print(f"Dataset with {len(data_lists[0])} elements")

    need_py_func = True
    if depth_type=='h5':
        load_depth_func = load_depth
    elif depth_type=='npy':
        load_depth_func = load_depth_npy
    elif depth_type=='png':
        load_depth_func = load_depth_png
        need_py_func = False
    else:
        raise ValueError("MD dataset currently only supports 'h5' and 'npy' depth files.")
    
    print(f"Dataset with depth_type {depth_type}")

    if not evaluation:
        dataset = dataset.map(prep_data_train(load_img, load_depth_func=load_depth_func, in_size=input_shape, out_size=output_shape, py_func=need_py_func,
                                random_flip=random_flip, in_transform=in_transform, out_transform=out_transform, io_transform=io_transform),
                                num_parallel_calls=num_parallel_calls, deterministic=not shuffle)
    else:
        dataset = dataset.map(prep_data_eval(load_img, load_depth_func=load_depth_func, in_size=input_shape, py_func=need_py_func, in_transform=in_transform),
                                num_parallel_calls=num_parallel_calls, deterministic=not shuffle)
        if batch_size != 1:
            print("In evaluation must have batch_size of 1 as multiple image sizes allowed. So changing it to 1")
            batch_size = 1
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset
