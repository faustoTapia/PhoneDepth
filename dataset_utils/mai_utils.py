import random
import pickle
import numpy as np
import time

import tensorflow as tf
from os import listdir, path
import os
from PIL import Image

def load_data_list(mai_dir= '/data/MAI/'):
    base_dir = mai_dir + '/rgb/'

    img_names = [img for img in listdir(base_dir) if path.isfile(path.join(base_dir, img))]

    img_list = [path.join('rgb', img) for img in img_names]
    target_list = [path.join('depth', img) for img in img_names]
    return list(zip(img_list, target_list))

def load_data_list_from_file(partition_file):
    base_dir = path.dirname(partition_file)
    with open(partition_file, 'rb') as f:
        train_data_list = np.load(f, allow_pickle=True)
    imgs = [path.join(base_dir, train_point[0]) for train_point in train_data_list]
    targets = [path.join(base_dir, train_point[1]) for train_point in train_data_list]
    return (imgs, targets)

def load_data_list_test(base_dir):
    img_names = [path.join(base_dir, img) for img in listdir(base_dir) 
        if (path.isfile(path.join(base_dir, img)) and (img.split('.')[-1]=='png' or img.split('.')[-1]=='jpg'))]
    return img_names
    

def generate_partition_files(mai_dir='/data/MAI/', validation_fraction=0.1, train_file='train.npy', val_file='val.npy'):
    data = load_data_list(mai_dir)
    random.seed(67)
    indeces = list(range(len(data)))
    random.shuffle(indeces)

    separator_indx = int((1-validation_fraction)*len(data))
    train_indcs = indeces[0:separator_indx]
    val_indcs = indeces[separator_indx:]

    train_data  = [(data[i][0], data[i][1]) for i in train_indcs]
    val_data    = [(data[i][0], data[i][1]) for i in val_indcs]

    file_to_save = path.join(mai_dir, train_file)
    with open(file_to_save, 'wb') as f:
        np.save(f, train_data)
    print(f'Saved : {file_to_save}')
    file_to_save = path.join(mai_dir, val_file)
    with open(file_to_save, 'wb') as f:
        np.save(f, val_data)
    print(f'Saved: {file_to_save}')


def load_img(file,  size=None):
    image = tf.io.read_file(file)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    if size is not None:
        image = tf.image.resize(image, tf.convert_to_tensor(size, tf.int32))
    return image

def load_depth(file, size=None):
    depth_img = tf.io.read_file(file)
    depth_img = tf.io.decode_png(depth_img, channels=1, dtype=tf.uint16)
    depth_img = tf.cast(depth_img, tf.float32)
    if size is not None:
        depth_img = tf.image.resize(depth_img, tf.convert_to_tensor(size, tf.int32), method='nearest')
    depth_img = depth_img
    return depth_img


def prep_data_train(load_img, load_depth, input_shape=None, random_flip=False, in_transform=None, out_transform=None, io_transform = None):

    def func(img_file, depth_file):
        img = load_img(img_file, input_shape)
        depth = load_depth(depth_file, input_shape)


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

        return img,depth
    return func

def mai_dataset(mai_dir, mode='train', input_shape=(480,640), batch_size=32, random_flip=False, n_images=None, in_transform=None, out_transform=None, io_transform = None, shuffle=True, num_parallel_calls=tf.data.AUTOTUNE):
    'Note that transform must be only a topological transform for the input image, do not change geometry as target will not be changed'
    buffer_size = 96492

    if mode == 'train':
        data_list_whole = load_data_list_from_file(path.join(mai_dir, 'train', 'train.npy'))
    elif mode == 'val':
        data_list_whole = load_data_list_from_file(path.join(mai_dir, 'train', 'val.npy'))
    elif mode == 'test':
        data_list_whole = load_data_list_from_file(path.join(mai_dir, 'train', 'val.npy'))
    else:
        raise(ValueError('MAI Generator mode can only be: train or val. As test data only provided for these splits.'))

    # Picking only first n_images
    if n_images and n_images <= len(data_list_whole[0]):
        data_list= []
        for i in range(len(data_list_whole)):
            data_list.append(data_list_whole[i][:n_images])
        data_list = tuple(data_list)
    else:
        data_list = data_list_whole
    print(f"Dataset with {len(data_list[0])} elements.")

    dataset = dataset_from_list_shuffle(data_list, shuffle, buffer_size)
    dataset = dataset.map(prep_data_train(load_img, load_depth, input_shape=input_shape, random_flip=random_flip, in_transform=in_transform, out_transform=out_transform, io_transform=io_transform),
                            num_parallel_calls=num_parallel_calls, deterministic=not shuffle)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def dataset_from_list_shuffle(dataset_inp_target, shuffle, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataset_inp_target)
    if shuffle:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=shuffle)
    return dataset


def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
            pass
    print("Execution time:", time.perf_counter() - start_time)
