import sys
sys.path.append('./')
from random import random
import tensorflow as tf
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.ops.gen_math_ops import mod
from os import path, environ
import time

from dataset_utils.mai_utils import mai_dataset
from dataset_utils.md_utils import md_dataset
from misc import load_newest_weights, setup_log
from models.effnet_parkmai import compile_md, effiparkmai_net
from dataset_utils.aug_utils import color_jitter, salty_noise, random_crop_and_resize, random_rotation, cascade_functions


model_input_size=(384,384,3) # B4
io_shape = model_input_size[:2]
num_features = 128

batch_size = 9 # B4

effnet_version = 4
trainable_encoder = True

epochs = 120

output_shape = io_shape

dataset_type = 'mai'

# Number of samples to pick from each dataset. Load all if None.
n_samples_train = None#batch_size*100
n_samples_val = None#batch_size

fine_tune = True

model_name="effnetB{}{}parkmai_{}{}_{:03d}x{:03d}".format(
            effnet_version,
            "frozen_" if not trainable_encoder else "",
            dataset_type,
            "_fineTune60_new" if fine_tune else "",
            model_input_size[0],
            model_input_size[1]
            )

if dataset_type == 'mai':

    # Define Augmentation transforms
    jitter = color_jitter(1.0, 0.1, 0.1, 0.1, 0.1)
    salt_noise = salty_noise(1.0, 0.01)
    normalization_img = lambda x: x
    normalization_depth = lambda x: x/(2**16-1)
    combined_input_aug = cascade_functions([jitter, salt_noise, normalization_img])

    crop_resize_transform = random_crop_and_resize(prob=0.3, min_size=0.6, max_size=1.0, img_shape=model_input_size[:2], depth_shape=output_shape, center_crop=False)
    rotation_aug_transform = random_rotation(1.0, 2.5)

    # Cascaded geometric transformation
    geometric_transform = cascade_functions([crop_resize_transform, rotation_aug_transform])

    # Note negative probability, for stability. Don't want to crop in validation
    val_geometric_transform = random_crop_and_resize(prob=-1e-5, img_shape=model_input_size[:2], depth_shape=output_shape)

    # Mai Datasets for training
    data_path = '/srv/beegfs02/scratch/efficient_nn_mobile/data/MAI2021_dataset'
    dataset_train = mai_dataset(data_path, mode='train', input_shape=io_shape, batch_size=batch_size,
                                random_flip=True, n_images=n_samples_train,
                                in_transform=combined_input_aug,
                                out_transform=normalization_depth,
                                io_transform=geometric_transform,
                                shuffle=True)
    dataset_val = mai_dataset(data_path, mode='val', input_shape=io_shape, batch_size=batch_size,
                                random_flip=False, n_images=n_samples_train,
                                in_transform=normalization_img,
                                out_transform=normalization_depth,
                                io_transform=val_geometric_transform,
                                shuffle=False)

elif dataset_type == 'md':
    # MD Datasets for training
    depth_data_type = 'npy'
    data_path = '/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1'
    dataset_train = md_dataset(data_path, 
                                mode='train', depth_type=depth_data_type,
                                input_shape=io_shape, batch_size=batch_size,
                                random_flip=True, n_images=n_samples_train,
                                in_transform=None,
                                out_transform=None,
                                io_transform = None,
                                shuffle=True)

    dataset_val = md_dataset(data_path, 
                                mode='val', depth_type=depth_data_type,
                                input_shape=io_shape, batch_size=batch_size,
                                random_flip=False, n_images=n_samples_train, 
                                in_transform=None,
                                out_transform=None,
                                io_transform = None,
                                shuffle=False)
    
logs_basepath=path.join(data_path, 'depth_logs') # where you want to safe your logs!

# Fine tunning
transfer_starting_model = None
if not fine_tune:
    transfer_starting_model = None


# Naming coventions
weight_base_dir = path.join(data_path, "networks") #the parent directory for all your different network weigths!
checkpoint_dir = path.join(weight_base_dir, model_name)
checkpoint_filepath = checkpoint_dir + '/weights_{epoch:03d}'

net = effiparkmai_net(version=effnet_version, input_shape=model_input_size, decoder_features=num_features, separated_submodules=True, trainable_encoder=trainable_encoder)

initial_epoch = load_newest_weights(net, checkpoint_dir=checkpoint_dir) # loading newest weights if weights are present and returning current epoch

if transfer_starting_model:
    net.load_weights(transfer_starting_model)
    print("Loaded pretrained model: {}".format(transfer_starting_model))

compile_md(net)

log_dir = setup_log(model_name, log_dir= logs_basepath) # continue with newest log or create dir for new one

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0, update_freq='epoch')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_weights_only=True, mode='min', monitor=['loss', 'val_loss'], verbose=1, save_best_only=False)
net.fit(dataset_train, epochs=epochs, initial_epoch =initial_epoch, callbacks=[model_checkpoint, tensorboard_callback], validation_data=dataset_val)
