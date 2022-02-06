import sys
sys.path.append('./')
from random import random
import tensorflow as tf
from tensorflow.python.eager.context import num_gpus
from tensorflow.python.ops.gen_math_ops import mod
from os import path, environ
import time

from dataset_utils.phoneDepth_utils import phoneDepth_dataset, available_phones, available_io_modes
from dataset_utils.mai_utils import mai_dataset
from dataset_utils.md_utils import md_dataset
from misc import load_newest_weights, setup_log
from models.compileStrategies import compile_md_no_ord, compile_md,  compile_md_doubleDepth
from models.effnet_parkmai import effiDepth_dualatt_parkmai_net, effi_dualatt_parkmai_net
from models.park_mai_depth import bts_model_f
from dataset_utils.aug_utils import color_jitter, salty_noise, random_crop_and_resize, random_rotation, cascade_functions


model_input_size=(224,224,3) # B4

io_shape = model_input_size[:2]
alpha = 0.75
num_features = 128
# batch_size = 64
batch_size = 64 # park_mai_md2_224x224
batch_size = 64 # park_mai_md_224x224

interm_shape = (480,  640)

# Depthwisesep network park based
block_type = 'depthwise_sep'
n_groups = 1
n_block_length = 1
depthwise_separable = True
batch_size = 9
learning_rate = 5e-5

# output_shape = io_shape
trainable_encoder = True

# Number of samples to pick from each dataset. Load all if None.
n_samples_train = None#batch_size*100
n_samples_val = None#batch_size

dataset = 'md'

phone = 'hua'
# io_mode = 'img2depth'
io_mode = "img2projected"
# io_mode = "img_depth2depth"
# io_mode = "img2depth_depth"
# io_mode = "img_depth2depth_depth"
naming_dict = {'img2depth': "I2D",
               'img2projected': "I2P",
               'img_depth2depth': "ID2P",
               'img2depth_depth': "I2DP",
               "img_depth2depth_depth": "ID2DP",
               }

fine_tune = True

# Number of samples to pick from each dataset. Load all if None.
n_samples_train = None#batch_size*100
n_samples_val = None#batch_size

model_name="park_mai{}{}{}_{}{}_{:03d}x{:03d}{}".format(
            "_fzenc" if not trainable_encoder else "",
            "_{}".format(naming_dict[io_mode]) if dataset=="mb" else "",
            phone if dataset=='mb' else "",
            dataset,
            "_fineTuneMBI2DP5e-04" if fine_tune else "",
            model_input_size[0],
            model_input_size[1],
            )


epochs = 120
# Define Augmentation transforms
jitter = color_jitter(1.0, 0.1, 0.1, 0.1, 0.1)
salt_noise = salty_noise(1.0, 0.01)
combined_img_aug_tranform = cascade_functions([jitter, salt_noise])

crop_resize_transform = random_crop_and_resize(prob=0.3, min_size=0.6, max_size=1.0, img_shape=model_input_size[:2], depth_shape=None, center_crop=False)
rotation_aug_transform = random_rotation(1.0, 2.5)

# Cascaded geometric transformation
geometric_transform = cascade_functions([crop_resize_transform, rotation_aug_transform])


# Dataset path

num_parallel_calls = 8
if dataset=='mb':
    data_path = '/scratch_net/minga/tfausto/data/FTDataset'
    results_path = '/srv/beegfs02/scratch/efficient_nn_mobile/data/FTDataset'

    dataset_train = phoneDepth_dataset(data_path, mode='train', input_size=interm_shape,
                                    batch_size=batch_size,
                                    random_flip=True, n_images=n_samples_train,
                                    phone=phone, io_mode=io_mode,
                                    geometric_aug_transform=geometric_transform,
                                    img_aug_transform=combined_img_aug_tranform,
                                    shuffle=True,
                                    num_parallel_calls=num_parallel_calls)

    dataset_val = phoneDepth_dataset(data_path, mode='val', input_size=interm_shape,
                                    batch_size=batch_size,
                                    random_flip=False, n_images=n_samples_train,
                                    phone=phone, io_mode=io_mode,
                                    geometric_aug_transform=geometric_transform,
                                    img_aug_transform=combined_img_aug_tranform,
                                    shuffle=False,
                                    num_parallel_calls=num_parallel_calls)
elif dataset=='md':
    data_path = '/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1'
    results_path = data_path
    dataset_train = md_dataset(dataset_dir=data_path, partition_list='final_list', mode='train',
                                depth_type='npy', input_shape=io_shape,
                                batch_size=batch_size, random_flip=True, shuffle=True,
                                num_parallel_calls=num_parallel_calls)

    dataset_val = md_dataset(dataset_dir=data_path, partition_list='final_list', mode='val',
                                depth_type='npy', input_shape=io_shape,
                                batch_size=batch_size, random_flip=False, shuffle=False,
                                num_parallel_calls=num_parallel_calls)
elif dataset=='mai':
    data_path = '/scratch_net/minga/tfausto/data/MAI2021_dataset'
    results_path = '/srv/beegfs02/scratch/efficient_nn_mobile/data/MAI2021_dataset'
    dataset_train = mai_dataset(mai_dir=data_path, mode='train', input_shape=interm_shape,
                                batch_size=batch_size, random_flip=True,
                                in_transform=combined_img_aug_tranform, io_transform=geometric_transform,
                                shuffle=True, num_parallel_calls=num_parallel_calls)
    dataset_val = mai_dataset(mai_dir=data_path, mode='val', input_shape=io_shape,
                                batch_size=batch_size, random_flip=False,
                                shuffle=False, num_parallel_calls=num_parallel_calls)

    
logs_basepath=path.join(results_path, 'depth_logs') # where you want to safe your logs!

transfer_starting_model = None

if not fine_tune:
    transfer_starting_model = None


# Naming coventions
weight_base_dir = path.join(results_path, "networks") #the parent directory for all your different network weigths!
checkpoint_dir = path.join(weight_base_dir, model_name)
checkpoint_filepath = checkpoint_dir + '/weights_{epoch:03d}'

print("Start create network")
net = bts_model_f(input_shape=model_input_size, alpha=alpha, num_features=num_features, minimalistic=False)

print("Created network")

if io_mode == "img2depth_depth" or io_mode == "img_depth2depth_depth":
    compile_md_doubleDepth(net, learning_rate=learning_rate)
else:
    compile_md(net, learning_rate=learning_rate)


print("Started loading weights")
if transfer_starting_model:
    net.load_weights(transfer_starting_model)
    print("Loaded pretrained model: {}".format(transfer_starting_model))
    initial_epoch = 0
else:
    initial_epoch = load_newest_weights(net, checkpoint_dir=checkpoint_dir) # loading newest weights if weights are present and returning current epoch



log_dir = setup_log(model_name, log_dir= logs_basepath) # continue with newest log or create dir for new one

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0, update_freq='epoch')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_weights_only=True, mode='min', monitor=['loss', 'val_loss'], verbose=1, save_best_only=False)
net.fit(dataset_train, epochs=epochs, initial_epoch =initial_epoch, callbacks=[model_checkpoint, tensorboard_callback], validation_data=dataset_val)
