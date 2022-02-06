import tensorflow as tf
from pathlib import Path
import sys
sys.path.append('./')
from pathlib import Path

from dataset_utils.aug_utils import color_jitter, cascade_functions, salty_noise, random_rotation, random_crop_and_resize
from dataset_utils.md_utils import md_dataset
from dataset_utils.mai_utils import mai_dataset
from dataset_utils.phoneDepth_utils import phoneDepth_dataset, confidence_indeces
from models.compileStrategies import compile_md_doubleDepth
from models.models import depth_model, final_models_checkpoints, dir_name_from_name
from misc import load_newest_weights, setup_log
from dataset_utils.viz_utils import save_imgs_viz
from depth_utils import preprocess_batch_mask
from dataset_utils.aug_utils import random_crop_and_resize


naming_dict = {'img2depth': "I2D",
               'img2projected': "I2P",
               'img_depth2depth': "ID2P",
               'img2depth_depth': "I2DP",
               "img_depth2depth_depth": "ID2DP"
               }


# Change these variables to the location of the datasets in your Directory your base directories here!
megadepth_basepath = Path('/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1')
mai_basepath = Path('/scratch_net/minga/tfausto/data/MAI2021_dataset')
mb_basepath = Path('/scratch_net/minga/tfausto/data/PhoneDepth')

dataset_paths = {'md': megadepth_basepath,  # Megadepth
                 'mai': mai_basepath,       # MAI
                'mb': mb_basepath}          # PhoneDepth

"""------------------------------------- Modify parameters according to need ------------------------------------------"""
dataset = 'md'
# For Phone depth dataset (mb): "hua" or "pxl"
phone = "hua"
interm_shape = (480,  640)

# mb_io_mode = 'img2depth'
mb_io_mode = "img2projected"
# mb_io_mode = "img_depth2depth"
# mb_io_mode = "img2depth_depth"
# mb_io_mode = "img_depth2depth_depth"
# mb_io_mode = None

dataset_path = dataset_paths[dataset]

# Defined Models
# Trained on MAI and fine-tunned
# model_name = "p_fastdepth_mai_224x224"
# model_name = "p_parkmai_mai_224x224"
# model_name = "p_effiB4park_mai_384x384"
# model_name = "p_fastdepth_mai_224x224_fineTuneMB"
# model_name = "p_parkmai_mai_224x224_fineTuneMB"
# model_name = "p_effiB4park_mai_384x384_fineTuneMB"

# # Trained on MD and fine-tunned
# model_name = "p_fastdepth_md_224x224"
# model_name = "p_parkmai_md_224x224"             
# model_name = "p_effiB4park_md_384x384"
# model_name = "p_fastdepth_md_224x224_fineTuneMB"
model_name = "p_parkmai_md_224x224_fineTuneMB"
# model_name = "p_effiB4park_md_384x384_fineTuneMB"

# # For I2P and I2DP comparison.
# model_name = "p_fastdepth_mbI2P_224x224"
# model_name = "p_parkmai_mbI2P_224x224"
# model_name = "p_effiB4park_mbI2P_384x384"
# model_name = "p_fastdepth_mbI2DP_224x224"
# model_name = "p_parkmai_mbI2DP_224x224"
# model_name = "p_effiB4park_mbI2DP_384x384"

# # Depth enhancement.
# model_name = "p_effiB4park_mbID2P_384x384"
# model_name = "p_effiB4park_mbID2DP_384x384"

batch_size = 9
epochs = 120

input_size = (224, 224, 3)
# input_size = (384, 384, 3)

pretrained_weights = None

"""------------------------------------------------------------------------------------------------------"""

def main():

    """--------------------------------------------------------DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING --------------------------------------------------------  """
    # Checking for Mobile Depth specific configurations
    if ("2depth_depth" not in mb_io_mode) or dataset != 'mb':
        compile_strategy = 'md'
    else:
        compile_strategy = 'md_dd'
    if dataset != 'mb' and mb_io_mode in["img_depth2depth", "img2depth_depth", "img_depth2depth_depth"]:
        raise ValueError("Invalid combination of dataset {} and mb_io_mode: {}".format(dataset, mb_io_mode))


    # Define Augmentation transforms
    jitter = color_jitter(1.0, 0.1, 0.1, 0.1, 0.1)
    salt_noise = salty_noise(1.0, 0.01)
    combined_img_aug_tranform = cascade_functions([jitter, salt_noise])

    crop_resize_transform = random_crop_and_resize(
        prob=0.3, min_size=0.6, max_size=1.0, img_shape=input_size[:2], depth_shape=None, center_crop=False, conf_indx=confidence_indeces[mb_io_mode])
    rotation_aug_transform = random_rotation(1.0, 2.5)

    # Cascaded geometric transformation
    geometric_transform = cascade_functions([crop_resize_transform, rotation_aug_transform])

    num_parallel_calls = 5
    data_path_str = str(dataset_path)
    if dataset=='mb':
        dataset_train = phoneDepth_dataset(data_path_str, mode='train', input_size=interm_shape,
                                        batch_size=batch_size,
                                        random_flip=True, n_images=None,
                                        phone=phone, io_mode=mb_io_mode,
                                        geometric_aug_transform=geometric_transform,
                                        img_aug_transform=combined_img_aug_tranform,
                                        shuffle=True,
                                        num_parallel_calls=num_parallel_calls)

        dataset_val = phoneDepth_dataset(data_path_str, mode='val', input_size=interm_shape,
                                        batch_size=batch_size,
                                        random_flip=False, n_images=None,
                                        phone=phone, io_mode=mb_io_mode,
                                        geometric_aug_transform=geometric_transform,
                                        img_aug_transform=combined_img_aug_tranform,
                                        shuffle=False,
                                        num_parallel_calls=num_parallel_calls)
    elif dataset=='md':
        dataset_train = md_dataset(dataset_dir=data_path_str, partition_list='final_list', mode='train',
                                    depth_type='npy', input_shape=input_size,
                                    batch_size=batch_size, random_flip=True, shuffle=True,
                                    num_parallel_calls=num_parallel_calls)

        dataset_val = md_dataset(dataset_dir=data_path_str, partition_list='final_list', mode='val',
                                    depth_type='npy', input_shape=input_size,
                                    batch_size=batch_size, random_flip=False, shuffle=False,
                                    num_parallel_calls=num_parallel_calls)
    elif dataset=='mai':
        dataset_train = mai_dataset(mai_dir=data_path_str, mode='train', input_shape=interm_shape,
                                    batch_size=batch_size, random_flip=True,
                                    in_transform=combined_img_aug_tranform, io_transform=geometric_transform,
                                    shuffle=True, num_parallel_calls=num_parallel_calls)
        dataset_val = mai_dataset(mai_dir=data_path_str, mode='val', input_shape=input_size[:-1],
                                    batch_size=batch_size, random_flip=False,
                                    shuffle=False, num_parallel_calls=num_parallel_calls)
    
    # From trianing location
    weight_base_dir = dataset_path / Path('networks')
    checkpoint_dir = weight_base_dir / Path(model_name)
    checkpoint_dir = checkpoint_dir.__str__()
    checkpoint_filepath = checkpoint_dir + '/weights_{epoch:03d}'
    logs_basepath= dataset_path / "depth_logs"

    model = depth_model(model_name, compile_strategy=compile_strategy)

    print("Started loading weights")
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        print("Loaded pretrained model: {}".format(pretrained_weights))
        initial_epoch = 0
    else:
        initial_epoch = load_newest_weights(model, checkpoint_dir=checkpoint_dir) # loading newest weights if weights are present and returning current epoch


    log_dir = setup_log(model_name, log_dir=logs_basepath.__str__()) # continue with newest log or create dir for new one 

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0, update_freq='epoch')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_weights_only=True, mode='min', monitor=['loss', 'val_loss'], verbose=1, save_best_only=False)

    print("Started fittiing")
    model.fit(dataset_train, epochs=epochs, initial_epoch =initial_epoch, callbacks=[model_checkpoint, tensorboard_callback], validation_data=dataset_val)

if __name__ == "__main__":
    main()