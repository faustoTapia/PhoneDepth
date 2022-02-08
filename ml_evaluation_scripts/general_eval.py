import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from pathlib import Path
import json
from typing import List, final
import sys

from tensorflow.python.types.core import Value

sys.path.append('./')
from pathlib import Path
from custom_callbacks import EvaluationWithSampleBatchIOViz
from tensorflow.python.keras.engine import data_adapter
import tensorflow.keras.backend as K

from dataset_utils.md_utils import md_dataset
from dataset_utils.mai_utils import mai_dataset
from dataset_utils.phoneDepth_utils import phoneDepth_dataset, phoneDepth_dataset
from models.compileStrategies import compile_md_doubleDepth
from models.models import depth_model, final_models_checkpoints, dir_name_from_name
from misc import load_newest_weights, setup_log
from dataset_utils.viz_utils import save_imgs_viz
from depth_utils import preprocess_batch_mask
from dataset_utils.aug_utils import random_crop_and_resize

# Change to your base directories where you stored hte datasets here!
megadepth_basepath = Path('/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1')
mai_basepath = Path('/srv/beegfs02/scratch/efficient_nn_mobile/data/MAI2021_dataset')
mb_basepath = Path('/srv/beegfs02/scratch/efficient_nn_mobile/data/PhoneDepth')

# Change these directories as they are where you results will be stored.
mai_storepath = mai_basepath
mb_storepath = mb_basepath
megadepth_storepath = megadepth_basepath

dataset_paths = {'md': megadepth_basepath,
                 'mai': mai_basepath,
                'mb': mb_basepath}

dataset_store_paths = {'md': megadepth_storepath,
                        'mai': mai_storepath,
                        'mb': mb_storepath
                        }

naming_dict = {'img2depth': "I2D",
               'img2projected': "I2P",
               'img_depth2depth': "ID2P",
               'img2depth_depth': "I2DP",
               "img_depth2depth_depth": "ID2DP"
               }


def main():
    dataset = 'mai'
    pretrain_datset = dataset
    cmap = 'viridis'

    mb_io_mode = 'img2depth'
    # mb_io_mode = "img2projected"
    # mb_io_mode = "img_depth2depth"
    # mb_io_mode = "img2depth_depth"
    # mb_io_mode = "img_depth2depth_depth"
    # mb_io_mode = None
    
    pre_training_dataset_store = dataset_store_paths[pretrain_datset]
    curr_dataset_store_path = dataset_store_paths[dataset]
    dataset_path = dataset_paths[dataset]

    # Checking for Mobile Depth specific configurations
    if ("2depth_depth" not in mb_io_mode) or dataset != 'mb':
        compile_strategy = 'md'
    else:
        compile_strategy = 'md_dd'
    if dataset != 'mb' and mb_io_mode in["img_depth2depth", "img2depth_depth", "img_depth2depth_depth"]:
        raise ValueError("Invalid combination of dataset {} and mb_io_mode: {}".format(dataset, mb_io_mode))

    # Paper models
    # Trained on MAI and fine-tunned
    model_name = "p_fastdepth_mai_224x224"
    # model_name = "p_parkmai_mai_224x224"
    # model_name = "p_effiB4park_mai_384x384"                     ##
    # model_name = "p_fastdepth_mai_224x224_fineTuneMB"         ##
    # model_name = "p_parkmai_mai_224x224_fineTuneMB"
    # model_name = "p_effiB4park_mai_384x384_fineTuneMB"        ##

    # # Trained on MD and fine-tunned
    # model_name = "p_fastdepth_md_224x224"                     ##
    # model_name = "p_parkmai_md_224x224"       
    # model_name = "p_effiB4park_md_384x384"
    # model_name = "p_fastdepth_md_224x224_fineTuneMB"
    # model_name = "p_parkmai_md_224x224_fineTuneMB"            ##
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



    # From trianing location
    weight_base_dir = pre_training_dataset_store / Path('p_networks')

    checkpoint_dir = weight_base_dir / Path(model_name)

    checkpoint_dir = str(checkpoint_dir)
    batch_size = 1

    # n_imgs = 500
    n_imgs = None

    start_img = 0
    input_size = (224, 224, 3)
    # input_size = (384, 384, 3)

    dataset_type = 'test'
    dataset_type = 'val'
    # dataset_type = 'train'

    tar_shape = input_size

    best_checkpoint = None
    if model_name in final_models_checkpoints.keys():
        best_checkpoint = final_models_checkpoints[model_name]
        print("Found predifined best checkpoint: {}".format(best_checkpoint))


    # Test set
    tf.random.set_seed(123)
    if dataset == 'md':
        img_type = 'all'
        dataset = md_dataset(dataset_path.as_posix(), partition_list="wo_ordinal_list", mode = dataset_type,
                            img_type=img_type, depth_type='npy', 
                            input_shape=input_size[:2],
                            output_shape=tar_shape[:2],
                            evaluation=True,
                            batch_size=batch_size,
                            n_images=n_imgs,
                            start_img=start_img,
                            in_transform = None,
                            shuffle=True)

    elif dataset == 'mai':
        in_transform = lambda img: tf.image.resize(img, input_size[:-1])
        dataset = mai_dataset(dataset_path.__str__(), mode = dataset_type, input_shape=(480,640), in_transform=in_transform, batch_size=batch_size, shuffle=True, num_parallel_calls=1)
    
    elif dataset == 'mb':
        dataset = phoneDepth_dataset(dataset_path.__str__(), mode=dataset_type, input_size=input_size[:2], out_size=(720,960),n_images=n_imgs, phone='hua', io_mode=mb_io_mode, batch_size=batch_size, shuffle=False, num_parallel_calls=1)
    else:
        raise ValueError("Invalid dataset.")
    
    # output_dir setup
    output_dir = curr_dataset_store_path / Path('p_eval_outputs') / Path(model_name + '_' + cmap + '_'+ dataset_type)
    imgs_dir = output_dir / Path('inp-outs')
    if not imgs_dir.is_dir():
        Path.mkdir(imgs_dir, parents=True)

    model = depth_model(model_name, compile_strategy=compile_strategy)
    results_to_store=['loss', 'si_rmse', 'rmse_nonlog','gradient_loss_multiscale', 'error_ordinal_wrap', 'rmse_nonlog', 'average_rel_error', 'average_log10_error']
    displayed_results_list =[]
    model.test_step = eval_storing_test_step(model, output_dir=output_dir, results_list=displayed_results_list,
                                                mask_pred=True, period_elm=10, results_to_store=results_to_store,
                                                cmap=cmap)

    initial_epoch = load_newest_weights(model, checkpoint_dir=checkpoint_dir, best_checkpoint=best_checkpoint) # loading newest weights and returning current epoch

    eval_metrics = model.evaluate(dataset)
    save_samples_metrics(results_to_store, displayed_results_list, output_dir)
    save_metrics(output_dir, model.metrics_names, eval_metrics)
    print("Model evaluation")
    print(model.metrics_names)
    print(eval_metrics)

def eval_storing_test_step(keras_model, output_dir:Path, results_list:List, mask_pred=True, period_elm=10, results_to_store=None, cmap='gray'):
    if results_to_store is None:
        results_to_store['loss', 'si_rmse', "si_mse", 'rmse_nonlog','gradient_loss_multiscale', 'error_ordinal_wrap', 'rmse_nonlog']
    original_test_step = keras_model.test_step
    imgs_dir = output_dir / Path("inp-outs")

    def save_eval_data(img, tar, pred, batch, result):
        batch = K.eval(batch)
        result = K.eval(result)
        img = K.eval(img)
        tar = K.eval(tar)
        pred = K.eval(pred)

        inp_depth = None
        tar_g = None

        if batch % period_elm == 0:
            inpt_img = tf.expand_dims(K.eval(img)[0], axis=0)
            print("inpt_img shape: {}".format(inpt_img.shape))
            if inpt_img.shape[-1] > 3:
                inp_depth = tf.expand_dims(inpt_img[:,:,:, 3], axis=-1)
                inpt_img = inpt_img[:,:,:,0:3]
            pred = tf.expand_dims(K.eval(pred)[0], axis=0)
            tar = tf.expand_dims(K.eval(tar)[0], axis=0)
            pred_depth = tf.math.exp(pred)

            if tar.shape[-1] > 1:
                tar_depth = tf.expand_dims(tar[:,:,:,-1], axis=-1)
                tar_g = tf.expand_dims(tar[:,:,:,0], axis=-1)
            else:
                tar_depth = tar

            tar_depth, pred_depth, mask = preprocess_batch_mask(tar_depth, pred_depth)
            if mask_pred:
                tar_depth = tf.math.multiply_no_nan(tar_depth, mask)
                pred_depth = tf.math.multiply_no_nan(pred_depth, mask)

            names = ['{:06d}_img.jpg'.format(batch),
                        '{:06d}_depth_tar.jpg'.format(batch),
                        '{:06d}_depth_pred_.jpg'.format(batch)]

            out_imgs=[inpt_img,
                        tar_depth,
                        pred_depth]
            
            if inp_depth is not None:
                out_imgs += [inp_depth]
                names += ['{:06d}_indepth.jpg'.format(batch)]
            if tar_g is not None:
                out_imgs += [tar_g]
                names += ['{:06d}_depth_tar_g.jpg'.format(batch)]
            
            for name, img_ in zip(names, out_imgs):
                print("Img '{}': {}".format(name, img_.shape))

            save_imgs_viz(out_imgs, names, imgs_dir.__str__(), cmap=cmap)
            results_list.append([batch] + [elem for elem in result])

    def eval_storing_step(original_data):
        if not hasattr(eval_storing_step, "batch_counter"):
            eval_storing_step.batch_counter = tf.Variable(0)
        
        result = original_test_step(original_data)

        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
    
        y_pred = keras_model(x, training=False)

        tf.py_function(save_eval_data, [x, y_true, y_pred, eval_storing_step.batch_counter, [result[key] for key in results_to_store if key in result.keys()]], [])

        # add anything here for on_train_batch_end-like behavior
        K.update_add(eval_storing_step.batch_counter, 1)

        return result
    

    return eval_storing_step


def save_samples_metrics(metrics_names, values_list, output_dir):
    labels = ['batch_num'] + metrics_names
    data = []
    for elem in values_list:
        instance = {}
        for label, value in zip(labels, elem):
            instance[label] = int(value)
        data.append(instance)
    output_path = output_dir / Path("samples_metrics.json")

    with open(output_path.as_posix(), 'w') as file:
        json.dump(data, file)

def save_metrics(output_dir: Path, labels: List[str], metrics: List[float]) -> None:
    data = {}
    for label, metric in zip(labels, metrics):
        data[label] = metric
    output_file = output_dir / Path('eval_metrics.json')
    with open(output_file.as_posix(), 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    main()