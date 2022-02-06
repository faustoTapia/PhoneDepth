import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Any, Dict
from pathlib import Path
from PIL import Image
import tensorboard as tb

tensorboard_logs_dic = {
                        "unet_quad_md_224x224": "a75dXRHNTzevu9M14h2bHw",
                        "unet_quad_md_288x288": "QubEvRbWQ420T2vugKCRCQ",
                        "park_mai_md2_224x224": "T5MxFXpASpKgg6hKiNdY1w",
                        "park_mai_md_224x224": "rfnC3yvmTxy3v6zpH0N95Q",
                        "mobilepy_small_net_md_224x224_32fil_5dep": "KAivbLqQSXGa3sGUq7yKDA",
                        "mobilepy_net_md_128x128_16fil_5dep": "LwGN0trvQDebKc7AOPZM9A",
                        "mobilepy_net_atrous_md_288x288_32fil_5dep": "An0MX9TBRMecOeMAp3INsA",
                        "mobilepy_net_atrous_depsep_gate_md_224x224_16fil_5dep": "xHw4b8i7RoK9T2ytNKOhDg",
                        "effnetB0parkmai_md_224x224": "0UYIbyALQBGSDNkjRw9MDA",                                 # Fully trained
                        "effnetB0parkmai_md_256x256": "DYfd4EJESJmmlVI0r5FNdA",                                 # Halfway trained
                        "effnetB0parkmai_old_md_224x224": "CgZpY5CnRVWxHo2qO1d0dA",                             # Halfway trained
                        "effnetB1parkmai_md_256x256": "S1QH3bYkTDyH72po9AxB8w",                                 # Fully trained
                        "effB1_pynetsmall_depsep_md_256x256_32fil_5dep": "8dBdNZiKTmWkJFBlxntnUQ",              # Fully trained
                        "effB1_pynetsmall_dualatt_depsep_md_256x256_32fil_5dep": "CzbWZnvoTkuJNNL7b0jAXQ",      # Fully trained
                        "effB1_pynetsmall_globNonLoc_depsep_md_256x256_32fil_5dep": "pWTyBpvcRkOoZs8A2vtKWg",   # Fully trained
                        "effB3_pynet_transposeConv_depsep_md_288x288_32fil_5dep": "rsw5SRS3SzK2FomyzCrnCw",     # Fully trained
                        "effB3_pynet_pxlShuffle_depsep_md_288x288_32fil_5dep":"lr89gmVoQHWP5gqTKOunxA",         # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_256x256_32fil_5dep": "QJzRS3qjSY21JS19z6PRCw",        # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_288x288_32fil_5dep": "lNc8zSOhR7m4vkOAOvV3Vg",        # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_320x320_32fil_5dep": "o2kOI452To6Nos1RP9rbKg",        # Fully trained
                        "effnetB4_dualatt_parkmai_md_384x384": "h5DJyrubTeesaVBRs0OBdg",
                        "effnetB4parkmai_md_384x384": "jdgYgrSXR8C2ORVA1IW3OQ",

                        # Not finalized yet
                        "effnetB4_dualatt_parkmai_group_convs_g4_n3_md_384x384":    "DS33oVJBSyaHB7uYNLqkOg",
                        "effnetB4_dualatt_parkmai_imdb_md_384x384":                 "maP2DRtjS0WXXygdt7K6Ig",
                        "effnetB4_dualatt_parkmai_rrdb_depthsep_md_384x384":        "j7XV6kJqSQ2rBV0v2071Gg",

                        # On MAI
                        "parkmai_mai_224x224": "Q1XVvuJQSGOLQXWdanz0sQ",

                        "effnetB4parkmai_mai_384x384": "4uZ8eTEBRRK5e4v2RwjIAg",
                        "effnetB4parkmai_mai_fineTune60_384x384": "GEGFPkB4RBikXeyFv2hy5Q",
                        "effnetB4_dualatt_parkmai_mai_384x384": "bKu3a9aLSqmFZ8L72PGbIw",
                        "effnetB4_dualatt_parkmai_mai_fineTune_last_384x384": "5oeACK7MS9e1AgsgurJR9Q",

                        "effnetB4_dualatt_parkmai_imdb_mai_384x384": "Ld3JIYwzQeiQTo7DyMimYQ",
                        "effnetB4_dualatt_parkmai_imdb_mai_fineTune_384x384": "XojmDQrsQ8KrADmUGLQ5Iw",

                        # On PhoneDepth
                        "effnetB4_I2DhuaDepth_dualatt_parkmai_imdb_mb_384x384":     "EGS7CExWQdWm3C3Hn0liAg",
                        "effnetB4_I2PhuaDepth_dualatt_parkmai_imdb_mb_384x384":     "dkDdEcBxT0yaGdXGgsM1GQ",
                        "effnetB4_ID2PhuaDepth_dualatt_parkmai_imdb_mb_384x384":    "3Kp2sONpSpGcp41nuQf7Rw",
                        "effnetB4_I2DPhuaDepth_dualatt_parkmai_imdb_mb_384x384":    "qLVN1aLNRAKKjdOurNr1YA",
                        "effnetB4_ID2DPhuaDepth_dualatt_parkmai_imdb_mb_384x384":   "bcxc3KXzRDCAnDwlODd6Cw",
}

model_training_epochs = {
                        "unet_quad_md_224x224":                                     [60],
                        "unet_quad_md_288x288":                                     [60],
                        "park_mai_md2_224x224":                                     [60],
                        "park_mai_md_224x224":                                      [60],
                        "mobilepy_small_net_md_224x224_32fil_5dep":                 [11, 12, 13, 12, 12],
                        "mobilepy_net_md_128x128_16fil_5dep":                       [13, 13, 13, 13, 13],
                        "mobilepy_net_atrous_md_288x288_32fil_5dep":                None,
                        "mobilepy_net_atrous_depsep_gate_md_224x224_16fil_5dep":    None,
                        "effnetB0parkmai_md_224x224":                               [60],                                 # Fully trained
                        "effnetB0parkmai_md_256x256":                               [9],                                # Halfway trained
                        "effnetB0parkmai_old_md_224x224":                           None,                             # Halfway trained
                        "effnetB1parkmai_md_256x256":                               None,                                 # Fully trained
                        "effB1_pynetsmall_depsep_md_256x256_32fil_5dep":            None,              # Fully trained
                        "effB1_pynetsmall_dualatt_depsep_md_256x256_32fil_5dep":    [12] * 5,     # Fully trained
                        "effB1_pynetsmall_globNonLoc_depsep_md_256x256_32fil_5dep": [12] * 5,   # Fully trained
                        "effB3_pynet_transposeConv_depsep_md_288x288_32fil_5dep":   [12] * 5,     # Fully trained
                        "effB3_pynet_pxlShuffle_depsep_md_288x288_32fil_5dep":      [12] * 5,        # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_256x256_32fil_5dep":      [12] * 5,        # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_288x288_32fil_5dep":      [12] * 5,        # Fully trained
                        "effB4_pynet_pxlShuffle_depsep_md_320x320_32fil_5dep":      [12] * 5,        # Fully trained
                        "effnetB4parkmai_md_384x384":                               [60],           # Fully trained
                        "effnetB4_dualatt_parkmai_md_384x384":                      [60],           # Fully trained

                        # Not finalized yet
                        "effnetB4_dualatt_parkmai_group_convs_g4_n3_md_384x384":    [60],
                        "effnetB4_dualatt_parkmai_imdb_md_384x384":                 [60],
                        "effnetB4_dualatt_parkmai_rrdb_depthsep_md_384x384":        None,

                        # On MAI
                        "parkmai_mai_224x224":                                      [120],

                        "effnetB4parkmai_mai_384x384":                              [120],
                        'effnetB4parkmai_mai_fineTune60_384x384':                   [60],
                        "effnetB4_dualatt_parkmai_mai_384x384":                     [120],
                        "effnetB4_dualatt_parkmai_mai_fineTune_last_384x384":       [60],
                        "effnetB4_dualatt_parkmai_imdb_mai_fineTune_384x384":       [60],

                        # On PhoneDepth
                        "effnetB4_I2DhuaDepth_dualatt_parkmai_imdb_mb_384x384":     [120],
                        "effnetB4_I2PhuaDepth_dualatt_parkmai_imdb_mb_384x384":     [97],
                        "effnetB4_ID2PhuaDepth_dualatt_parkmai_imdb_mb_384x384":    [118],
                        "effnetB4_I2DPhuaDepth_dualatt_parkmai_imdb_mb_384x384":    [110],
                        "effnetB4_ID2DPhuaDepth_dualatt_parkmai_imdb_mb_384x384":   [120],
}


def load_data_frame(model_name=""):
    if model_name not in tensorboard_logs_dic.keys():
        raise ValueError("Model not found")
    experiment_id = tensorboard_logs_dic[model_name]
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    return df

def generate_dataframes():
    for model_name in  tensorboard_logs_dic.keys():
        df = load_data_frame(model_name)
        file_path = './plotting/data/' + model_name + ".csv"
        df.to_csv(file_path)
        print("Generated file: {}".format(file_path))


default_metrics_to_collect = ['epoch_error_ordinal_wrap', 'epoch_gradient_loss_multiscale',
                              'epoch_loss', 'epoch_rmse_nonlog', 'epoch_rmse_nonlog_wscale','epoch_si_rmse']


def data_frame_to_metric_dic(data_frame, layer_epochs=None):
    labels = data_frame["run"].unique()
    training_labels = sorted([label for label in labels if "train" in label], reverse=True)
    validation_labels = sorted([label for label in labels if "validation" in label], reverse=True)

    metrics = data_frame["tag"].unique()

    output_dict = {'train': {}, 'validation': {}}

    if layer_epochs is not None and (len(training_labels) != len(validation_labels) or len(training_labels) != len(layer_epochs)):
        ValueError("Dataframe inconsistent number of labels or layer_epochs does not match the numberof labels.")

    for i, (training_label, validation_label) in enumerate(zip(training_labels, validation_labels)):
        for metric in metrics:
            training_metric = data_frame[data_frame["run"] == training_label][data_frame["tag"] == metric]
            validation_metric = data_frame.loc[data_frame["run"] == validation_label][data_frame["tag"] == metric]
            if metric not in output_dict['train'].keys():
                output_dict['train'][metric] = []
            if metric not in output_dict['validation'].keys():
                output_dict['validation'][metric] = []
            curr_training_metric = list(training_metric['value'].values)
            curr_validation_metric = list(validation_metric['value'].values)
            if layer_epochs is not None:
                curr_training_metric = curr_training_metric[:layer_epochs[i]]
                curr_validation_metric = curr_validation_metric[:layer_epochs[i]]
            output_dict['train'][metric] += curr_training_metric
            output_dict['validation'][metric] += curr_validation_metric

    return output_dict

def plot_metrics_from_datadicts(data_dicts: List[Any], labels: List[str], metric: str, split='train', show=True,
                                xlims=None, ylims=None, ax=None):
    if split not in ['train', 'validation']:
        raise ValueError("split must be either of {}".format(split))
    
    if ax is None:
        ax = plt.gca()

    for label, data_dict in zip(labels, data_dicts):
        curr_data = data_dict[split][metric]
        plt.plot(curr_data, label=label)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.legend()
    plt.grid(True)
    return ax

def plot_training_from_datadics(data_dicts: List[Any], labels: List[str], loss_range=None, metric_range=None, epochs_range=None, font_size=15, **kwargs):

    fig, axs = plt.subplots(2, 2, **kwargs)

    for label, data_dict in zip(labels, data_dicts):
        axs[0, 0].plot(data_dict['train']['epoch_loss'], label=label)
        axs[1, 0].plot(data_dict['train']['epoch_si_rmse'], label=label)
        axs[0, 1].plot(data_dict['validation']['epoch_loss'], label=label)
        axs[1, 1].plot(data_dict['validation']['epoch_si_rmse'], label=label)

    if epochs_range is not None:
        axs[0, 0].set_xlim(*epochs_range)
        axs[0, 1].set_xlim(*epochs_range)
        axs[1, 0].set_xlim(*epochs_range)
        axs[1, 1].set_xlim(*epochs_range)
    
    if loss_range is not None:
        axs[0, 0].set_ylim(*loss_range)
        axs[0, 1].set_ylim(*loss_range)
    if metric_range is not None:
        axs[1, 0].set_ylim(*metric_range)
        axs[1, 1].set_ylim(*metric_range)
    

    axs[0, 1].legend()
    axs[1, 0].set_xlabel('Epochs', fontsize=font_size)
    axs[1, 1].set_xlabel('Epochs', fontsize=font_size)
    axs[0, 0].set_ylabel('Loss', fontsize=font_size)
    axs[1, 0].set_ylabel('si-RMSE', fontsize=font_size)

    axs[0, 0].set_title('Training Loss', fontsize=font_size)
    axs[0, 1].set_title('Validation Loss', fontsize=font_size)
    axs[1, 0].set_title('Training si-RMSE', fontsize=font_size)
    axs[1, 1].set_title('Validation si-RMSE', fontsize=font_size)

    for ax in axs.flatten():
        ax.grid(True)

    fig.tight_layout()
    return fig

def load_models_data_dics(data_frames_dir='plotting/data', models_to_extract=None):
    data_frames_dic = {}
    if models_to_extract is None:
        models_to_extract = model_training_epochs.keys()
    for model_name in model_training_epochs:
        data_frame_csv = data_frames_dir + '/' + model_name + '.csv'
        data_frames_dic[model_name] = data_frame_to_metric_dic(pd.read_csv(data_frame_csv), layer_epochs=model_training_epochs[model_name])
    return data_frames_dic


def retrieve_samples_from_dirs(dirs: List[Path], phone_depth_tar: List[bool]):
    if len(dirs) != len(phone_depth_tar):
        raise ValueError('Arguments must have same number of elements')

    img_pattern = "*_img.jpg"
    proj_depth_pattern = "*_depth_tar.jpg"
    phone_in_depth_pattern = "*_indepth.jpg"
    phone_depth_tar_pattern = "*_depth_tar_g.jpg"
    pred_depth_pattern = "*_depth_pred_.jpg"

    sample_blocks = []

    for is_phone_target, dir_base in zip(phone_depth_tar, dirs):
        dir = dir_base / 'inp-outs'
        imgs_available = sorted(list(dir.glob(img_pattern)))
        phone_depths_available = sorted(list(dir.glob(phone_in_depth_pattern)))
        if len(phone_depths_available) < 1:
            phone_depths_available = sorted(list(dir.glob(phone_depth_tar_pattern)))
        proj_depths_available = sorted(list(dir.glob(proj_depth_pattern)))
        pred_depths_available = sorted(list(dir.glob(pred_depth_pattern)))

        if is_phone_target:
            phone_depths_available = proj_depths_available
            proj_depths_available = []

        sample_block = {
                        "imgs": imgs_available,
                        "phone_depths": phone_depths_available,
                        "proj_depths": proj_depths_available,
                        "pred_depths": pred_depths_available
        }
        
        sample_blocks.append(sample_block)
    
    return sample_blocks

def get_image_matrix_from_samples(sample_blocks: List[Dict[str, List[Path]]], sample_indeces: List[int]):

    # Checking content of folders
    elems_to_pick = []
    for samples_block in sample_blocks:
        if len(samples_block['imgs'])>0 and 'imgs' not in elems_to_pick:
            elems_to_pick.append('imgs')
    for samples_block in sample_blocks:
        if len(samples_block['phone_depths'])>0 and 'phone_depths' not in elems_to_pick:
            elems_to_pick.append('phone_depths')
    for samples_block in sample_blocks:
        if len(samples_block['proj_depths'])>0 and 'proj_depths' not in elems_to_pick:
            elems_to_pick.append('proj_depths')
    for samples_block in sample_blocks:
        if len(samples_block['pred_depths'])>0 and 'pred_depths' not in elems_to_pick:
            elems_to_pick.append('pred_depths')


    out_blocks = []
    for i in sample_indeces:
        out_block = [None] * (len(elems_to_pick) + len(sample_blocks) - 1)
        
        for block in sample_blocks:
            # Do not check for predictions as they are always included
            for j, keyw in enumerate(elems_to_pick[:-1]):
                if len(block[keyw]) > 0 and out_block[j] is None:
                    out_block[j] = block[keyw][i]
        
        for j, block in enumerate(sample_blocks):
            out_block[j + len(elems_to_pick) - 1] = block['pred_depths'][i]
        out_blocks.append(out_block)
    return out_blocks
    

def plot_image_matrix(imgs_blocks: List[List[Path]], labels: List[str], img_size=(480,640), font_size=36, as_rows=True, **kwargs):
    if len(imgs_blocks[0]) != len(labels):
        raise ValueError("imgs_blocks and labels should have the same number of elements, blocks have {} and labels have {}".format(len(imgs_blocks[0]), len(labels)))

    aux_list_length = None
    for img_block in imgs_blocks:
        if aux_list_length is None:
            aux_list_length = len(img_block)
        if aux_list_length != len(img_block):
            raise ValueError("Invalid blocks, they must all have the same number of elements.")

    n_rows = len(imgs_blocks)
    n_cols = len(imgs_blocks[0])
    if not as_rows:
        n_rows, n_cols = n_cols, n_rows

    fig, axs = plt.subplots(n_rows, n_cols, **kwargs)
    
    for i in range(len(imgs_blocks)):
        curr_block = imgs_blocks[i]
        for j in range(len(curr_block)):
            curr_img = curr_block[j]
            if curr_img is not None:
                col_indx, row_indx = (i, j) if as_rows else (j, i)
                img = Image.open(curr_img.__str__())
                img = img.resize(img_size[::-1])
                img = np.array(img)
                axs[col_indx, row_indx].imshow(img)
                axs[col_indx, row_indx].set_xticks([])
                axs[col_indx, row_indx].set_yticks([])
                # axs[col_indx, row_indx].set_axis_off()
    
    # Setup labels
    if as_rows:
        for i in range(n_cols):
            axs[0, i].set_title(labels[i], fontsize=font_size)
    else:
        for i in range(n_rows):
            axs[i, 0].set_ylabel(labels[i], fontsize=font_size)
    fig.tight_layout()
    return fig

if __name__=="__main__":
    # Generating dataframes from tensorflow logs
    generate_dataframes()

    print("Finished")
