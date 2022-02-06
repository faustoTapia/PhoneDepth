import sys
from numpy.core.defchararray import index

from numpy.lib.type_check import imag
from matplotlib import pyplot as plt
sys.path.append('./')
from plotting.plotting_utils import data_frame_to_metric_dic
from plotting.plotting_utils import plot_image_matrix
from dataset_utils.phoneDepth_utils import load_data_list_from_file, IOModes
from PIL import Image
import numpy as np
import random

def plot_samples_images(dataset_path ="/srv/beegfs02/scratch/efficient_nn_mobile/data/FTDataset/", split='train', index = 0):
    filename = "sample_{:05d}".format(index) + ".jpg"
    list_file = dataset_path + split + '_list.json'
    figsize = (24, 35)
    data_list_hua = load_data_list_from_file(list_file, phone='hua', io_mode=IOModes.IMG2DEPTH_DEPTH_CONF)
    data_list_pxl = load_data_list_from_file(list_file, phone='pxl', io_mode=IOModes.IMG2DEPTH_DEPTH_CONF)

    images_hua = []
    images_pxl = []

    for i in range(len(data_list_hua)):
        curr_img = data_list_hua[i][index]
        images_hua.append(np.array(curr_img))
        curr_img = data_list_pxl[i][index]
        images_pxl.append(np.array(curr_img))

    images_hua[2:] = images_hua[3:1:-1]
    images_pxl[2:] = images_pxl[3:1:-1]

    images = [images_hua, images_pxl]

    fig = plot_image_matrix(images, [""] * len(images[0]), font_size=10, as_rows=False, figsize=figsize)

    fig.savefig("./plotting_results/" + filename)
    plt.close(fig)

    print('done')

if __name__ == "__main__":

    indeces = list(range(3000, 5000, 20))

    for i in indeces:
        plot_samples_images(index=i)