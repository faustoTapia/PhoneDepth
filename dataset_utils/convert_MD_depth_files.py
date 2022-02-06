# This script augments MD datset with .npy and .png versions of the depthmaps.
# Allowing for faster loading of depth files (which originally are in .h5 format).

from __future__ import division
from attr import asdict
import numpy as np
import pickle 
import h5py
from tqdm import tqdm
import time
from time import gmtime
from time import strftime
from os import path, mkdir
from skimage import io as skimg_io

def generate_depth_files(root = "/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1/"):
    subroot = "phoenix/S6/zl548/MegaDepth_v1"
    orientation_list = ["landscape", "portrait"]

    print_frequency = 100

    for orientation in orientation_list:

        dir_load_train_val_target = root + "/final_list/train_val_list/" + orientation + "/targets_MD.p"
        dir_load_test_target = root + "/final_list/test_list/" + orientation + "/targets_MD.p"

        target_list = np.load(dir_load_train_val_target, allow_pickle=True) + np.load(dir_load_test_target, allow_pickle=True)

        for i, target_file in enumerate(tqdm(target_list)):
            new_target_np_dir = path.join(root, subroot, path.dirname(target_file)+"_npy")
            if not path.isdir(new_target_np_dir):
                mkdir(new_target_np_dir)
            new_target_png_dir = path.join(root, subroot, path.dirname(target_file)+"_png")
            if not path.isdir(new_target_png_dir):
                mkdir(new_target_png_dir)
            new_target_png_filt_dir = path.join(root, subroot, path.dirname(target_file)+"_pngfilt")
            if not path.isdir(new_target_png_filt_dir):
                mkdir(new_target_png_filt_dir)
            
            file_name = path.basename(target_file).split('.')[0]

            h5_file = h5py.File(path.join(root, subroot, target_file), 'r')
            depth_img = h5_file['/depth']

            depth_img_png = np.array(depth_img, dtype=np.float32)
            depth_img_png = depth_map_norm_16bitpng(depth_img_png)

            depth_img_png_filt = np.array(depth_img, dtype=np.float32)
            depth_img_png_filt = depth_map_norm_16bitpng(depth_img_png_filt, filt_invalid=True)

            depth_img_np = np.array(depth_img, dtype=np.float32)
            
            np.save(path.join(new_target_np_dir, file_name+'.npy'),depth_img_np, allow_pickle=True)
            skimg_io.imsave(path.join(new_target_png_dir, file_name+'.png'), depth_img_png)
            skimg_io.imsave(path.join(new_target_png_filt_dir, file_name+'.png'), depth_img_png_filt)

            if (i+1) %print_frequency ==0:
                print("Saved {} img number {:07d}: {}".format(orientation,i+1, path.join(new_target_png_filt_dir, file_name+'.png')))
            
            h5_file.close()


def generate_converted_depths_lists(root = "/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1/"):
    subroot = "phoenix/S6/zl548/MegaDepth_v1"
    lists_dir = 'final_list'

    splits = ['train_list', 'val_list', 'test_list']
    orientation_list = ["landscape", "portrait"]

    for split in splits:
        for orientation in orientation_list:
            depth_file = path.join(root, lists_dir, split, orientation, 'targets_MD.p')
            file = open(depth_file, 'rb')
            depth_list = pickle.load(file)
            file.close()

            png_list = []
            npy_list = []
            for depth_file in depth_list:
                base_name = path.basename(depth_file).split('.')[0]
                dir_name = path.dirname(depth_file)
                npy_path = path.join(dir_name+'_npy', base_name+'.npy')
                png_path = path.join(dir_name+'_png', base_name+'.png')
                npy_list.append(npy_path)
                png_list.append(png_path)
            
            png_list_path = path.join(root, lists_dir, split, orientation, 'targets_MD_png.p')
            npy_list_path = path.join(root, lists_dir, split, orientation, 'targets_MD_npy.p')

            png_list_file = open(png_list_path, 'wb')
            pickle.dump(png_list, png_list_file)
            png_list_file.close()
            print(f"Stored png list file: {png_list_path}")

            npy_list_file = open(npy_list_path, 'wb')
            pickle.dump(npy_list, npy_list_file)
            npy_list_file.close()
            print(f"Stored npy list file: {npy_list_path}")

def depth_map_norm_16bitpng(depth_map, filt_invalid=False):
    min_val = np.min(depth_map)
    depth_map_16 = np.copy(depth_map)

    if min_val < -0.1:
        depth_map_16[depth_map_16<0] = 2**16 - 1
    else:
        if filt_invalid:
            depth_map_16[depth_map_16 > np.percentile(depth_map_16[depth_map_16 > 1e-8], 98)] = 0
            depth_map_16[depth_map_16 < np.percentile(depth_map_16[depth_map_16>1e-8], 1)] = 0
        
        max_val = np.max(depth_map_16)
        depth_map_16 = depth_map_16 * (2**16 -2 ) / max_val
    
    return depth_map_16.astype(dtype=np.uint16)

if __name__ == "__main__":
    # Generating converted files
    start_time = time.time()
    generate_depth_files()
    delta_t = time.time()-start_time
    print ("Time take: "+ strftime("%H:%M:%S", gmtime(delta_t)))

    # Generating file lists
    start_time = time.time()
    generate_converted_depths_lists()
    delta_t = time.time()-start_time
    print ("Time take: "+ strftime("%H:%M:%S", gmtime(delta_t)))

