from __future__ import division
import numpy as np
import sys
import random
import os
import shutil
import pickle 
import h5py
from tqdm import tqdm
import time
from time import gmtime
from time import strftime


def main():
    only_No_ordinal = False
    root = "/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1/" # you need to change this line
    subroot = "phoenix/S6/zl548/MegaDepth_v1"
    orientation_list = ["landscape", "portrait"]

    ratio = 0.03

    for orientation in orientation_list:

        dir_load_all_img = root +"/final_list/train_val_list/" + orientation + "/imgs_MD.p"
        dir_load_all_target = root + "/final_list/train_val_list/" + orientation + "/targets_MD.p"

        dir_save_train_img = root + "w_ordinal_list/train_list/" + orientation + "/imgs_MD.p"
        dir_save_train_target = root + "w_ordinal_list/train_list/" + orientation + "/targets_MD.p"

        dir_save_val_img = root + "w_ordinal_list/val_list/" + orientation + "/imgs_MD.p"
        dir_save_val_target = root + "w_ordinal_list/val_list/" + orientation + "/targets_MD.p"

        img_list = np.load(dir_load_all_img, allow_pickle=True)
        target_list = np.load(dir_load_all_target, allow_pickle=True)

        val_num = int(round(ratio * len(img_list)) )

        shuffle_list = list(range(len(img_list)))
        random.shuffle(shuffle_list)

        train_img_list = []
        train_targets_list =[]
        val_img_list = []
        val_targets_list =[]

        for i in tqdm(range(0, val_num), desc='Loading Validation Set: '):
            target_path = os.path.join(root, subroot, target_list[shuffle_list[i]])

            if only_No_ordinal:
                hdf5_file = h5py.File(target_path, 'r')
                gt = hdf5_file.get('depth')
                gt = np.array(gt)
                if np.amin(gt) < 0:
                    continue
                else:
                    pass

            val_targets_list.append(target_list[shuffle_list[i]])
            val_img_list.append(img_list[shuffle_list[i]])

        for i in tqdm(range(val_num, len(img_list)), desc='Loanding Training Set: '):
            target_path = os.path.join(root, subroot, target_list[shuffle_list[i]])

            if only_No_ordinal:
                hdf5_file = h5py.File(target_path, 'r')
                gt = hdf5_file.get('depth')
                gt = np.array(gt)
                if np.amin(gt) < 0:
                    continue
                else:
                    pass

            train_targets_list.append(target_list[shuffle_list[i]])
            train_img_list.append(img_list[shuffle_list[i]])

        print("orientation: %s"%orientation)
        print("train list length : %d"%(len(train_img_list)))
        print("validation list length : %d"%(len(val_img_list)))


        # save train list
        img_list_file = open(dir_save_train_img, 'wb')
        pickle.dump(train_img_list, img_list_file)
        img_list_file.close()

        img_list_file = open(dir_save_train_target, 'wb')
        pickle.dump(train_targets_list, img_list_file)    
        img_list_file.close()


        # save validation list
        img_list_file = open(dir_save_val_img, 'wb')
        pickle.dump(val_img_list, img_list_file)
        img_list_file.close()

        img_list_file = open(dir_save_val_target, 'wb')
        pickle.dump(val_targets_list, img_list_file)    
        img_list_file.close()



if __name__ == "__main__":
    start_time = time.time()
    main()
    delta_t = time.time()-start_time
    print ("Time take: "+ strftime("%H:%M:%S", gmtime(delta_t)))

# 
