import numpy as np
import pickle
import h5py

def main():
    root = "/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1" # you need to change this line
    orientation_list = ["landscape", "portrait"]
    base_path = "/srv/beegfs02/scratch/efficient_nn_mobile/data/MegaDepth_v1/phoenix/S6/zl548/MegaDepth_v1/"

    split = "train"
    # split = "val"

    for orientation in orientation_list:

        dir_load_all_img = root +"/final_list/{}_list/".format(split) + orientation + "/imgs_MD.p"
        dir_load_all_target = root + "/final_list/{}_list/".format(split) + orientation + "/targets_MD.p"

        dir_save_val_img = root + "/wo_ordinal_list/{}_noOrdinal/".format(split) + orientation + "/imgs_MD.p"
        dir_save_val_target = root + "/wo_ordinal_list/{}_noOrdinal/".format(split) + orientation + "/targets_MD.p"

        img_list = np.load(dir_load_all_img, allow_pickle=True)
        target_list = np.load(dir_load_all_target, allow_pickle=True)

        i = 0
        while i < len(img_list):
            hdf5_file = h5py.File(base_path + target_list[i], 'r')
            gt = hdf5_file.get('depth')
            gt = np.array(gt)
            print(np.shape(gt))
            print('successfully loaded', i, "from ", len(img_list))
            if np.amin(gt) < 0:
                del img_list[i]
                del target_list[i]
            else:
                i += 1


        print("orientation: %s"%orientation)
        print("List length : %d"%(len(img_list)))

        # save validation list
        img_list_file = open(dir_save_val_img, 'wb')
        pickle.dump(img_list, img_list_file)
        img_list_file.close()

        img_list_file = open(dir_save_val_target, 'wb')
        pickle.dump(target_list, img_list_file)
        img_list_file.close()


if __name__ == "__main__":
    main()

#
