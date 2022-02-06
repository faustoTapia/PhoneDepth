from depth_utils import *
import tensorflow as tf

import glob
import imageio
from scipy.io import loadmat
import h5py
import datetime
import os

def load_newest_weights(model, checkpoint_dir, best_checkpoint=None):
    initial_epoch = 0
    weight_files = sorted(glob.glob( checkpoint_dir + '/*.index'))
    if weight_files:
        if best_checkpoint is None:
            newest_weights = ".".join(weight_files[-1].split('.')[:-1])
            initial_epoch = int(newest_weights[-3:])
            print('Loading weights ' + newest_weights + ' successfully, with epoch: ', initial_epoch)
            model.load_weights(newest_weights)
        else:
            initial_epoch = int(best_checkpoint)
            newest_weights = ".".join(weight_files[-1].split('.')[:-1])[:-3] + "{:03d}".format(initial_epoch)
            print('Loading weights ' + newest_weights + ' successfully, with epoch: ', initial_epoch)
            model.load_weights(newest_weights)
        print("Finished loading weights")
    else:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    return initial_epoch

def setup_log(model_name, log_dir='C:/Users/Richu/Documents/thesis/python/depth/logs'):
    log_files = sorted(glob.glob(log_dir + '/{}/*'.format(model_name)))
    if log_files:
        log_dir = log_files[-1]
        print("successfully continue log: " + log_dir)
    else:
        log_dir = log_dir + "/{}/".format(model_name) + datetime.datetime.now().strftime("%m%d-%H%M")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    return log_dir
