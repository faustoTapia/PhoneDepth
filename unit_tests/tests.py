import unittest
import tensorflow as tf
import numpy as np

import sys
sys.path.append('./')

from depth_utils import si_rmse, si_mse, md_error_gradients, error_ordinal
from depth_utils import gradient_loss_multiscale_func, md_error_gradients, md_loss_total_doubleDepth
from depth_utils import make_mask, check_ordinal

from dataset_utils.aug_utils import salty_noise, color_jitter, random_crop_and_resize

class TestLossFunctions(unittest.TestCase):
                 
    def test_si_rmse(self):
        # Calculated manually
        result = 0.083394933
        gt = np.array([
                        [[1.0, 2.0 ], [0, 3.3]],
                        [[-1.0, 5.0], [3.3, 4.1]],
                        [[1.3, 5.5], [7.7, 8]]
                    ])
        gt = np.array(gt, dtype=np.float32)
        gt = np.expand_dims(gt, -1)
        pred = np.array([
                        [[1.3, 2.1 ], [1.5, 3.5]],
                        [[0.5, 5.5], [3.3, 4.1]],
                        [[1.5, 5.4], [8.1, 7.9]]
                    ])
        pred = np.array(pred, dtype=np.float32)
        pred = np.expand_dims(pred, -1)
        pred_log = np.log(pred)

        loss = si_rmse(gt, pred_log)
        
        self.assertAlmostEqual(loss, result,
                                msg=f"Obtained result {loss.numpy()} is not expected result {result}.",
                                delta=1e-7)

        # Calculated manually
        result = 0.083394933
        gt = np.array([
                        [[1.0, 2.0 ], [0, 3.3]],
                        [[-1.0, 5.0], [3.3, 4.1]],
                        [[1.3, 5.5], [7.7, 8]]
                    ])
        gt = np.array(gt, dtype=np.float32)
        gt = np.expand_dims(gt, -1)
        gt = np.concatenate([np.random.rand(*gt.shape), gt], axis=-1)
        gt = np.array(gt, dtype=np.float32)
        pred = np.array([
                        [[1.3, 2.1 ], [1.5, 3.5]],
                        [[0.5, 5.5], [3.3, 4.1]],
                        [[1.5, 5.4], [8.1, 7.9]]
                    ])
        pred = np.array(pred, dtype=np.float32)
        pred = np.expand_dims(pred, -1)
        pred_log = np.log(pred)

        loss = si_rmse(gt, pred_log)
        
        self.assertAlmostEqual(loss, result,
                                msg=f"Obtained result {loss.numpy()} is not expected result {result}.",
                                delta=1e-7)

    def test_md_total_loss_double(self):
        gt = np.array([
                        [[1.0, 2.0 ], [0, 3.3]],
                        [[-1.0, 5.0], [3.3, 4.1]],
                        [[1.3, 5.5], [7.7, 8]]
                    ])
        gt = np.array(gt, dtype=np.float32)
        gt = np.expand_dims(gt, -1)
        gt = np.concatenate([np.random.rand(*gt.shape), gt], axis=-1)
        gt = np.array(gt, dtype=np.float32)
        pred = np.array([
                        [[1.3, 2.1 ], [1.5, 3.5]],
                        [[0.5, 5.5], [3.3, 4.1]],
                        [[1.5, 5.4], [8.1, 7.9]]
                    ])
        pred = np.array(pred, dtype=np.float32)
        pred = np.expand_dims(pred, -1)
        pred_log = np.log(pred)

        loss = md_loss_total_doubleDepth(gt, pred_log)
        
        self.assertTrue(np.isfinite(loss), msg='Calculated loss needs to be finenite, given: {}'.format(loss))

    def test_grad_loss(self):
        gt1 = np.concatenate((np.arange(0, 8), np.arange(0, 8)[::-1]), axis = 0)
        gt1 = np.expand_dims(gt1, 0)
        gt1 = np.array(np.repeat(gt1, 16, axis=0), dtype=np.float32)
        gt1 = np.expand_dims(gt1, -1)
        gt1 = np.expand_dims(gt1, 0)
        gt1 = np.exp(gt1)                   # Just did for testing purposees, easier to interpret if values are right

        gt2 = np.array(np.random.randint(0, 7, size=(16,16)), dtype=np.float32)
        gt2[0:8][0:8] = -1.0
        gt2 = np.expand_dims(gt2, -1)
        gt2 = np.expand_dims(gt2, 0)

        gt3 = np.concatenate((np.arange(0, 8), np.arange(0, 8)[::-1]), axis = 0)
        gt3 = np.expand_dims(gt3, 1)
        gt3 = np.array(np.repeat(gt3, 16, axis=1), dtype=np.float32)
        gt3 = np.expand_dims(gt3, -1)
        gt3 = np.expand_dims(gt3, 0)
        gt3 = np.exp(gt3)

        gt = np.concatenate((gt1, gt2, gt3), axis=0)

        pred = tf.ones_like(gt)


        pred_batch = tf.keras.backend.in_train_phase(pred, tf.image.resize(pred, [tf.shape(gt)[1], tf.shape(gt)[2]], method=tf.image.ResizeMethod.BILINEAR))
        [data_mask, ordinal_mask] = tf.py_function(check_ordinal, [gt], [tf.float32,tf.float32])

        pred_batch = tf.math.multiply(data_mask, pred)
        tar_batch = tf.math.multiply(data_mask, gt)

        mask = make_mask(tar_batch)

        loss_grad_md = md_error_gradients(tar_batch, pred_batch, mask)
        print("Loss grad md: {}".format(loss_grad_md))
        loss_grad_mine = gradient_loss_multiscale_func()(gt, pred)
        print("Loss grad mine: {}".format(loss_grad_mine))
        
        self.assertAlmostEqual(loss_grad_md, loss_grad_mine,
                                msg=f"Obtained result {loss_grad_md.numpy()} is not expected result {loss_grad_mine.numpy()}.",
                                delta=1e-7)

    def test_ord_loss(self):
        c = -0.14813757
        def err_func(x_b, x_f):
            if x_f - x_b < 0.25:
                return tf.math.log( 1 + tf.math.exp(x_f - x_b))
            else:
                return tf.math.log(1 + tf.math.exp(tf.math.sqrt(x_f - x_b))) + c

        val_1 = 0.7
        val_2 = 512

        pred = np.kron(np.array([[val_1, 3], [0.8, val_2]]), np.ones((20,20)))
        pred_tf = tf.convert_to_tensor(pred)
        pred_tf = tf.expand_dims(pred_tf, 0)
        pred_tf = tf.expand_dims(pred_tf, -1)
        
        tar = np.kron(np.array([[-1,0], [0, 2]]), np.ones((20, 20)))
        tar_tf = tf.convert_to_tensor(tar)
        tar_tf = tf.expand_dims(tar_tf, 0)
        tar_tf = tf.expand_dims(tar_tf, -1)

        print("Pred: \n{}".format(pred))

        print("Neg tar: \n{}".format(tar))
        err_ord_neg = error_ordinal(tar_tf, pred_tf)
        err_ord_neg_calc = err_func(val_1, val_2)
        print(f"Error neg: {err_ord_neg}, {err_ord_neg_calc}")

        tar = np.kron(np.array([[2.0,0], [0, -1.0]]), np.ones((20, 20)))
        tar_tf = tf.convert_to_tensor(tar)
        tar_tf = tf.expand_dims(tar_tf, 0)
        tar_tf = tf.expand_dims(tar_tf, -1)

        err_ord_pos = error_ordinal(tar_tf, pred_tf)
        err_ord_pos_calc = err_func(val_2, val_1)
        print("Pos tar \n{}".format(tar))
        print(f"Error pos: {err_ord_pos}, {err_ord_pos_calc}")
        self.assertAlmostEqual(err_ord_pos, err_ord_pos_calc,
                                msg = f"Obtained result {err_ord_pos} is not expected result {err_ord_pos_calc}",
                                delta=1e-6)

        self.assertAlmostEqual(err_ord_neg, err_ord_neg_calc,
                                msg = f"Obtained result {err_ord_neg} is not expected result {err_ord_neg_calc}",
                                delta=1e-6)

class TestAugmentation(unittest.TestCase):
    def test_salty_noise(self):
        dist_threshold = 0.995
        noise = salty_noise(dist_threshold)
        shape = (2,500,500,3)

        x = tf.zeros(shape, dtype = tf.uint8)
        y = noise(x)

        avg = tf.reduce_mean(tf.cast(y, tf.float32))
        expected_avg = 127.5 * ((1-dist_threshold)*2)
        delta = expected_avg*0.05

        self.assertAlmostEqual(avg, expected_avg,
                                msg=f"Invalid distribution of salty noise: Expected {expected_avg}, but resulted in {avg}",
                                delta=delta)


    def test_color_jitter(self):
        jitter = color_jitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        x = tf.zeros([3,50,50,3])
        y = jitter(x)

        self.assertEqual(x.shape, y.shape,
                        msg=f"Jitter input size= {x.shape} != output size= {y.shape}")
    
    def test_rot_crop(self):
        img_shape = (2,480,640,3)
        destination_shape = (128,160)

        batch = tf.random.uniform(img_shape)
        transform = random_crop_and_resize(0.5, min_size=0.5, max_size=1.0, img_shape=destination_shape)

        augmented_batch = transform(batch)
        for i in range(img_shape[0]):
            self.assertEqual(tuple(augmented_batch[i].shape), (*destination_shape, img_shape[-1]),
                        msg=f"Output shape {augmented_batch[i].shape} != desired shape {destination_shape}")

if __name__=="__main__":
    unittest.main()