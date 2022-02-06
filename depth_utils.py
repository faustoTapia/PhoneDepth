import numpy as np
import tensorflow as tf

def make_mask(depth_map, confidence=1.0):
    mask = tf.greater(depth_map, 1.0e-8)
    mask = tf.cast(mask, tf.float32) * confidence
    return mask

def preprocess_batch_mask(tar, pred, pred_transf=None):
    if pred_transf is not None:
        pred = pred_transf(pred)
    pred = tf.keras.backend.in_train_phase(pred, tf.image.resize(pred, [tf.shape(tar)[1], tf.shape(tar)[2]], method=tf.image.ResizeMethod.BILINEAR))
    # tar = tf.keras.backend.in_train_phase(set_ordinal_to_zero(tar), tar)
    tar = set_ordinal_to_zero(tar)

    mask = make_mask(tar)
    return tar, pred, mask

def set_ordinal_to_zero(tar):
    res = tf.math.reduce_min(tar, [1, 2, 3], keepdims=True)
    res = tf.greater_equal(res, -0.1)
    res = tf.cast(res, tf.float32)
    res = tf.math.multiply_no_nan(tar, res)
    return res

def md_loss_total(tar_batch, pred_batch):

    # Accouting for double depth supervisoiin
    tar_batch = separate_target(tar_batch)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))
    [data_mask, ordinal_mask] = tf.py_function(check_ordinal, [tar_batch], [tf.float32,tf.float32])

    pred_batch_ord = tf.math.multiply(ordinal_mask, pred_batch)
    tar_batch_ord = tf.math.multiply(ordinal_mask, tar_batch)
    pred_batch = tf.math.multiply(data_mask, pred_batch)
    tar_batch = tf.math.multiply(data_mask, tar_batch)

    mask = make_mask(tar_batch)

    total_loss = 0.
    [error_o] = tf.py_function(error_ordinal, [tar_batch_ord, pred_batch_ord], [tf.float32])

    total_loss += md_error_data(tar_batch, pred_batch, mask)
    total_loss += 0.5 * md_error_gradients(tar_batch, pred_batch, mask)
    total_loss += 0.1 * error_o

    return total_loss

def md_loss_total_doubleDepth(tar_batch, pred_batch):
    
    depth_tar = separate_target(tar_batch, 1)
    grad_tar = separate_target(tar_batch, 0)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))

    [data_mask_depth, ordinal_mask_depth] = tf.py_function(check_ordinal, [depth_tar], [tf.float32,tf.float32])
    depth_tar = tf.math.multiply(data_mask_depth, depth_tar)
    pred_depth = tf.math.multiply(data_mask_depth, pred_batch)
    mask_depth = make_mask(depth_tar)

    [data_mask_grad, ordinal_mask_grad] = tf.py_function(check_ordinal, [grad_tar], [tf.float32,tf.float32])
    grad_tar = tf.math.multiply(data_mask_grad, grad_tar)
    pred_grad = tf.math.multiply(data_mask_grad, pred_batch)
    mask_grad = make_mask(grad_tar)

    total_loss = 0.0
    total_loss += md_error_data(depth_tar, pred_depth, mask_depth)
    total_loss += 0.5 * md_error_gradients(grad_tar, pred_grad, mask_grad)

    return total_loss

def md_loss_total_conf(tar_batch, pred_batch):
    
    depth_tar = separate_target(tar_batch, 1)
    conf_tar = separate_target(tar_batch, 0)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))

    [data_mask_depth, ordinal_mask_depth] = tf.py_function(check_ordinal, [depth_tar], [tf.float32,tf.float32])
    depth_tar = tf.math.multiply(data_mask_depth, depth_tar)
    pred_depth = tf.math.multiply(data_mask_depth, pred_batch)
    mask_depth = make_mask(depth_tar, confidence = conf_tar)

    total_loss = 0.0
    total_loss += md_error_data(depth_tar, pred_depth, mask_depth)
    total_loss += 0.5 * md_error_gradients(depth_tar, pred_depth, mask_depth)

    return total_loss

def md_loss_total_doubleDepthConf(tar_batch, pred_batch):
    
    depth_tar = separate_target(tar_batch, 2)
    conf_tar = separate_target(tar_batch, 1)
    grad_tar = separate_target(tar_batch, 0)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))

    [data_mask_depth, ordinal_mask_depth] = tf.py_function(check_ordinal, [depth_tar], [tf.float32,tf.float32])
    depth_tar = tf.math.multiply(data_mask_depth, depth_tar)
    pred_depth = tf.math.multiply(data_mask_depth, pred_batch)
    mask_depth = make_mask(depth_tar, confidence = conf_tar)

    [data_mask_grad, ordinal_mask_grad] = tf.py_function(check_ordinal, [grad_tar], [tf.float32,tf.float32])
    grad_tar = tf.math.multiply(data_mask_grad, grad_tar)
    pred_grad = tf.math.multiply(data_mask_grad, pred_batch)
    mask_grad = make_mask(grad_tar)

    total_loss = 0.0
    total_loss += md_error_data(depth_tar, pred_depth, mask_depth)
    total_loss += 0.5 * md_error_gradients(grad_tar, pred_grad, mask_grad)

    return total_loss

def md_loss_noOrd(tar_batch, pred_batch):

    tar_batch = separate_target(tar_batch)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))
    [data_mask, ordinal_mask] = tf.py_function(check_ordinal, [tar_batch], [tf.float32,tf.float32])

    pred_batch_ord = tf.math.multiply(ordinal_mask, pred_batch)
    tar_batch_ord = tf.math.multiply(ordinal_mask, tar_batch)
    pred_batch = tf.math.multiply(data_mask, pred_batch)
    tar_batch = tf.math.multiply(data_mask, tar_batch)

    mask = make_mask(tar_batch)

    total_loss = 0.

    total_loss += md_error_data(tar_batch, pred_batch, mask)
    total_loss += 0.5 * md_error_gradients(tar_batch, pred_batch, mask)

    return total_loss

def md_error_data(tar_batch, pred_batch, mask):
    n = tf.reduce_sum(mask, [1, 2, 3]) #can be zero for some layers of the maps -> the layers that represent the ordinal pics
    log_pred = pred_batch
    log_tar = tf.math.log(tar_batch)
    log_dif = tf.math.subtract(log_pred, log_tar)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)
    s1 = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.pow(log_dif, 2), [1,2,3]) , n)
    s2 = tf.math.divide_no_nan(tf.math.pow(tf.math.reduce_sum(log_dif, [1,2,3]), 2) , (n*n))
    res = s1 - s2
    count_nonOrdinalBatches = tf.cast(tf.math.count_nonzero(n), tf.float32)
    loss_data = tf.math.divide_no_nan(tf.math.reduce_sum(res), count_nonOrdinalBatches)

    return loss_data

def md_error_gradients(tar_batch, pred_batch, mask):
    log_tar = tf.math.log(tar_batch)
    log_dif = tf.math.subtract(pred_batch, log_tar)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)

    sum = 0.0
    n = tf.reduce_sum(mask)
    for scale in [1, 2, 4, 8]:#32?
    # for scale in [2, 4, 8, 16]:#32?
        dy, dx = image_gradients_abs(log_dif, scale, mask)
        sum += tf.reduce_sum(dy) + tf.reduce_sum(dx)

    return tf.math.divide_no_nan(sum, n)

def image_gradients_abs(image, scale, mask):
    # mask_y = tf.multiply(mask[:, scale::scale,:,:], mask[:, :-scale:scale, :,:])
    # dy = tf.abs(image[:, scale::scale, :, :] - image[:, :-scale:scale, :, :])
    # dy = tf.multiply(dy, mask_y)
    # mask_x = tf.multiply(mask[:, :, scale::scale,:], mask[:, :, :-scale:scale,:])
    # dx = tf.abs(image[:, :, scale::scale, :] - image[:, :, :-scale:scale, :])
    # dx = tf.multiply(dx, mask_x)

    img = image[:, ::scale, ::scale, :]
    mask_down = mask[:, ::scale, ::scale]
    dy =tf.abs(img[:, 1:] - img[:,:-1])
    mask_y = tf.multiply(mask_down[:, 1:],mask_down[:, :-1])
    dy = tf.math.multiply_no_nan(dy, mask_y)
    dx = tf.abs(img[:,:,1:] - img[:,:,:-1])
    mask_x = tf.multiply(mask_down[:,:,1:], mask_down[:,:,:-1])
    dx = tf.math.multiply_no_nan(dx, mask_x)

    return dy, dx

def check_ordinal(tar_batch):

    # tar = tar_batch.numpy()
    # batch_size = np.shape(tar)[0]
    # for i in range(batch_size):
    #     if np.amin(tar[i]) < -0.1: #if negative value, should be -1 is present in picture, it is an ordinal depth map
    #         tar[i] = 0.0
    #     else:
    #         tar[i] = 1.0

    # data_mask = tf.convert_to_tensor(tar, tf.float32)
    # ordinal_mask = tf.convert_to_tensor(np.float32(tar < 0.5), tf.float32)

    data_mask = tf.ones_like(tar_batch)
    data_indicator = tf.reduce_min(tar_batch, [1,2,3], keepdims=True)
    data_indicator = data_indicator >= -0.1
    data_indicator = tf.cast(data_indicator, tf.float32)
    data_mask = tf.math.multiply_no_nan(data_mask, data_indicator)
    
    ordinal_mask = tf.ones_like(tar_batch)
    ordinal_indicator = tf.reduce_min(tar_batch, [1,2,3], keepdims=True)
    ordinal_indicator = ordinal_indicator < -0.1
    ordinal_indicator = tf.cast(ordinal_indicator, tf.float32)
    ordinal_mask = tf.math.multiply_no_nan(ordinal_mask, ordinal_indicator)

    return [data_mask, ordinal_mask]

def error_ordinal_wrap(tar_batch, pred_batch):
    
    tar_batch = separate_target(tar_batch)

    pred_batch = tf.keras.backend.in_train_phase(pred_batch, tf.image.resize(pred_batch, [tf.shape(tar_batch)[1], tf.shape(tar_batch)[2]], method=tf.image.ResizeMethod.BILINEAR))
    [data_mask, ordinal_mask] = tf.py_function(check_ordinal, [tar_batch], [tf.float32,tf.float32])

    pred_batch_ord = tf.math.multiply(ordinal_mask, pred_batch)
    tar_batch_ord = tf.math.multiply(ordinal_mask, tar_batch)

    [error_o] = tf.py_function(error_ordinal, [tar_batch_ord, pred_batch_ord], [tf.float32])
    
    return error_o

def error_ordinal(tar, pred):

    tar = separate_target(tar)

    pred = pred.numpy()[:,:,:,0]#since this are 4D arrays with batch_sizexhxwxd and d=1
    tar = tar.numpy()[:,:,:,0]
    batch_size = np.shape(tar)[0]
    count_ordinal_img = 0
    res = 0.
    tau = 0.25
    # c = tf.math.log(1 + tf.math.exp(tau)) - tf.math.log(1 + tf.math.exp(tf.math.sqrt(tau)))
    c = -0.14813757

    for i in range(batch_size):
        if (np.amin(tar[i]) < -0.5):
            [y_b_list, x_b_list] = np.where(tar[i] < 0)
            [y_f_list, x_f_list] = np.where(tar[i] > 0)

            if y_f_list.shape[0] < 100 or y_b_list.shape[0] < 100:
                continue
            count_ordinal_img += 1
            b_id = np.random.randint(0, y_b_list.shape[0])
            f_id = np.random.randint(0, y_f_list.shape[0])

            y_f = y_f_list[f_id]
            x_f = x_f_list[f_id]
            y_b = y_b_list[b_id]
            x_b = x_b_list[b_id]

            p = -(pred[i, y_b, x_b] - pred[i, y_f, x_f])

            if p <= tau:
                res += tf.math.log(1 + tf.math.exp(p))
            else:
                res += tf.math.log(1 + tf.math.exp(tf.math.sqrt(p))) + c

    if count_ordinal_img > 0:
        res /= count_ordinal_img
    return tf.cast(res, tf.float32)

def average_rel_error(tar, pred):
    tar = separate_target(tar)
    tar, pred, mask = preprocess_batch_mask(tar, pred)
    n = tf.math.reduce_sum(mask) + 1e-8

    pred = rescale_nonlog(tar, pred, mask)

    dif = tf.math.abs(tf.math.subtract(tar, pred))
    rel = tf.math.divide_no_nan(dif, tar)
    rel_av = tf.math.reduce_sum(tf.math.multiply(rel, mask)) / n

    return rel_av

def average_log10_error(tar, pred):
    tar = separate_target(tar)
    tar, pred, mask = preprocess_batch_mask(tar, pred)
    n = tf.math.reduce_sum(mask) + 1e-8

    pred = rescale_nonlog(tar, pred, mask)

    log10 = tf.math.log(10.0)
    log_tar = tf.math.log(tar) / log10
    log_pred = tf.math.log(pred) / log10
    abs_dif = tf.abs(log_tar - log_pred)
    res = tf.math.reduce_sum(tf.math.multiply_no_nan(abs_dif, mask)) / n

    return res


def pred_rescaled_nonlog(tar, pred):

    tar = separate_target(tar)

    mask = tf.py_function(make_mask, [tar], tf.float32)
    n = tf.math.reduce_sum(mask) + 1e-8

    log_tar = tf.math.log(tar)
    log_dif = tf.math.subtract(log_tar, pred)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)
    lsq_shift = tf.math.reduce_sum(log_dif) / n
    pred_shift = pred + lsq_shift
    pred_nonlog = tf.math.exp(pred_shift)
    return pred_nonlog

def rescale_nonlog(tar, pred, mask):
    n = tf.math.reduce_sum(mask) + 1e-8
    log_tar = tf.math.log(tar)
    log_dif = tf.math.subtract(log_tar, pred)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)
    lsq_shift = tf.math.reduce_sum(log_dif) / n
    pred_shift = pred + lsq_shift
    pred_nonlog = tf.math.exp(pred_shift)
    return pred_nonlog

def rmse_nonlog(tar, pred):

    tar = separate_target(tar)

    tar, pred, mask = preprocess_batch_mask(tar, pred)
    n = tf.math.reduce_sum(mask) + 1e-8

    pred_nonlog = rescale_nonlog(tar, pred, mask)

    squares = tf.math.multiply_no_nan(tf.math.pow(tar - pred_nonlog, 2), mask)
    loss = tf.math.reduce_sum(squares) / n
    loss = tf.math.sqrt(loss)

    return loss

def rmse_nonlog_wscale(tar, pred):

    tar = separate_target(tar)

    tar, pred, mask = preprocess_batch_mask(tar, pred)
    n = tf.math.reduce_sum(mask) + 1e-8

    pred_up_to_scale = tf.math.exp(pred)
    rmse = tf.math.multiply(tf.math.pow(tar - pred_up_to_scale, 2), mask) 
    rmse = tf.math.divide(tf.reduce_sum(rmse),n)
    rmse = tf.math.sqrt(rmse)

    return rmse

def si_mse_mask(tar_batch, pred_batch, mask):
    n = tf.reduce_sum(mask, [1, 2, 3])

    log_pred = pred_batch
    log_tar = tf.math.log(tar_batch)
    log_dif = tf.math.subtract(log_pred, log_tar)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)
    s1 = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.pow(log_dif, 2), [1,2,3]) , n)
    s2 = tf.math.divide_no_nan(tf.math.pow(tf.math.reduce_sum(log_dif, [1,2,3]), 2) , (n*n))
    res = s1 - s2
    n_nonOrd_imgs = tf.cast(tf.math.count_nonzero(n), tf.float32)
    loss_data = tf.math.divide_no_nan(tf.math.reduce_sum(res), n_nonOrd_imgs)

    return loss_data

def si_mse(tar_batch, pred_batch):

    tar_batch = separate_target(tar_batch)

    tar_batch, pred_batch, mask = preprocess_batch_mask(tar_batch, pred_batch)
    return si_mse_mask(tar_batch, pred_batch, mask)

def si_rmse(tar_batch, pred_batch):

    tar_batch = separate_target(tar_batch)

    return tf.math.sqrt(si_mse(tar_batch, pred_batch))

def si_mse_balanced_mask(tar_batch, pred_batch, mask, balance_fact=0.85):
    n = tf.reduce_sum(mask, [1, 2, 3])

    log_pred = pred_batch
    log_tar = tf.math.log(tar_batch)
    log_dif = tf.math.subtract(log_pred, log_tar)
    log_dif = tf.math.multiply_no_nan(log_dif, mask)
    s1 = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.pow(log_dif, 2), [1,2,3]) , n)
    s2 = tf.math.divide_no_nan(tf.math.pow(tf.math.reduce_sum(log_dif, [1,2,3]), 2) , (n*n))
    res = s1 - balance_fact * s2
    count_nonOrdinalBatches = tf.cast(tf.math.count_nonzero(n), tf.float32)
    loss_data = tf.math.divide_no_nan(tf.math.reduce_sum(res), count_nonOrdinalBatches)
    return loss_data

def si_rmse_balanced_func(balance_fact = 0.85):
    def si_rmse_balanced(tar_batch, pred_batch):
        tar_batch = separate_target(tar_batch)
        tar_batch, pred_batch, mask = preprocess_batch_mask(tar_batch, pred_batch)
        return tf.math.sqrt(si_mse_balanced_mask(tar_batch, pred_batch, mask, balance_fact))
    return si_rmse_balanced


def compute_scale_and_shift(target, prediction, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    mask_f = tf.cast(mask, target.dtype)
    a_00 = tf.reduce_sum(mask_f * prediction * prediction, (1, 2), keepdims=True)
    a_01 = tf.reduce_sum(mask_f * prediction, (1, 2), keepdims=True)
    a_11 = tf.reduce_sum(mask_f, (1, 2), keepdims=True)

    # right hand side: b = [b_0, b_1]
    b_0 = tf.reduce_sum(mask_f * prediction * target, (1, 2), keepdims=True)
    b_1 = tf.reduce_sum(mask_f * target, (1, 2), keepdims=True)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    det = a_00 * a_11 - a_01 * a_01
    zero = tf.constant(0, dtype=det.dtype)
    valid = tf.not_equal(det, zero)

    x_0= (a_11 * b_0 - a_01 * b_1) / det
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    x_0 = tf.where(valid, x_0, 0.0)
    x_1 = tf.where(valid, x_1, 0.0)

    return x_0, x_1

# Mean Squared Error function with mask
def mse_mask(target, prediction, mask):
    mask_casted = tf.cast(mask, target.dtype)
    n = tf.reduce_sum(mask_casted, (1,2))
    res = prediction - target
    image_loss = tf.reduce_sum(mask_casted * res * res)
    return tf.math.divide_no_nan(image_loss, (2 * n))

def mse_loss(target, prediction):
    target = separate_target(target)
    target, prediction, mask = preprocess_batch_mask(target, prediction,
                                    pred_transf=tf.math.exp)
    return mse_mask(target, prediction, mask)

# To be used only with mask. I.e. not directly as a loss for training
def gradient_loss_mask(target, prediction, mask):
    mask_casted = tf.cast(mask, target.dtype)
    tar_log = tf.math.log(target)
    pred_log = prediction

    diff = pred_log - tar_log
    diff = tf.math.multiply_no_nan(diff, mask_casted)

    grad_x = tf.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = tf.multiply(mask_casted[:, :, 1:], mask_casted[:, :, :-1])
    grad_x = tf.multiply(mask_x, grad_x)

    grad_y = tf.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = tf.multiply(mask_casted[:, 1:, :], mask_casted[:, :-1, :])
    grad_y = tf.multiply(mask_y, grad_y)

    batch_loss = tf.reduce_sum(grad_x) + tf.reduce_sum(grad_y)

    return batch_loss

def gradient_err_mask(target, prediction, mask, scale):
    log_tar = tf.math.log(target)
    log_pred = prediction
    log_diff = tf.math.subtract(log_pred, log_tar)
    log_diff = tf.math.multiply_no_nan(log_diff, mask)

    dy, dx = image_gradients_abs(log_diff, scale, mask)
    err = tf.math.reduce_sum(dy) + tf.math.reduce_sum(dx)
    return err

def gradient_loss_multiscale_mask(target, prediction, mask, scales=1):
    total = tf.constant(0.0, dtype=target.dtype)
    n = tf.reduce_sum(mask)
    for scale_num in range(scales):
        scale = tf.pow(2, scale_num)
        total += gradient_err_mask(target, prediction, mask, scale)
    return tf.math.divide_no_nan(total, n)

# Defines a multiscale gradient function given number of scales
def gradient_loss_multiscale_func(scales = 4):
    def gradient_loss_multiscale(target, prediction):
        target = separate_target(target)
        # Note pred_transf is given assuming that network outputs log-depth
        target, prediction, mask = preprocess_batch_mask(target, prediction)

        return gradient_loss_multiscale_mask(target, prediction, mask, scales)
    return gradient_loss_multiscale


# Scale and Shift invariant loss function
def ssi_loss_func(alpha = 0.5, scales = 4):
    data_loss = mse_mask
    regularization_loss = gradient_loss_multiscale_mask

    def ssi_loss(target, prediction):
        target = separate_target(target)
        target, prediction, mask = preprocess_batch_mask(target, prediction,
                                        pred_transf=tf.math.exp)

        scale, shift = compute_scale_and_shift(target=target,
                                                prediction=prediction,
                                                mask=mask)
        prediction_ssi = scale * prediction + shift

        total_loss = data_loss(target, prediction_ssi, mask)
        if alpha > 0:
            total_loss += alpha * regularization_loss(target, prediction_ssi, scales, mask)

        return total_loss

    return ssi_loss

def parkmai_total_loss_func(si_balance=0.85, ssi_alpha=0.5, ssi_scales=4, 
                        si_weight = 2.0, ssi_weight=10.0, mse_weight=10.0):
    loss_si = si_rmse_balanced_func(balance_fact=si_balance)
    loss_ssi = ssi_loss_func(alpha=ssi_alpha, scales=ssi_scales)
    loss_mse = mse_loss

    def parkmai_total_loss(target, prediction):
        return si_weight*loss_si(target,prediction) + ssi_weight*loss_ssi(target,prediction) + mse_weight*loss_mse(target, prediction)

    return parkmai_total_loss

# ==========================================================================

def uint8_conv_mask(img, mask):
    mask_bool = tf.not_equal(mask, 0)
    relev_values = img[mask_bool]
    shift = tf.math.reduce_min(relev_values)
    relev_values = relev_values - shift
    range = tf.math.reduce_max(relev_values)
    img_rsc = tf.math.multiply(img - shift, 255.0/range)
    img_rsc = tf.math.multiply(img_rsc, mask)
    img_uint8 = tf.cast(img_rsc, tf.uint8)

    return img_uint8


def separate_target(tar_tensor, indx=1):
    if tar_tensor.shape[-1] > 1:
        return tf.expand_dims(tar_tensor[:,:,:, indx], -1)
    else:
        return tar_tensor

if __name__=="__main__":
    shape= (3,64,64,1)
    seed_num= 125

    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    targ_np = np.random.random(shape)
    # Add ordinal label
    targ_np[2,:,:,:] = 0.0
    targ_np[2,10:30,10:30,:] = -1.0
    targ_np[2,40:60,40:50,:] = 2.0
    
    targ = tf.convert_to_tensor(targ_np, tf.float32)

    pred = targ + tf.random.normal(shape) * 0.2
    neg_mask = pred>0
    pred = tf.where(neg_mask, pred, 1.0e-8)
    log_pred = tf.math.log(pred)


    data_mask, ordinal_mask = check_ordinal(targ)

    mask = tf.cast(targ>0.0, dtype=tf.float32) * data_mask
    err_md = md_error_gradients(targ*data_mask, log_pred*data_mask, mask)
    err_mine = gradient_loss_multiscale_mask(targ*data_mask, log_pred*data_mask, 4, mask)

    print(f"MD grad loss: {err_md}")
    print(f"My grad loss: {err_mine}")

    print(f"---------Testing MD loss---------")

    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    tot_error_md = md_loss_total(targ, pred)
    data_loss = si_mse(targ, pred)
    grad_loss = gradient_loss_multiscale_func()(targ, pred)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    ord_loss = error_ordinal_wrap(targ, pred)
    print(f"Tot md loss: {tot_error_md}")
    print(f"Data md loss: {data_loss}")
    print(f"Grad md loss: {grad_loss}")
    print(f"Ord md loss: {ord_loss}")
    np.random.seed(seed_num)