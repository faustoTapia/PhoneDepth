import tensorflow as tf

from depth_utils import md_loss_total, md_loss_total_doubleDepth, md_loss_noOrd, si_rmse, si_mse, rmse_nonlog
from depth_utils import md_loss_total_conf, md_loss_total_doubleDepthConf
from depth_utils import gradient_loss_multiscale_func, error_ordinal_wrap
from depth_utils import average_rel_error, average_log10_error

def compile_md(model, learning_rate=5e-5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = md_loss_total
    metrics = [si_rmse, gradient_loss_multiscale_func(), error_ordinal_wrap, si_mse, rmse_nonlog, average_rel_error, average_log10_error]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_doubleDepth(model, learning_rate=5e-5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = md_loss_total_doubleDepth
    metrics = [si_rmse, gradient_loss_multiscale_func(), si_mse, rmse_nonlog, average_rel_error, average_log10_error]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_conf(model, learning_rate=5e-5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = md_loss_total_conf
    metrics = [si_rmse, gradient_loss_multiscale_func(), si_mse, rmse_nonlog, average_rel_error, average_log10_error]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_doubleDepthConf(model, learning_rate=5e-5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = md_loss_total_doubleDepthConf
    metrics = [si_rmse, gradient_loss_multiscale_func(), si_mse, rmse_nonlog, average_rel_error, average_log10_error]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def compile_md_no_ord(model, learning_rate=5e-5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = md_loss_noOrd
    metrics = [si_rmse, gradient_loss_multiscale_func(), rmse_nonlog]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
