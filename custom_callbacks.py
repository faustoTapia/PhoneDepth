import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from depth_utils import preprocess_batch_mask
from dataset_utils.viz_utils import save_corresponding_batches_viz

class EvaluationWithSampleBatchIOViz(Callback):
    def __init__(self, batch_to_display, output_dir):
        super().__init__()
        self._display_input_imgs = batch_to_display[0]
        self._display_orig_depth = batch_to_display[1]
        self._output_dir = output_dir

    def on_test_begin(self, batch, logs=None):
        pred_log_depth = self.model.predict(self._display_input_imgs)
        pred_depth = tf.math.exp(pred_log_depth)
        
        input_imgs = self._display_input_imgs
        tar, pred, mask = preprocess_batch_mask(self._display_orig_depth, pred_depth)

        tar = tf.math.multiply_no_nan(tar, mask)
        pred = tf.math.multiply_no_nan(pred, mask)

        labels = ['img_orig', 'depth_orig', 'depth_pred']
        batches = [input_imgs, tar, pred]
        
        save_corresponding_batches_viz(batches, labels, self._output_dir, n_samples=None)
        print("Saved sample images at: " + self._output_dir)