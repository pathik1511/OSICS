import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

C1 = tf.constant(70, dtype='float32')
C2 = tf.constant(1000, dtype="float32")

def competition_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # y_pred has 3 columns for quantiles 0.2, 0.5, 0.8
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt(tf.cast(2, dtype=tf.float32))
    metric = - (sq2 * delta / sigma_clip) - tf.math.log(sq2 * sigma_clip)
    return K.mean(metric)

def pinball_loss(y_true, y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)

def combined_loss(_lambda=0.8):
    def loss(y_true, y_pred):
        # We want to MINIMIZE loss, so we use -competition_metric (since it's a log-likelihood)
        return _lambda * pinball_loss(y_true, y_pred) + (1 - _lambda) * (-competition_metric(y_true, y_pred))
    return loss
