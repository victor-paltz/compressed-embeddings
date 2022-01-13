"""Several loss functions."""

import tensorflow as tf
import tensorflow.keras.backend as K


ALPHA = 0.75
X_MAX = 100.0


@tf.function
def glove_pmi_loss(y_true, y_pred):
    """
    This is GloVe's loss function
    :param y_true: The actual values, in our case the 'observed' X_ij co-occurrence values
    :param y_pred: The predicted (log-)co-occurrences from the model
    :return: The loss associated with this batch
    """
    return K.mean(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), ALPHA) * K.square(y_pred - y_true), axis=-1)


@tf.function
def glove_log_pmi_loss(y_true, y_pred):

    epsilon = 1e-9
    return K.mean(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), ALPHA) * K.square(y_pred - K.log(y_true + epsilon)), axis=-1)


@tf.function
def mse_log_pmi(y_true, y_pred):

    epsilon = 1e-9
    return K.mean(K.square(y_pred - K.log(y_true + epsilon)), axis=-1)
