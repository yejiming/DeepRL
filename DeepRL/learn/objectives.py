import tensorflow as tf


_FLOATX = tf.float32
_EPSILON = 1e-10


def softmax_categorical_crossentropy(y_pred, y_true):
    with tf.name_scope("SoftmaxCrossentropy"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_true))
        return loss


def categorical_crossentropy(y_pred, y_true):
    with tf.name_scope("Crossentropy"):
        y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape())-1, keep_dims=True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(
            y_pred, tf.cast(_EPSILON, dtype=_FLOATX),
            tf.cast(1.-_EPSILON, dtype=_FLOATX)
        )
        cross_entropy = - tf.reduce_sum(
            y_true * tf.log(y_pred),
            reduction_indices=len(y_pred.get_shape())-1
        )
        return tf.reduce_mean(cross_entropy)


def binary_crossentropy(y_pred, y_true):
    with tf.name_scope("BinaryCrossentropy"):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))


def mean_square(y_pred, y_true):
    with tf.name_scope("MeanSquare"):
        return tf.reduce_mean(tf.square(y_pred - y_true))


def hinge_loss(y_pred, y_true):
    with tf.name_scope("HingeLoss"):
        return tf.reduce_mean(tf.maximum(1. - y_true * y_pred, 0.))


def roc_auc_score(y_pred, y_true):
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))