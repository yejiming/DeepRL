import tensorflow as tf

from DeepRL.learn import learn_utils as utils


def variable(shape, initializer, name, regularizer=None):
    if isinstance(initializer, str):
        initializer = utils.get_initializer(initializer)()
    return tf.get_variable(
        name, shape=shape, dtype=tf.float32,
        initializer=initializer,
        regularizer=regularizer
    )
