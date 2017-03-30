import math
import tensorflow as tf


def zeros(shape=None, dtype=tf.float32, seed=None):
    if shape:
        return tf.zeros(shape, dtype=dtype)
    else:
        return tf.constant_initializer(0.)


def uniform(shape=None, minval=0, maxval=None, dtype=tf.float32, seed=None):
    if shape:
        return tf.random_uniform(shape, minval=minval, maxval=maxval,
                                 seed=seed, dtype=dtype)
    else:
        return tf.random_uniform_initializer(minval=minval, maxval=maxval,
                                             seed=seed, dtype=dtype)


def uniform_scaling(shape=None, factor=1.0, dtype=tf.float32, seed=None):
    if shape:
        input_size = 1.0
        for dim in shape[:-1]:
            input_size *= float(dim)
        max_val = math.sqrt(3 / input_size) * factor
        return tf.random_ops.random_uniform(shape, -max_val, max_val,
                                            dtype, seed=seed)
    else:
        return tf.uniform_unit_scaling_initializer(seed=seed, dtype=dtype)


def normal(shape=None, mean=0.0, stddev=0.02, dtype=tf.float32, seed=None):
    if shape:
        return tf.random_normal(shape, mean=mean, stddev=stddev, seed=seed,
                                dtype=dtype)
    else:
        return tf.random_normal_initializer(mean=mean, stddev=stddev,
                                            seed=seed, dtype=dtype)


def truncated_normal(shape=None, mean=0.0, stddev=0.02, dtype=tf.float32,
                     seed=None):
    if shape:
        return tf.truncated_normal(shape=shape, mean=mean, stddev=stddev,
                                   seed=seed, dtype=dtype)
    else:
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev,
                                               seed=seed, dtype=dtype)
