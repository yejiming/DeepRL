from inspect import getmembers

import numpy as np
import tensorflow as tf
from DeepRL.learn import activations
from DeepRL.learn import objectives
from DeepRL.learn import regularizers

from DeepRL.learn import initializations


def indexing(tensor, index, dim):
    index = tf.reshape(tf.one_hot(index, dim), [-1, dim])
    index = tf.cast(index, tf.bool)
    return tf.boolean_mask(tensor, index)


def get_initializer(initializer):
    members = dict(getmembers(initializations))
    return members[initializer]


def get_activation(activation):
    members = dict(getmembers(activations))
    return members[activation]


def get_objective(objective):
    members = dict(getmembers(objectives))
    return members[objective]


def get_regularizer(regularizer):
    members = dict(getmembers(regularizers))
    return members[regularizer]


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def autoformat_kernel_2d(strides):
    if type(strides) is int:
        return [1, strides, strides, 1]
    elif type(strides) in [tuple, list]:
        if len(strides) == 2:
            return [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            return [strides[0], strides[1], strides[2], strides[3]]
        else:
            raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 2 or 4 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


def autoformat_filter_conv2d(fsize, in_depth, out_depth):
    if type(fsize) is int:
        return [fsize, fsize, in_depth, out_depth]
    elif type(fsize) in [tuple, list]:
        if len(fsize) == 2:
            return [fsize[0], fsize[1], in_depth, out_depth]
        else:
            raise Exception("filter length error: " + str(len(fsize))
                            + ", only a length of 2 is supported.")
    else:
        raise Exception("filter format error: " + str(type(fsize)))


def autoformat_padding(padding):
    if padding in ['same', 'SAME', 'valid', 'VALID']:
        return str.upper(padding)
    else:
        raise Exception("Unknow padding! Accepted values: 'same', 'valid'.")