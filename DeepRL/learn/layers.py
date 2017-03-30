import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from DeepRL.learn import collections as cl
from DeepRL.learn import config as cfg
from DeepRL.learn import learn_utils as utils

from DeepRL.learn import variables as vs


def conv2d(incoming, nb_filter, filter_size, activation='linear',
           weights_init="uniform_scaling", bias_init="zeros",
           regularizer=None, weight_decay=0.001, strides=1,
           padding="same", name="Conv2D"):
    input_shape = utils.get_incoming_shape(incoming)
    filter_size = utils.autoformat_filter_conv2d(filter_size, input_shape[-1], nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        W_regul = None
        if regularizer:
            W_regul = lambda x: utils.get_regularizer(regularizer)(x, weight_decay)
        W = vs.variable(
            shape=filter_size, initializer=weights_init,
            name=scope+"weights", regularizer=W_regul
        )
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

        b = vs.variable(shape=[nb_filter], initializer=bias_init, name=scope+"biases")
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)

        inference = tf.nn.conv2d(incoming, W, strides, padding)
        inference = tf.nn.bias_add(inference, b)
        inference = utils.get_activation(activation)(inference)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    return inference


def max_pool_2d(incoming, kernel_size, strides=None, padding="same",
                name="MaxPool2D"):
    assert padding in ['same', 'valid', 'SAME', 'VALID'], "Padding must be same' or 'valid'"
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool(incoming, kernel, strides, padding)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    return inference


def fully_connected(incoming, n_units, activation='linear', name="FullyConnected",
                    weights_init="uniform_scaling", bias_init="zeros",
                    regularizer=None, weight_decay=0.001):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    with tf.name_scope(name) as scope:
        W_regul = None
        if regularizer:
            W_regul = lambda x: utils.get_regularizer(regularizer)(x, weight_decay)
        W = vs.variable(
            shape=[n_inputs, n_units], initializer=weights_init,
            name=scope+"weights", regularizer=W_regul
        )
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, W)

        b = vs.variable(shape=[n_units], initializer=bias_init, name=scope+"biases")
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)

        inference = incoming
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])
        inference = tf.matmul(inference, W)
        inference = tf.nn.bias_add(inference, b)
        inference = utils.get_activation(activation)(inference)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    return inference


def batch_normalization(incoming, beta=0.0, gamma=1.0, epsilon=1e-5,
                        decay=0.9, stddev=0.002, name="BatchNormalization"):

    input_shape = utils.get_incoming_shape(incoming)
    input_ndim = len(input_shape)

    gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)

    with tf.name_scope(name) as scope:
        beta = vs.variable(shape=[input_shape[-1]], initializer=tf.constant_initializer(beta), name=scope+"beta")
        gamma = vs.variable(shape=[input_shape[-1]], initializer=gamma_init, name=scope+"gamma")
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, beta)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, gamma)

        axis = list(range(input_ndim - 1))

        moving_mean = vs.variable(input_shape[-1:], initializer=tf.zeros_initializer(), name=scope+"moving_mean")
        moving_variance = vs.variable(input_shape[-1:],
                                      initializer=tf.constant_initializer(1.),
                                      name=scope+"moving_variance")

        # Define a function to update mean and variance
        def update_mean_var():
            mean, variance = tf.nn.moments(incoming, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False)

            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        # Retrieve variable managing training mode
        is_training = cfg.get_training_mode()
        mean, var = tf.cond(is_training, update_mean_var, lambda: (moving_mean, moving_variance))

        inference = tf.nn.batch_normalization(incoming, mean, var, beta, gamma, epsilon)
        inference.set_shape(input_shape)


        # Add attributes for easy access
        inference.scope = scope
        inference.beta = beta
        inference.gamma = gamma

        # Track output tensor.
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, inference)

    return inference


def dropout(incoming, keep_prob, name="Dropout"):
    with tf.name_scope(name) as scope:

        inference = incoming

        def apply_dropout():
            if type(inference) in [list, np.array]:
                for x in inference:
                    x = tf.nn.dropout(x, keep_prob)
                return inference
            else:
                return tf.nn.dropout(inference, keep_prob)

        is_training = cfg.get_training_mode()
        inference = tf.cond(is_training, apply_dropout, lambda: inference)

    return inference


def reshape(incoming, new_shape, name="Reshape"):
    with tf.name_scope(name) as scope:
        inference = incoming
        if isinstance(inference, list):
            inference = tf.concat(inference, 0)
            inference = tf.cast(inference, tf.float32)
        inference = tf.reshape(inference, shape=new_shape)

    tf.add_to_collection(cl.LAYER_TENSOR + '/' + name, inference)
    return inference


def flatten(incoming, name="Flatten"):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    dims = int(np.prod(input_shape[1:]))
    x = reshape(incoming, [-1, dims], name)

    tf.add_to_collection(cl.LAYER_TENSOR + '/' + name, x)
    return x


def merge(tensors_list, mode, axis=1, name="Merge"):
    """ Merge.

    Merge a list of `Tensor` into a single one.

    Input:
        List of Tensors.

    Output:
        Merged Tensors.

    Arguments:
        tensors_list: A list of `Tensor`, A list of tensors to merge.
        mode: `str`. Merging mode, it supports:
            ```
            'concat': concatenate outputs along specified axis
            'elemwise_sum': outputs element-wise sum
            'elemwise_mul': outputs element-wise sum
            'sum': outputs element-wise sum along specified axis
            'mean': outputs element-wise average along specified axis
            'prod': outputs element-wise multiplication along specified axis
            'max': outputs max elements along specified axis
            'min': outputs min elements along specified axis
            'and': `logical and` btw outputs elements along specified axis
            'or': `logical or` btw outputs elements along specified axis
            ```
        axis: `int`. Represents the axis to use for merging mode.
            In most cases: 0 for concat and 1 for other modes.
        name: A name for this layer (optional). Default: 'Merge'.

    """

    assert len(tensors_list) > 1, "Merge required 2 or more tensors."

    with tf.name_scope(name) as scope:
        tensors = [l for l in tensors_list]
        if mode == 'concat':
            inference = tf.concat(axis, tensors)
        elif mode == 'elemwise_sum':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.add(inference, tensors[i])
        elif mode == 'elemwise_mul':
            inference = tensors[0]
            for i in range(1, len(tensors)):
                inference = tf.mul(inference, tensors[i])
        elif mode == 'sum':
            inference = tf.reduce_sum(tf.concat(axis, tensors),
                                      reduction_indices=axis)
        elif mode == 'mean':
            inference = tf.reduce_mean(tf.concat(axis, tensors),
                                       reduction_indices=axis)
        elif mode == 'prod':
            inference = tf.reduce_prod(tf.concat(axis, tensors),
                                       reduction_indices=axis)
        elif mode == 'max':
            inference = tf.reduce_max(tf.concat(axis, tensors),
                                      reduction_indices=axis)
        elif mode == 'min':
            inference = tf.reduce_min(tf.concat(axis, tensors),
                                      reduction_indices=axis)
        elif mode == 'and':
            inference = tf.reduce_all(tf.concat(axis, tensors),
                                      reduction_indices=axis)
        elif mode == 'or':
            inference = tf.reduce_any(tf.concat(axis, tensors),
                                      reduction_indices=axis)
        else:
            raise Exception("Unknown merge mode", str(mode))

    return inference
