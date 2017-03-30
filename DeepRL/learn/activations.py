import tensorflow as tf


def linear(x):
    return x


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def softmax(x):
    return tf.nn.softmax(x)


def softplus(x):
    return tf.nn.softplus(x)


def softsign(x):
    return tf.nn.softsign(x)


def relu(x):
    return tf.nn.relu(x)