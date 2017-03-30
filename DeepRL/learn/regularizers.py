import tensorflow as tf
slim = tf.contrib.slim


def L1(tensor, wd=0.001):
    """ L1.

    Computes the L1 norm of a tensor:

      output = sum(|t|) * wd

    Arguments:
        tensor: `Tensor`. The tensor to apply regularization.
        wd: `float`. The decay.

    Returns:
        The regularization `Tensor`.

    """
    return tf.multiply(tf.reduce_sum(tf.abs(tensor)), wd, name='L1-Loss')


def L2(tensor, wd=0.001):
    """ L2.

    Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2 * wd

    Arguments:
        tensor: `Tensor`. The tensor to apply regularization.
        wd: `float`. The decay.

    Returns:
        The regularization `Tensor`.

    """
    return tf.multiply(tf.nn.l2_loss(tensor), wd, name='L2-Loss')
