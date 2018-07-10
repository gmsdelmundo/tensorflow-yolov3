import tensorflow as tf
from tensorflow.contrib import slim


@tf.contrib.framework.add_arg_scope
def fixed_padding(inputs, kernel_size):
    """Pad the input along the spatial dimensions, independent of input size.
    Reference: https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

    :param inputs: A tensor of size [batch, channels, height, width] or [batch, height, width, channels].
    :param kernel_size: the convolution kernel size.
    :return: A tensor of size [batch, height + kernel_size - 1, width + kernel_size - 1, channels].
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total//2
    pad_end = pad_total - pad_beg

    padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

    padded_inputs = tf.pad(inputs, padding)
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    """Perform a strided 2D convolution with explicit padding.

    :param inputs: A tensor of size [batch, height, width, channels].
    :param filters: The number of convolution filters.
    :param kernel_size: The convolution kernel size.
    :param strides: The convolution kernel stride.
    :return: A tensor of size [batch, new_height, new_width, filters].
    """
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    outputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return outputs
