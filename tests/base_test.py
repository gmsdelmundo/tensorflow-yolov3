import tensorflow as tf


class BaseTest:
    """Base class for the test classes."""
    nhwc_inputs = tf.random_uniform([1, 418, 418, 3], 0, 255, dtype=tf.float32)
    nchw_inputs = tf.random_uniform([1, 3, 418, 418], 0, 255, dtype=tf.float32)
