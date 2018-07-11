import tensorflow as tf

from utils.config import Config


class BaseTest:
    normalizer_params = {'decay': Config.BATCH_NORM_DECAY,
                         'epsilon': Config.BATCH_NORM_EPSILON,
                         'scale': True,
                         'is_training': True,
                         'fused': None}

    nhwc_inputs = tf.random_uniform([1, 418, 418, 3], 0, 255, dtype=tf.float32)
    nchw_inputs = tf.random_uniform([1, 3, 418, 418], 0, 255, dtype=tf.float32)

    LEAKY_RELU = Config.LEAKY_RELU
    REUSE = Config.REUSE
