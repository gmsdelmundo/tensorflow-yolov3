import tensorflow as tf
from tensorflow.contrib import slim

from tests.darknet53_test import Darknet53Test
from tests.tf_utils_test import TFUtilsTest
from utils import tf_utils
from utils.config import Config

normalizer_params = {'decay': Config.BATCH_NORM_DECAY,
                     'epsilon': Config.BATCH_NORM_EPSILON,
                     'scale': True,
                     'is_training': True,
                     'fused': None}
LEAKY_RELU = Config.LEAKY_RELU
REUSE = Config.REUSE

with slim.arg_scope([slim.conv2d],
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=normalizer_params,
                    biases_initializer=None,
                    activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=LEAKY_RELU)):
    with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], reuse=REUSE):
        Darknet53Test()
        TFUtilsTest()

if __name__ == '__main__':
    tf.test.main()
