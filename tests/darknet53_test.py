import tensorflow as tf
from tensorflow.contrib import slim

from models.darknet53 import Darknet53
from tests.base_test import BaseTest
from utils import tf_utils


class Darknet53Test(BaseTest, tf.test.TestCase):
    """Test class for Darknet53."""

    def test_Darknet53_NHWC(self):
        """Test Darknet53 for an input tensor with NHCW data format."""
        print("test_Darknet53_NHWC")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NHWC',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                route1, route2, outputs = Darknet53.build_model(self.nhwc_inputs)

                self.assertAllEqual(route1.shape, [1, 27, 27, 512])
                self.assertAllEqual(route2.shape, [1, 53, 53, 256])
                self.assertAllEqual(outputs.shape, [1, 14, 14, 1024])

    def test_Darknet53_NCHW(self):
        """Test Darknet53 for an input tensor with NCHW data format."""
        print("test_Darknet53_NCHW")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NCHW',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                route1, route2, outputs = Darknet53.build_model(self.nchw_inputs)

                self.assertAllEqual(route1.shape, [1, 512, 27, 27])
                self.assertAllEqual(route2.shape, [1, 256, 53, 53])
                self.assertAllEqual(outputs.shape, [1, 1024, 14, 14])


if __name__ == '__main__':
    tf.test.main()
