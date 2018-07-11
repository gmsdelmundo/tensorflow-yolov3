import tensorflow as tf
from tensorflow.contrib import slim

from tests.base_test import BaseTest
from utils import tf_utils


class TFUtilsTest(BaseTest, tf.test.TestCase):
    """Test class for all TensorFlow helper functions."""

    def test_fixed_padding_NHWC(self):
        """Test fixed_padding for an input tensor with NHWC data format."""
        print("test_fixed_padding_NHWC")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NHWC',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                outputs = tf_utils.fixed_padding(self.nhwc_inputs, 3)

                self.assertAllEqual(outputs.shape, [1, 420, 420, 3])

    def test_fixed_padding_NCHW(self):
        """Test fixed_padding for an input tensor with NCHW data format."""
        print("test_fixed_padding_NCHW")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NCHW',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                outputs = tf_utils.fixed_padding(self.nchw_inputs, 3)

                self.assertAllEqual(outputs.shape, [1, 3, 420, 420])

    def test_conv2d_fixed_padding_NHWC(self):
        """Test conv2d_fixed_padding for an input tensor with NHWC data format."""
        print("test_conv2d_fixed_padding_NHWC")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NHWC',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                outputs = tf_utils.conv2d_fixed_padding(self.nhwc_inputs, 256, 3)

                self.assertAllEqual(outputs.shape, [1, 418, 418, 256])

    def test_conv2d_fixed_padding_NCHW(self):
        """Test conv2d_fixed_padding for an input tensor with NCHW data format."""
        print("test_conv2d_fixed_padding_NCHW")

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NCHW',
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=self.normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                outputs = tf_utils.conv2d_fixed_padding(self.nchw_inputs, 256, 3)

                self.assertAllEqual(outputs.shape, [1, 256, 418, 418])


if __name__ == '__main__':
    tf.test.main()
