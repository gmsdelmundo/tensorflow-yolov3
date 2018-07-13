import tensorflow as tf
from tensorflow.contrib import slim

from tests import BaseTest
from utils import tf_utils


class TestTFUtils(BaseTest, tf.test.TestCase):
    """Test class for all TensorFlow helper functions."""

    def test_fixed_padding_NHWC(self):
        """Test fixed_padding for an input tensor with NHWC data format."""
        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NHWC'):
            outputs = tf_utils.fixed_padding(self.nhwc_inputs, 3)

            self.assertAllEqual(outputs.shape, [1, 420, 420, 3])

    def test_fixed_padding_NCHW(self):
        """Test fixed_padding for an input tensor with NCHW data format."""
        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NCHW'):
            outputs = tf_utils.fixed_padding(self.nchw_inputs, 3)

            self.assertAllEqual(outputs.shape, [1, 3, 420, 420])

    def test_conv2d_fixed_padding_NHWC(self):
        """Test conv2d_fixed_padding for an input tensor with NHWC data format."""
        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NHWC'):
            outputs = tf_utils.conv2d_fixed_padding(self.nhwc_inputs, 256, 3)

            self.assertAllEqual(outputs.shape, [1, 418, 418, 256])

    def test_conv2d_fixed_padding_NCHW(self):
        """Test conv2d_fixed_padding for an input tensor with NCHW data format."""
        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format='NCHW'):
            outputs = tf_utils.conv2d_fixed_padding(self.nchw_inputs, 256, 3)

            self.assertAllEqual(outputs.shape, [1, 256, 418, 418])
