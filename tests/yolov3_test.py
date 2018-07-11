import numpy as np
import tensorflow as tf

from models.yolov3 import YOLOv3
from tests.base_test import BaseTest
from utils.config import config


class YOLOv3Test(BaseTest, tf.test.TestCase):
    """Test class for YOLOv3."""

    def test_yolov3_num_params(self):
        """Test to see if the number of parameters in YOLOv3 is 62001757."""
        with tf.variable_scope('model'):
            _ = YOLOv3(config)
        global_vars = tf.global_variables(scope='model')

        num_params = 0
        for var in global_vars:
            shape = var.get_shape().as_list()
            num_params += np.prod(shape)

        self.assertAllEqual(num_params, 62001757)
