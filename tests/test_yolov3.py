import copy

import numpy as np
import tensorflow as tf

from models import YOLOv3
from tests import BaseTest
from config import config


class TestYOLOv3(BaseTest, tf.test.TestCase):
    """Test class for YOLOv3."""

    def test_yolov3_num_params_NHWC(self):
        """Test to see if the number of parameters in YOLOv3 is 62001757 with tensors in NCHW format."""
        c = copy.deepcopy(config)
        c['DATA_FORMAT'] = 'NHWC'
        with tf.variable_scope('model'):
            _ = YOLOv3(c)
        global_vars = tf.global_variables(scope='model')

        num_params = 0
        for var in global_vars:
            shape = var.get_shape().as_list()
            num_params += np.prod(shape)

        self.assertAllEqual(num_params, 62001757)

    def test_yolov3_num_params_NCHW(self):
        """Test to see if the number of parameters in YOLOv3 is 62001757 with tensors in NCHW format."""
        c = copy.deepcopy(config)
        c['DATA_FORMAT'] = 'NCHW'
        with tf.variable_scope('model'):
            _ = YOLOv3(c)
        global_vars = tf.global_variables(scope='model')

        num_params = 0
        for var in global_vars:
            shape = var.get_shape().as_list()
            num_params += np.prod(shape)

        self.assertAllEqual(num_params, 62001757)
