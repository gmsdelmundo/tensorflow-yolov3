import tensorflow as tf

from tests.darknet53_test import Darknet53Test
from tests.tf_utils_test import TFUtilsTest

Darknet53Test()
TFUtilsTest()

if __name__ == '__main__':
    tf.test.main()
