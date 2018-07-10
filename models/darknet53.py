from utils import tf_utils


class Darknet53:
    """The Darknet-53 framework.

    Reference: https://pjreddie.com/darknet/
    """
    @staticmethod
    def _block(inputs, filters):
        """A convolution block.

        :param inputs: The input tensor.
        :param filters: The number of convolution filters.
        :return: The output tensor.
        """
        shortcut = inputs

        outputs = tf_utils.conv2d_fixed_padding(inputs, filters, 1)
        outputs = tf_utils.conv2d_fixed_padding(outputs, filters*2, 3)

        outputs = shortcut + outputs
        return outputs

    @staticmethod
    def build_model(inputs):
        """Build the Darknet-53 model.

        :param inputs: The input tensor.
        :return: The two intermediate route tensors, and the output tensor.
        """
        outputs = tf_utils.conv2d_fixed_padding(inputs, 32, 3)
        outputs = tf_utils.conv2d_fixed_padding(outputs, 64, 3, strides=2)
        outputs = Darknet53._block(outputs, 32)
        outputs = tf_utils.conv2d_fixed_padding(outputs, 128, 3, strides=2)

        for _ in range(2):
            outputs = Darknet53._block(outputs, 64)

        outputs = tf_utils.conv2d_fixed_padding(outputs, 256, 3, strides=2)

        for _ in range(8):
            outputs = Darknet53._block(outputs, 128)

        route2 = outputs
        outputs = tf_utils.conv2d_fixed_padding(outputs, 512, 3, strides=2)

        for _ in range(8):
            outputs = Darknet53._block(outputs, 256)

        route1 = outputs
        outputs = tf_utils.conv2d_fixed_padding(outputs, 1024, 3, strides=2)

        for _ in range(4):
            outputs = Darknet53._block(outputs, 512)

        return route1, route2, outputs
