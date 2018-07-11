import tensorflow as tf
from tensorflow.contrib import slim

from models.darknet53 import Darknet53
from utils import tf_utils


class YOLOv3:
    """The YOLO framework.

    Reference: https://pjreddie.com/darknet/yolo/
    """

    def __init__(self, config):
        """The YOLOv3 constructor.

        :param config: The configuration object.
        """
        self.BATCH_NORM_DECAY = config.BATCH_NORM_DECAY
        self.BATCH_NORM_EPSILON = config.BATCH_NORM_EPSILON
        self.LEAKY_RELU = config.LEAKY_RELU

        self.ANCHORS = config.ANCHORS

        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.NUM_CLASSES = config.NUM_CLASSES
        self.REUSE = config.REUSE
        self.DATA_FORMAT = config.DATA_FORMAT

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='inputs')
        self.outputs = self.build_model()

    def _block(self, inputs, filters):
        """A convolution block.

        :param inputs: The input tensor.
        :param filters: The number of convolution filters.
        :return: The output tensor.
        """
        outputs = inputs
        route = None

        for i in range(3):
            outputs = tf_utils.conv2d_fixed_padding(outputs, filters, 1)
            if i == 2:
                route = outputs
            outputs = tf_utils.conv2d_fixed_padding(outputs, filters*2, 3)

        return route, outputs

    def _get_grid_size(self, inputs):
        """Get the grid size.

        :param shape: The input tensor.
        :return: The grid size.
        """
        shape = inputs.get_shape().as_list()
        if len(shape) == 4:
            shape = shape[1:]
        return shape[1:3] if self.DATA_FORMAT == 'NCHW' else shape[0:2]

    def _detection_layer(self, inputs, anchors):
        """Make predictions from the input feature map.

        :param inputs: The input tensor.
        :param anchors: The anchors to use.
        :return: The predicted, de-normalised bounding boxes.
        """
        num_anchors = len(anchors)
        predictions = slim.conv2d(inputs, num_anchors*(5 + self.NUM_CLASSES), 1, stride=1, activation_fn=None)

        grid_size = self._get_grid_size(predictions)
        dim = grid_size[0]*grid_size[1]
        bbox_attrs = 5 + self.NUM_CLASSES

        if self.DATA_FORMAT == 'NCHW':
            predictions = tf.reshape(predictions, [-1, num_anchors*bbox_attrs, dim])
            predictions = tf.transpose(predictions, [0, 2, 1])

        predictions = tf.reshape(predictions, [-1, num_anchors*dim, bbox_attrs])

        stride = (self.IMAGE_SIZE//grid_size[0], self.IMAGE_SIZE//grid_size[1])

        anchors = [(a[0]/stride[0], a[1]/stride[1]) for a in anchors]

        box_centres, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, self.NUM_CLASSES], axis=-1)

        box_centres = tf.nn.sigmoid(box_centres)
        confidence = tf.nn.sigmoid(confidence)

        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        a, b = tf.meshgrid(grid_x, grid_y)

        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))

        xy_offset = tf.concat([x_offset, y_offset], axis=-1)
        xy_offset = tf.reshape(tf.tile(xy_offset, [1, num_anchors]), [1, -1, 2])

        box_centres = box_centres + xy_offset
        box_centres = box_centres*stride

        anchors = tf.tile(anchors, [dim, 1])
        box_sizes = tf.exp(box_sizes)*anchors
        box_sizes = box_sizes*stride

        detections = tf.concat([box_centres, box_sizes, confidence], axis=-1)

        classes = tf.nn.sigmoid(classes)
        predictions = tf.concat([detections, classes], axis=-1)
        return predictions

    def _upsample(self, inputs, output_shape):
        """Upsample a tensor.

        :param inputs: The input tensor.
        :param output_shape: The desired shape to upsize to.
        :return: The upsized tensor.
        """
        # Pad by 1 pixel first
        outputs = tf_utils.fixed_padding(inputs, 3, mode='SYMMETRIC')

        # Convert to NHWC if in NCHW format for upsample functions
        if self.DATA_FORMAT == 'NCHW':
            outputs = tf.transpose(outputs, [0, 2, 3, 1])
            (height, width) = (output_shape[3], output_shape[2])
        else:
            (height, width) = (output_shape[2], output_shape[1])

        (new_height, new_width) = (height + 4, width + 4)
        outputs = tf.image.resize_bilinear(outputs, (new_height, new_width))
        outputs = outputs[:, 2:-2, 2:-2, :]

        # Convert back to NCHW format
        if self.DATA_FORMAT == 'NCHW':
            outputs = tf.transpose(outputs, [0, 3, 1, 2])
        return outputs

    def build_model(self):
        """Build the YOLOv3 model."""
        # Transpose from the default channel-last (NHWC for CPU) to channel-first (NCHW for GPU)
        outputs = tf.transpose(self.inputs, [0, 3, 1, 2]) if self.DATA_FORMAT == 'NCHW' else self.inputs
        outputs = outputs/255
        normalizer_params = {'decay': self.BATCH_NORM_DECAY,
                             'epsilon': self.BATCH_NORM_EPSILON,
                             'scale': True,
                             'is_training': False,
                             'fused': None}

        with slim.arg_scope([slim.batch_norm, slim.conv2d, tf_utils.fixed_padding], data_format=self.DATA_FORMAT,
                            reuse=self.REUSE):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=normalizer_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                with tf.variable_scope('darknet53'):
                    route1, route2, outputs = Darknet53.build_model(outputs)

                with tf.variable_scope('yolov3'):
                    # First detection block
                    route, outputs = self._block(outputs, 512)

                    detect1 = self._detection_layer(outputs, self.ANCHORS[6:9])
                    detect1 = tf.identity(detect1, name='detect1')

                    # Second detection block
                    outputs = tf_utils.conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route2.get_shape().as_list()
                    outputs = self._upsample(outputs, upsample_size)
                    outputs = tf.concat([outputs, route2], axis=(1 if self.DATA_FORMAT == 'NCHW' else 3))

                    route, outputs = self._block(outputs, 256)

                    detect2 = self._detection_layer(outputs, self.ANCHORS[3:6])
                    detect2 = tf.identity(detect2, name='detect2')

                    # Third detection block
                    outputs = tf_utils.conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route1.get_shape().as_list()
                    outputs = self._upsample(outputs, upsample_size)
                    outputs = tf.concat([outputs, route1], axis=(1 if self.DATA_FORMAT == 'NCHW' else 3))

                    _, outputs = self._block(outputs, 128)

                    detect3 = self._detection_layer(outputs, self.ANCHORS[:3])
                    detect3 = tf.identity(detect3, name='detect3')

                    detections = tf.concat([detect1, detect2, detect3], axis=1, name='outputs')
                    return detections
