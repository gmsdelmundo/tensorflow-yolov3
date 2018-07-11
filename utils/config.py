import os


config = {
    'BATCH_NORM_DECAY': 0.9,
    'BATCH_NORM_EPSILON': 1e-05,
    'LEAKY_RELU': 0.1,

    'ANCHORS': [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
    'IMAGE_SIZE': 416,
    'NUM_CLASSES': 80,

    'REUSE': False,
    'DATA_FORMAT': 'NCHW',  # NCHW for GPU, NHWC for CPU

    'WEIGHTS_PATH': os.path.join('weights', 'yolov3.weights')
}
