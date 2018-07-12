import cv2
import numpy as np
import tensorflow as tf

from models.yolov3 import YOLOv3
from utils import postprocessing
from config.config import config
from utils.load_yolov3_weights import load_yolov3_weights

with tf.variable_scope('model'):
    print("Constructing computational graph...")
    model = YOLOv3(config)
    print("Done")

    print("Loading weights...")
    global_vars = tf.global_variables(scope='model')
    assign_ops = load_yolov3_weights(global_vars, config['WEIGHTS_PATH'])
    print("Done")
    print("=============================================")

print("Loading class names...")
classes = []
f = open(config['CLASS_PATH'], 'r').read().splitlines()
for line in f:
    classes.append(line)
print(classes)
print("Done")
print("=============================================")

print("Running YOLOv3...")
boxes = postprocessing.detections_to_bboxes(model.outputs)

image = cv2.imread('000000532481.jpg')
image_resized = cv2.resize(image, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))

with tf.Session() as sess:
    sess.run(assign_ops)
    detected_boxes = sess.run(boxes, feed_dict={model.inputs: np.expand_dims(image_resized, axis=0)})
print("Done")
print("=============================================")

print("Doing postprocessing...")
filtered_boxes = postprocessing.nms(detected_boxes, conf_thresh=config['CONF_THRESH'], iou_thresh=config['IOU_THRESH'])

print(filtered_boxes)

for k, v in filtered_boxes.items():
    print(classes[k], v)

    colour = tuple([int(z) for z in np.random.uniform(0, 255, size=3)])

    for detection in v:
        x1, y1, x2, y2 = detection['box'].reshape(2, 2).reshape(-1)

        cv2.rectangle(image_resized, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(image_resized, classes[k], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

print("Done")
print("=============================================")

cv2.imshow('Result', image_resized)
cv2.waitKey(0)