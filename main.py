import argparse

import cv2
import numpy as np
import tensorflow as tf

from models.yolov3 import YOLOv3
from utils import postprocessing
from config.config import config
from utils.load_yolov3_weights import load_yolov3_weights

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', help='Input image/video.')
args = parser.parse_args()

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
colours = {}
f = open(config['CLASS_PATH'], 'r').read().splitlines()
for i, line in enumerate(f):
    classes.append(line)
    colours[i] = tuple([int(z) for z in np.random.uniform(0, 255, size=3)])
print(classes)
print("Done")
print("=============================================")

print("Running YOLOv3...")
with tf.Session() as sess:
    sess.run(assign_ops)
    boxes = postprocessing.detections_to_bboxes(model.outputs)

    if args.input[-4:] == '.mp4' or args.input[-4:] == '.avi':
        video = cv2.VideoCapture(args.input)

        while video.isOpened():
            ret, frame = video.read()
            if ret is not True:
                video.release()
                break

            resized_frame = cv2.resize(frame, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))

            detected_boxes = sess.run(boxes, feed_dict={model.inputs: np.expand_dims(resized_frame, axis=0)})
            filtered_boxes = postprocessing.nms(detected_boxes, conf_thresh=config['CONF_THRESH'],
                                                iou_thresh=config['IOU_THRESH'])

            for class_id, v in filtered_boxes.items():
                for detection in v:
                    box = detection['box']

                    original_size = np.array(frame.shape[:2][::-1])
                    resized_size = np.array([config['IMAGE_SIZE'], config['IMAGE_SIZE']])
                    ratio = original_size/resized_size
                    box = box.reshape(2, 2)*ratio
                    box = list(box.reshape(-1))
                    x1, y1, x2, y2 = [int(z) for z in box]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), colours[class_id], 2)
                    cv2.putText(frame, classes[class_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                                cv2.LINE_AA)

            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.release()
                break
        print("Done")
    else:
        image = cv2.imread(args.input)
        resized_image = cv2.resize(image, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))
        detected_boxes = sess.run(boxes, feed_dict={model.inputs: np.expand_dims(resized_image, axis=0)})
        filtered_boxes = postprocessing.nms(detected_boxes, conf_thresh=config['CONF_THRESH'],
                                            iou_thresh=config['IOU_THRESH'])

        for class_id, v in filtered_boxes.items():
            for detection in v:
                box = detection['box']

                original_size = np.array(image.shape[:2][::-1])
                resized_size = np.array([config['IMAGE_SIZE'], config['IMAGE_SIZE']])
                ratio = original_size / resized_size
                box = box.reshape(2, 2) * ratio
                box = list(box.reshape(-1))
                x1, y1, x2, y2 = [int(z) for z in box]

                cv2.rectangle(image, (x1, y1), (x2, y2), colours[class_id], 2)
                cv2.putText(image, classes[class_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)

        print("Done")
        cv2.imshow('Result', image)
        cv2.waitKey(0)
