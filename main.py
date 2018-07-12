import argparse

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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


def resize_bbox_to_original(original_image, bbox):
    """Resize a detected bounding box to fit the original image.

    :param original_image: The original image.
    :param bbox: The bounding box.
    :return: The resized bounding box.
    """
    original_size = np.array(original_image.shape[:2][::-1])
    resized_size = np.array([config['IMAGE_SIZE'], config['IMAGE_SIZE']])
    ratio = original_size/resized_size
    bbox = bbox.reshape(2, 2)*ratio
    bbox = list(bbox.reshape(-1))
    bbox = [int(z) for z in bbox]
    return bbox


def label_bboxes(original_image, bbox, class_id, score):
    """Draw a bounding box on the original image with a label.

    :param original_image: The original iamge.
    :param bbox: The bounding box.
    :param class_id: The class ID of the bounding box.
    :param score: The objectness score.
    :return: The labeled image.
    """
    x1, y1, x2, y2 = resize_bbox_to_original(original_image, bbox)
    label = '{}: {}%'.format(classes[class_id], int(score*100))

    cv2.rectangle(frame, (x1, y1), (x2, y2), colours[class_id], 2)

    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    w, h = text_size
    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 - h), colours[class_id], cv2.FILLED)

    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return original_image


with tf.Session() as sess:
    sess.run(assign_ops)
    boxes = postprocessing.detections_to_bboxes(model.outputs)

    if args.input[-4:] == '.mp4' or args.input[-4:] == '.avi':
        video = cv2.VideoCapture(args.input)

        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(unit='frame', total=video_length)

        while video.isOpened():
            ret, frame = video.read()
            if ret is not True:
                video.release()
                break

            resized_frame = cv2.resize(frame, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))
            detected_bboxes = sess.run(boxes, feed_dict={model.inputs: np.expand_dims(resized_frame, axis=0)})
            filtered_bboxes = postprocessing.nms(detected_bboxes, conf_thresh=config['CONF_THRESH'],
                                                 iou_thresh=config['IOU_THRESH'])

            for class_id, v in filtered_bboxes.items():
                for detection in v:
                    label_bboxes(frame, detection['bbox'], class_id, detection['score'])

            pbar.update(1)
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video.release()
                break
        print("Done")
    else:
        frame = cv2.imread(args.input)
        resized_frame = cv2.resize(frame, (config['IMAGE_SIZE'], config['IMAGE_SIZE']))
        detected_bboxes = sess.run(boxes, feed_dict={model.inputs: np.expand_dims(resized_frame, axis=0)})
        filtered_bboxes = postprocessing.nms(detected_bboxes, conf_thresh=config['CONF_THRESH'],
                                             iou_thresh=config['IOU_THRESH'])

        for class_id, v in filtered_bboxes.items():
            for detection in v:
                label_bboxes(frame, detection['bbox'], class_id, detection['score'])

        print("Done")
        cv2.imshow('Result', frame)
        cv2.waitKey(0)
