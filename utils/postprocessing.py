import numpy as np
import tensorflow as tf


def detections_to_bboxes(detections):
    """Convert detection boxes to bounding boxes.

    :param detections: YOLOv3's output.
    :return: The converted bounding boxes.
    """
    centre_x, centre_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width/2
    h2 = height/2
    x0 = centre_x - w2
    y0 = centre_y - h2
    x1 = centre_x + w2
    y1 = centre_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections


def get_iou(box1, box2):
    """Get the intersection over union of two boxes.

    :param box1: The first box.
    :param box2: The second box.
    :return: The intersection over union.
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0)*(int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0)*(b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0)*(b2_y1 - b2_y0)

    iou = int_area/(b1_area + b2_area - int_area + 1e-05)
    return iou


def nms(bboxes, conf_thresh, iou_thresh=0.4):
    """Apply non-maximum suppresion on the bounding boxes.

    :param bboxes: The predicted bounding boxes.
    :param conf_thresh: The threshold to decide whether a predicted bounding box is an object.
    :param iou_thresh: The threshold to decide whether two boxes overlap or not.
    :return: A dictionary, where the key is the class, and the value is the bbox-score tuple.
    """
    conf_mask = np.expand_dims((bboxes[:, :, 4] > conf_thresh), -1)
    predictions = bboxes*conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append({'box': box, 'score': score})
                cls_boxes = cls_boxes[1:]
                ious = np.array([get_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_thresh
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result
