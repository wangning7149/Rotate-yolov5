# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
import math
import os
import numpy as np
import colorsys
import cv2

from shapely.geometry import Polygon
def get_classes(class_file_name='./coco_classes.names'):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def xywha2xyxyxyxy(p):
    """
    Args:
        p: 1-d tensor which contains (x, y, w, h, a)
    Returns: bbox coordinates (x1, y1, x2, y2, x3, y3, x4, y4) which is transferred from (x, y, w, h, a)
    """
    x, y, w, h, a = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4]

    x1, y1, x2, y2 = x + w / 2, y - h / 2, x + w / 2, y + h / 2
    x3, y3, x4, y4 = x - w / 2, y + h / 2, x - w / 2, y - h / 2

    P1 = np.array((x1, y1, 1)).reshape(3, -1)
    P2 = np.array((x2, y2, 1)).reshape(3, -1)
    P3 = np.array((x3, y3, 1)).reshape(3, -1)
    P4 = np.array((x4, y4, 1)).reshape(3, -1)
    P = np.stack((P1, P2, P3, P4)).squeeze(2).T
    P = rotate(x, y, a, P)
    X1, X2, X3, X4 = P[0]
    Y1, Y2, Y3, Y4 = P[1]

    return X1, Y1, X2, Y2, X3, Y3, X4, Y4
def R(theta):
    """
    Args:
        theta: must be radian
    Returns: rotation matrix
    """
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return r


def T(x, y):
    """
    Args:
        x, y: values to translate
    Returns: translation matrix
    """
    t = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
    return t

def rotate(center_x, center_y, a, p):
    P = np.dot(T(center_x, center_y), np.dot(R(a), np.dot(T(-center_x, -center_y), p)))
    return P[:2]

def skewiou(box1, box2):
    assert len(box1) == 5 and len(box2[0]) == 5
    iou = []
    g = np.stack(xywha2xyxyxyxy(box1))
    g = Polygon(g.reshape((4, 2)))
    for i in range(len(box2)):
        p = np.stack(xywha2xyxyxyxy(box2[i]))
        p = Polygon(p.reshape((4, 2)))
        if not g.is_valid or not p.is_valid:
            print("something went wrong in skew iou")
            return 0
        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        iou.append(inter / (union + 1e-16))
    return np.stack(iou)

def xywh2xyxy(x):
    y = np.zeros(x.shape,dtype=np.float)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    scale = min(current_dim * 1.0 / orig_w, current_dim * 1.0 / orig_h)
    new_h, new_w = int(scale * orig_h), int(scale * orig_w)

    if new_h < new_w:
        pad1 = (current_dim - new_h) // 2
        boxes[:, 1] -= pad1
    elif new_w < new_h:
        pad1 = (current_dim - new_w) // 2
        boxes[:, 0] -= pad1
    boxes[:, :4] /= scale

    return boxes

def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)
def plot_boxes(img, boxes, class_names, img_size, color=None):
    import cv2 as cv


    boxes = rescale_boxes(boxes, img_size[0], img.shape[:2])
    boxes = np.array(boxes)

    for i in range(len(boxes)):

        box = boxes[i]
        x, y, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(np.array([x, y, w, h, theta]))
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

        bbox = np.int0([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)])
        cv.drawContours(img, [bbox], 0, (0, 255, 0), 2)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)

        cls_id = np.squeeze(int(box[7]))
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        red = get_color(2, offset, classes)
        green = get_color(1, offset, classes)
        blue = get_color(0, offset, classes)
        if color is None:
            rgb = (red, green, blue)

        # img = cv.putText(img, class_names[cls_id] + ":" + str(round(box[5] * box[6], 2)),
        #                  (X1, Y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 1)


    cv.imwrite('demo.jpg', img)


def sigmoid(x):
    ss = x.shape
    if len(ss) == 4:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for m in range(x.shape[3]):

                        if x[i][j][k][m] >= 0:
                            x[i][j][k][m] = 1.0 / (1 + np.exp(-x[i][j][k][m]))
                        else:
                            x[i][j][k][m] = np.exp(x[i][j][k][m]) / (1 + np.exp(x[i][j][k][m]))
    else:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for m in range(x.shape[3]):
                        for n in range(x.shape[4]):
                            if x[i][j][k][m][n] >= 0:
                                x[i][j][k][m][n] = 1.0 / (1 + np.exp(-x[i][j][k][m][n]))
                            else:
                                x[i][j][k][m][n]=np.exp(x[i][j][k][m][n]) / (1 + np.exp(x[i][j][k][m][n]))

    return x




def post_process(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Args:
        prediction: size-> [batch, ((grid x grid) + (grid x grid) + (grid x grid)) x num_anchors, 8]
                    ex: [1, ((52 x 52) + (26 x 26) + (13 x 13)) x 18, 8] in my case
                    last dimension-> [x, y, w, h, a, conf, num_classes]
    Returns:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    for batch, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 5] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs = image_pred[:, 6:].max(1, keepdims=True)  # class_preds-> index of classes
        class_preds = image_pred[:, 6:].argmax(1)
        class_preds = class_preds.reshape([class_confs.shape[0],1])
        detections = np.concatenate((image_pred[:, :6], class_confs, class_preds), 1)

        # non-maximum suppression
        keep_boxes = []
        labels = np.unique(detections[:, -1])
        for label in labels:
            detect = detections[detections[:, -1] == label]
            while len(detect):
                large_overlap = skewiou(detect[0, :5], detect[:, :5]) > nms_thres
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                weights = detect[large_overlap, 5:6]
                # Merge overlapping bboxes by order of confidence
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                keep_boxes += [detect[0]]
                detect = detect[~large_overlap]
            if keep_boxes:
                output[batch] = np.stack(keep_boxes)

    return output


def draw_bboxs(image, bboxes, gt_classes_index=None, classes=None):
    """draw the bboxes in the original image
    """
    if classes is None:
        classes = get_classes()
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = classes[class_index]
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(
            image,
            bbox_mess, (c1[0], c1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, (0, 0, 0),
            bbox_thick // 2,
            lineType=cv2.LINE_AA)
        print("{} is in the picture with confidence:{:.4f}".format(
            classes_name, score))
        cv2.imwrite("demo.jpg", image)
    return image


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    calculate the nms for bboxes
    """

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])


            weight = np.ones((len(iou), ), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou**2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight

            score_mask = cls_bboxes[:, 4] > 0.

            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def bboxes_iou(boxes1, boxes2):
    """calculate iou for a list of bboxes"""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                  (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                  (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def postprocess_boxes(pred_bbox,
                      org_img_shape,
                      input_shape,
                      score_threshold=0.5):
    """post process boxes"""
    valid_scale = [0, np.inf]
    org_h, org_w = org_img_shape
    input_h, input_w = input_shape
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, :4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([
        pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
        pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5
    ],
                               axis=-1)

    # (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    resize_ratio = min(input_h / org_h, input_w / org_w)
    dw = (input_w - resize_ratio * org_w) / 2
    dh = (input_h - resize_ratio * org_h) / 2
    pred_coor[:, 0] = 1.0 * (pred_coor[:, 0] - dw)
    pred_coor[:, 1] = 1.0 * (pred_coor[:, 1] - dh)
    pred_coor[:,0:4] /= resize_ratio
    # clip the range of bbox
    pred_coor = np.concatenate([
        np.maximum(pred_coor[:, :2], [0, 0]),
        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])
    ],
                               axis=-1)
    # drop illegal boxes whose max < min
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # discard invalid boxes
    bboxes_scale = np.sqrt(
        np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                (bboxes_scale < valid_scale[1]))

    # discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    # scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    scores = pred_conf

    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def yolo_decoder(conv_output, num_anchors, num_classes, anchors, stride):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    batch_size = conv_output.shape[0]
    output_size = conv_output.shape[-1]
    conv_output = np.transpose(conv_output, (0, 2, 3, 1))
    conv_output = np.reshape(
        conv_output,
        (batch_size, output_size, output_size, num_anchors, 5 + num_classes))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = np.tile(
        np.arange(output_size, dtype=np.int32)[:, np.newaxis],
        [1, output_size])
    x = np.tile(
        np.arange(output_size, dtype=np.int32)[np.newaxis, :],
        [output_size, 1])
    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                             axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, :, :, np.newaxis, :],
                      [batch_size, 1, 1, num_anchors, 1])
    xy_grid = xy_grid.astype(np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (np.exp(conv_raw_dwdh) * anchors) * stride
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
    return decode_output


def yolov5_decoder(conv_output, num_anchors, num_classes, anchors, stride):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Five dimension output: [batch_size, num_anchors, output_size, output_size, 5 + num_classes]
    batch_size = conv_output.shape[0]
    output_size = conv_output.shape[-2]
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = np.tile(
        np.arange(output_size, dtype=np.int32)[:, np.newaxis],
        [1, output_size])
    x = np.tile(
        np.arange(output_size, dtype=np.int32)[np.newaxis, :],
        [output_size, 1])
    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                             axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, np.newaxis, :, :, :],
                      [batch_size, num_anchors, 1, 1, 1])
    xy_grid = xy_grid.astype(np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) * 2.0 - 0.5 + xy_grid) * stride
    pred_wh = (sigmoid(conv_raw_dwdh) *
               2.0)**2 * anchors[np.newaxis, :, np.newaxis, np.newaxis, :]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
    return decode_output
def yolov6_decoder(conv_output, num_anchors, num_classes, anchors, stride):
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Five dimension output: [batch_size, num_anchors, output_size, output_size, 5 + num_classes]
    batch_size = conv_output.shape[0]
    output_size = conv_output.shape[-2]
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:6]

    y = np.tile(
        np.arange(output_size, dtype=np.int32)[:, np.newaxis],
        [1, output_size])
    x = np.tile(
        np.arange(output_size, dtype=np.int32)[np.newaxis, :],
        [output_size, 1])
    xy_grid = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis]],
                             axis=-1)
    xy_grid = np.tile(xy_grid[np.newaxis, np.newaxis, :, :, :],
                      [batch_size, num_anchors, 1, 1, 1])
    xy_grid = xy_grid.astype(np.float32)

    pred_xy = (sigmoid(conv_raw_dxdy) * 2.0 - 0.5 + xy_grid) * stride
    pred_wh = (sigmoid(conv_raw_dwdh) *
               2.0)**2 * anchors[np.newaxis, :, np.newaxis, np.newaxis, :]
    pred_xywh = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_conf = sigmoid(conv_raw_conf)
    pred_prob = sigmoid(conv_raw_prob)

    decode_output = np.concatenate([pred_xywh, pred_conf, pred_prob], axis=-1)
    return decode_output


if __name__ == '__main__':
    a = sigmoid(-2121)