import torch
from tools.utils import skewiou


def post_process_1(prediction, conf_thres=0.5, nms_thres=0.4):
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

        image_pred = image_pred[image_pred[:, 1] >= 0]


        for i in range(len(image_pred)):
            if image_pred[i][1] < 0:
                print(image_pred[i])

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 5] * image_pred[:, 6:].max(1)[0]
        # score = image_pred[:, 6:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 6:].max(1, keepdim=True)  # class_preds-> index of classes
        detections = torch.cat((image_pred[:, :6], class_confs.float(), class_preds.float()), 1)


        # non-maximum suppression
        keep_boxes = []
        labels = detections[:, -1].unique()
        for label in labels:
            detect = detections[detections[:, -1] == label]
            while len(detect):
                large_overlap = skewiou(detect[0, :5], detect[:, :5]) > nms_thres
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                weights = detect[large_overlap, 5:6]
                # Merge overlapping bboxes by order of confidence
                detect[0, :4] = (weights * detect[large_overlap, :4]).sum(0) / weights.sum()
                keep_boxes += [detect[0].detach()]
                detect = detect[~large_overlap]
            if keep_boxes:
                output[batch] = torch.stack(keep_boxes)

    return output



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

    for j in range(len(prediction)):
        pred = prediction[j]

        xc = pred[..., 5] > conf_thres  # candidates

        x = pred[xc]

        x[:, 6:] *= x[:, 5:6]  # conf = obj_conf * cls_conf
        box = x[x[:, 5].argsort(descending=True)]

        if len(box) == 0:
            continue
        #
        class_confs, class_preds = box[:, 6:].max(1, keepdim=True)
        box = torch.cat((box[:, :6], class_confs.float(), class_preds.float()), 1)
        #
        keep_boxes = torch.ones(box.shape[0], dtype=torch.bool)
        for i in range(box.shape[0] - 1):
            if keep_boxes[i]:
                overlap = skewiou(box[i,:5], box[(i + 1):,:5])

                keep_overlap = torch.logical_or(overlap < nms_thres, torch.tensor(0))
                keep_boxes[(i + 1):] = torch.logical_and(keep_overlap, keep_boxes[(i + 1):])
        idxes = torch.where(keep_boxes)
        bbox = box[idxes]
        xc = bbox[...,-2] > 0

        output[j] = bbox[xc]
    return output # 最后一个 是类别  末二个是 分数



