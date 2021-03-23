import torch
import argparse
import numpy as np
from tools.plot import load_class_names
from tools.post_process import post_process
from tools.utils import get_batch_statistics, ap_per_class
from tools.load import LoadDataset
from models.yolo import Model
import cv2 as cv
import os
def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    scale = min(current_dim * 1.0 / orig_w, current_dim * 1.0 / orig_h)
    new_h, new_w = int(scale * orig_h), int(scale * orig_w)

    if new_h < new_w:
        pad1 = (current_dim - new_h) // 2
        boxes[:,1] -= pad1
    elif new_w < new_h:
        pad1 = (current_dim - new_w) // 2
        boxes[:,0] -= pad1
    boxes[:,:4] /= scale

    return boxes

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

import random
def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  ./checkpoint/model_2/yolov5_510.pth   ../rotation_v4/data/test1  ../UVA/test
    parser.add_argument("--test_folder", type=str, default="../SSDD+_val", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="./weights1/yolov5_170.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/ssdd+.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.75, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold for evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=672, help="size of each image dimension")
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    args = parser.parse_args()
    print(args)
    init()
    with open(args.class_path,'r') as f:
        lines = f.readlines()
    nc = 0
    all_files = os.listdir('./')
    for ll in all_files:
        if 'Task1_' in ll:
            os.system('rm ' + ll)

    label_dict = {}
    for l in lines:
        l = l.strip('\n')
        if l:
            label_dict[nc] = l
            nc += 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = Model(args.cfg,nc=nc)
    model = model.to(device)
    model.load_state_dict(pretrained_dict)

    model.eval()

    # Get dataloader
    train_dataset, train_dataloader = LoadDataset(args.test_folder, args.img_size, args.batch_size,
                                                 shuffle=False, augment=False, multiscale=False)
    print("Compute mAP...")
    labels = []

    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets,shapes) in enumerate(train_dataloader):
        print(_,batch_i)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:6] *= args.img_size

        imgs = torch.autograd.Variable(imgs.type(FloatTensor), requires_grad=False)

        with torch.no_grad():
            outputs, loss,loss1,loss2,loss3 = model(imgs)
            outputs = torch.cat(outputs, 1)
            outputs = post_process(outputs.cpu(), conf_thres=args.conf_thres, nms_thres=args.nms_thres)

            img = np.array(cv.imread(_[0]))
            if outputs[0] is None:
                continue
            boxes = rescale_boxes(outputs[0], args.img_size, img.shape[:2])
            boxes = np.array(boxes)

            for i in range(len(boxes)):
                box = boxes[i]

                x, y, w, h, theta = box[0], box[1], box[2], box[3], box[4]

                X1, Y1, X2, Y2, X3, Y3, X4, Y4 = xywha2xyxyxyxy(np.array([x, y, w, h, theta]))
                X1, Y1, X2, Y2, X3, Y3, X4, Y4 = int(X1), int(Y1), int(X2), int(Y2), int(X3), int(Y3), int(X4), int(Y4)

                cls_id = np.squeeze(int(box[7]))
                with open('./Task1_' + label_dict[int(cls_id)] + '.txt','a',encoding='utf-8') as f:
                    f.write(_[0].split('/')[-1].split('.')[0] + ' ' + str(box[6]) + ' ' + str(X1)+ ' ' + str(Y1)+ ' ' + str(X2)+ ' ' + str(Y2)+ ' ' + str(X3)
                            + ' ' + str(Y3)+ ' ' + str(X4) + ' ' + str(Y4))
                    f.write('\n')




