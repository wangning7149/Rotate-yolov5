import torch
import time
import argparse
import os
from torch.utils.data import DataLoader
from tools.plot import load_class_names, plot_boxes

from tools.load import ImageDataset
from models.yolo import Model
import random

def iou(box1,box2):


    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
    b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2
    # iou = torch.sqrt((b2_x1 - b1_x1)* (b2_x1 - b1_x1) + (b2_y1 - b1_y1)*(b2_y1 - b1_y1))
    # Union Area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    eps = 1e-9
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou



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
        keep_boxes = torch.ones(box.shape[0], dtype=torch.bool).to(device)
        for i in range(box.shape[0] - 1):
            if keep_boxes[i]:
                overlap = iou(box[i,:5], box[(i + 1):,:5])

                keep_overlap = torch.logical_or(overlap < nms_thres, torch.tensor(0))
                keep_boxes[(i + 1):] = torch.logical_and(keep_overlap, keep_boxes[(i + 1):])
        idxes = torch.where(keep_boxes)
        bbox = box[idxes]
        xc = bbox[...,-2] > 0

        output[j] = bbox[xc]
    return output # 最后一个 是类别  末二个是 分数
def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/detect-SSDD+", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="./checkpoint/ssdd+_model/yolov5_85.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/ssdd+.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.65, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=672, help="size of each image dimension")
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    args = parser.parse_args()
    print(args)
    init()
    with open(args.class_path,'r') as f:
        lines = f.readlines()
    nc = 0
    label_dict = {}
    for l in lines:
        l = l.strip('\n')
        if l:
            label_dict[nc] = l
            nc += 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_class_names(args.class_path)
    pretrained_dict = torch.load(args.weights_path, map_location=torch.device('cpu'))
    model = Model(args.cfg,nc=nc)
    model = model.to(device)

    model.load_state_dict(pretrained_dict)

    model.eval()

    dataset = ImageDataset(args.image_folder, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    boxes = []
    imgs = []

    start = time.time()
    for img_path, img in dataloader:
        flag = False
        img = torch.autograd.Variable(img.type(FloatTensor))
        with torch.no_grad():
            temp = time.time()

            output ,_ ,loss1,loss2,loss3 = model(img)  # batch=1 -> [1, n, n], batch=3 -> [3, n, n]

            # print(output[0].shape)
            # print(output[0])
            # ww = output[0][0][0]
            # np.savetxt("./ets.txt", ww, fmt='%f', delimiter=',')
            #
            # exit()
            output = torch.cat(output,1)

            temp1 = time.time()

            box = post_process(output, args.conf_thres, args.nms_thres)

            for j in range(len(box)):
                if box[j] == None:
                    flag = True
            if flag:
                continue
            temp2 = time.time()

            boxes.extend(box)


            print('-----------------------------------')
            num = 0
            for b in box:
                num += len(b)
            print("{}-> {} objects found".format(img_path, num))
            print("Inference time : ", round(temp1 - temp, 5))
            print("Post-processing time : ", round(temp2 - temp1, 5))
            print('-----------------------------------')

            imgs.extend(img_path)
    rms = os.listdir('./outputs')
    if len(rms):
        for i in rms:
            os.system('rm ./outputs/' + i)

    for i, (img_path, box) in enumerate(zip(imgs, boxes)):
        plot_boxes(img_path, box.cpu(), class_names, args.img_size)

    end = time.time()

    print('-----------------------------------')
    print("Total detecting time : ", round(end - start, 5))
    print('-----------------------------------')
