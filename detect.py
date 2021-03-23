import torch
import time
import argparse
import os
from torch.utils.data import DataLoader
from tools.plot import load_class_names, plot_boxes
from tools.post_process import post_process
from tools.load import ImageDataset
from models.yolo import Model
import random
def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/detect-AOD", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="./checkpoint/aod_model/yolov5_510.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.75, help="object confidence threshold")
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

            box = post_process(output.cpu(), args.conf_thres, args.nms_thres)

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
        plot_boxes(img_path, box, class_names, args.img_size)

    end = time.time()

    print('-----------------------------------')
    print("Total detecting time : ", round(end - start, 5))
    print('-----------------------------------')
