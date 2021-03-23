import torch
import argparse
import numpy as np
from tools.plot import load_class_names
from tools.post_process import post_process
from tools.utils import get_batch_statistics, ap_per_class
from tools.load import LoadDataset
from models.yolo import Model

import random
def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #  "../rotation_v4/data/test1"     ../UVA/test     ../total/train/  ../SSDD+_val
    parser.add_argument("--test_folder", type=str, default="../SSDD+_val", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="./weights1/yolov5_170.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/ssdd+.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.75, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold for evaluation")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
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
            imgs.to(device)
            outputs, loss ,loss1,loss2,loss3 = model(imgs)
            outputs = torch.cat(outputs, 1)

            outputs = post_process(outputs.cpu(), conf_thres=args.conf_thres, nms_thres=args.nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=args.iou_thres)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    with open("./map.txt",'a') as f:
        f.write(f"mAP: {AP.mean()}")
        f.write('\n')
