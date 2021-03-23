"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ Export PYTHONPATH="$PWD" && python models/Export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from Export.yolo1 import Model
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
import random
import numpy as np
def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoint/ssdd+_model/yolov5_85.pth', help='weights path')  # from yolov5/models/
    parser.add_argument("--class_path", type=str, default="data/ssdd+.names", help="path to class label file")
    parser.add_argument('--img-size', nargs='+', type=int, default=[672, 672], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    opt = parser.parse_args()
    init()
    with open(opt.class_path,'r') as f:
        lines = f.readlines()
    nc = 0
    label_dict = {}
    for l in lines:
        l = l.strip('\n')
        if l:
            label_dict[nc] = l
            nc += 1
    print(opt)
    set_logging()
    pretrained_dict = torch.load(opt.weights, map_location=torch.device('cpu'))
    t = time.time()
    model = Model(opt.cfg,nc=nc)


    model.load_state_dict(pretrained_dict)

    model.eval()

    # model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model


    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign Export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = nn.LeakyReLU()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    with torch.no_grad():
        y = model(img,export=True)  # dry run
    print(model)

    try:
        import onnx

        print('\nStarting ONNX Export with onnx %s...' % onnx.__version__)

        torch.onnx.export(model, img, 'yolov5.onnx', verbose=False, opset_version=10, input_names=['data'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks

    except Exception as e:
        print('ONNX Export failure: %s' % e)


    print('\n Successful')
