import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable
from torch.cuda import amp
from models.yolo import Model
from tools.load import LoadDataset
from tools.scheduler import CosineAnnealingWarmupRestarts


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA


def weights_init_normal(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #   ../total/train/  ../rotation_v4/data/train1
    parser.add_argument("--train_folder", type=str, default='../rotation_v4/data/train1', help="path to dataset")
    # parser.add_argument("--train_folder", type=str, default="../UVA/train", help="path to dataset")
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--weights_path", type=str, default="./weights1/yolov5_0.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
    parser.add_argument("--subdivisions", type=int, default=3, help="size of mini batches")
    parser.add_argument("--img_size", type=int, default=672, help="size of each image dimension")
    args = parser.parse_args()
    print(args)
    init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.class_path,'r') as f:
        lines = f.readlines()
    nc = 0
    label_dict = {}
    for l in lines:
        l = l.strip('\n')
        if l:
            label_dict[nc] = l
            nc += 1
    model = Model(args.cfg,nc=nc)
    model = model.to(device)
    model_dict = model.state_dict()
    if args.resume:
        pretrained_dict = torch.load(args.weights_path, map_location=device)
        pretrained_dict = {k: v for i, (k, v) in enumerate(pretrained_dict.items()) if i < 552}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.apply(weights_init_normal)
    model.load_state_dict(model_dict)

    _, train_dataloader = LoadDataset(args.train_folder, args.img_size, args.batch_size,augment=False)
    num_iters_per_epoch = len(train_dataloader)
    scheduler_iters = round(args.epochs * len(train_dataloader) / args.subdivisions)
    total_step = num_iters_per_epoch * args.epochs

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=scheduler_iters,
                                              cycle_mult=1.0,
                                              max_lr=args.lr,
                                              min_lr=0,
                                              warmup_steps=round(scheduler_iters * 0.1),
                                              gamma=1.0)
    cuda = device.type != 'cpu'
    # scaler = amp.GradScaler(enabled=cuda)
    model.to(device)

    ema = ModelEMA(model)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        log_str = "\n---- [Epoch %d/%d] ----\n" % (epoch + 1, args.epochs)
        print(log_str)

        for batch, (_, imgs, targets,shapes) in enumerate(train_dataloader):

            # with open("./we.txt",'a') as f:
            #     f.write(str(_))
            #     f.write('\n')
            global_step = num_iters_per_epoch * epoch + batch + 1
            imgs = Variable(imgs.to(device), requires_grad=False)
            targets = Variable(targets.to(device), requires_grad=False)

            outputs, loss, loss_reg, loss_conf, loss_cls = model(imgs, targets)

            # todo 2
            loss.backward()
            total_loss += loss.item()

            if global_step % args.subdivisions == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_log = [
                "Step: %d/%d, loss: %f, reg_loss: %f, conf_loss: %f, cls_loss: %f" % (global_step, total_step, loss
                                                                                      , loss_reg, loss_conf, loss_cls)]
            print(loss_log)
        ckpt = {'epoch': epoch,

                'model': ema.ema, }
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "weights1/yolov5_{}.pth".format(epoch))
    torch.save(model.state_dict(), "weights1/yolov5_train.pth")
