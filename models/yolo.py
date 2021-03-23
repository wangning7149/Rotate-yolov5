import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import numpy as np
from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class FocalLoss(nn.Module):
    # Reference: https://github.com/ultralytics/yolov5/blob/8918e6347683e0f2a8a3d7ef93331001985f6560/utils/loss.py#L32
    def __init__(self, alpha=0.25, gamma=2, reduction="none"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
def bbox_xywha_ciou(pred_boxes, target_boxes):
    """
    :param pred_boxes: [num_of_objects, 4], boxes predicted by yolo and have been scaled
    :param target_boxes: [num_of_objects, 4], ground truth boxes and have been scaled
    :return: ciou loss
    """
    assert pred_boxes.size() == target_boxes.size()

    # xywha -> xyxya
    pred_boxes = torch.cat(
        [pred_boxes[..., :2] - pred_boxes[..., 2:4] / 2,
         pred_boxes[..., :2] + pred_boxes[..., 2:4] / 2,
         pred_boxes[..., 4:]], dim=-1)
    target_boxes = torch.cat(
        [target_boxes[..., :2] - target_boxes[..., 2:4] / 2,
         target_boxes[..., :2] + target_boxes[..., 2:4] / 2,
         target_boxes[..., 4:]], dim=-1)

    w1 = pred_boxes[:, 2] - pred_boxes[:, 0]
    h1 = pred_boxes[:, 3] - pred_boxes[:, 1]
    w2 = target_boxes[:, 2] - target_boxes[:, 0]
    h2 = target_boxes[:, 3] - target_boxes[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (pred_boxes[:, 2] + pred_boxes[:, 0]) / 2
    center_y1 = (pred_boxes[:, 3] + pred_boxes[:, 1]) / 2
    center_x2 = (target_boxes[:, 2] + target_boxes[:, 0]) / 2
    center_y2 = (target_boxes[:, 3] + target_boxes[:, 1]) / 2

    inter_max_xy = torch.min(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    inter_min_xy = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    out_max_xy = torch.max(pred_boxes[:, 2:4], target_boxes[:, 2:4])
    out_min_xy = torch.min(pred_boxes[:, :2], target_boxes[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = inter_diag / outer_diag

    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)

    # alpha is a constant, it don't have gradient
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)

    ciou_loss = iou - (u + alpha * v)
    ciou_loss = torch.clamp(ciou_loss, min=-1.0, max=1.0)

    angle_factor = torch.abs(torch.cos(pred_boxes[:, 4] - target_boxes[:, 4]))
    # skew_iou = torch.abs(iou * angle_factor) + 1e-16
    skew_iou = iou * angle_factor
    return skew_iou, ciou_loss
def anchor_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

class Detect(nn.Module):
    stride = None  # strides computed during build
    # Export = False  # onnx Export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        radian = np.pi / 180
        self.num_classes = nc  # number of classes

        self.nl = 2  # number of detection layers  3
        self.num_anchors = 18  # number of anchors  3
        output_ch = (5 + 1 + nc) * 3 * 6
        self.output_ch = output_ch  # number of outputs per anchor
        self.reduction = "mean"

        # yolov5
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]


        self.angles =[
            [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90],
            [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90],
            [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90]
        ]


        self.scale_x_y = [1.2,1.1,1.05]
        self.ignore_thresh=0.6
        self.stride = [8,16,32]
        self.lambda_coord = 1.0
        self.lambda_conf_scale = 10.0
        self.lambda_cls_scale = 1.0
        self.m = nn.ModuleList(nn.Conv2d(x, output_ch, 1) for x in ch)  # output conv


    def build_targets(self,stride, masked_anchors,pred_boxes, pred_cls, target):
        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
        nB, nA, nG, _, nC = pred_cls.size() #nB,batch nA,anchor数量 nG,:边长 _, nC:class_num

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0) # [1,18,92,92]
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        skew_iou = FloatTensor(nB, nA, nG, nG).fill_(0)
        ciou_loss = FloatTensor(nB, nA, nG, nG).fill_(0)
        ta = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert ground truth position to position that relative to the size of box (grid size)
        target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:]), dim=-1) # [6,5]
        gxy = target_boxes[:, :2] #[6,2]
        gwh = target_boxes[:, 2:4] # [6,2]
        ga = target_boxes[:, 4] # [6]

        # Get anchors with best iou and their angle difference with ground truths
        arious = []
        offset = []
        for anchor in masked_anchors:   # [18,3]
            ariou = anchor_wh_iou(anchor[:2], gwh)  # [6]
            cos = torch.abs(torch.cos(torch.sub(anchor[2], ga)))
            arious.append(ariou * cos)
            offset.append(torch.abs(torch.sub(anchor[2], ga)))
        arious = torch.stack(arious)  # [18,6]
        offset = torch.stack(offset)  # [18,6] 角度的偏移量

        best_ious, best_n = arious.max(0)  #[6] 找出与target最匹配的anchor

        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gi, gj = gxy.long().t()

        # Set masks to specify object's location

        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0
        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, (anchor_ious, angle_offset) in enumerate(zip(arious.t(), offset.t())):

            noobj_mask[b[i], (anchor_ious > self.ignore_thresh), gj[i], gi[i]] = 0
            # if iou is greater than 0.4 and the angle offset if smaller than 15 degrees then ignore training
            noobj_mask[b[i], (anchor_ious > 0.4) & (angle_offset < (np.pi / 12)), gj[i], gi[i]] = 0


        # Angle (encode)
        ta[b, best_n, gj, gi] = ga - masked_anchors[best_n][:, 2]

        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        tconf = obj_mask.float()

        # Calculate ciou loss
        iou, ciou = bbox_xywha_ciou(pred_boxes[b, best_n, gj, gi], target_boxes)
        with torch.no_grad():
            img_size = stride * nG
            bbox_loss_scale = 2.0 - 1.0 * gwh[:, 0] * gwh[:, 1] / (img_size ** 2)
        ciou = bbox_loss_scale * (1.0 - ciou)

        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = iou

        # for reg loss
        skew_iou[b, best_n, gj, gi] = torch.exp(1 - iou) - 1

        # unit vector for reg loss
        ciou_loss[b, best_n, gj, gi] = ciou

        obj_mask = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf

    def forward(self, x,target=None,export=False):
        # x = x.copy()  # for profiling
        # self.nl = len(x)

        if export:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            return x

        else:
            outputs = []
            Pred_cls = []
            Pred_boxes = []


            for i in range(self.nl):
                scale_x_y = self.scale_x_y[i]
                output = self.m[i](x[i])
                stride  =self.stride[i]
                masked_anchors = [(a_w / stride, a_h / stride, a) for a_w, a_h in self.anchors[i] for a in
                               self.angles[i]]
                FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor
                #
                # output.shape-> [batch_size, num_anchors * (num_classes + 5), grid_size, grid_size]
                batch_size, grid_size = output.size(0), output.size(2)

                # prediction.shape-> torch.Size([1, num_anchors, grid_size, grid_size, num_classes + 5])
                prediction = (
                    output.view(batch_size, self.num_anchors, self.num_classes + 6, grid_size, grid_size)
                        .permute(0, 1, 3, 4, 2).contiguous()
                )  # [1,18,92,92,8]

                pred_x = torch.sigmoid(prediction[..., 0]) * scale_x_y - (
                            scale_x_y - 1) / 2  # [1,18,92,92]
                pred_y = torch.sigmoid(prediction[..., 1]) * scale_x_y - (scale_x_y - 1) / 2
                pred_w = prediction[..., 2]
                pred_h = prediction[..., 3]
                pred_a = prediction[..., 4]
                pred_conf = torch.sigmoid(prediction[..., 5])
                pred_cls = torch.sigmoid(prediction[..., 6:])  # [1,18,92,92,2]

                grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(
                    FloatTensor)
                # grid_x [1,1,92,92]
                grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(
                    FloatTensor)


                masked_anchors = FloatTensor(masked_anchors)  # [18,3] 18个anchor，（w,h,theta）
                anchor_w = masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
                anchor_h = masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])
                anchor_a = masked_anchors[:, 2].view([1, self.num_anchors, 1, 1])  # [1,18,1,1]

                # decode
                pred_boxes = FloatTensor(prediction[..., :5].shape)
                pred_boxes[..., 0] = (pred_x + grid_x)
                pred_boxes[..., 1] = (pred_y + grid_y)
                pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
                pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
                pred_boxes[..., 4] = pred_a + anchor_a  # [1,18,92,92,5]   #  在这里的box 都是在feature map上的大小

                output = torch.cat(
                    (
                        torch.cat([pred_boxes[..., :4] * self.stride[i], pred_boxes[..., 4:]], dim=-1).view(batch_size,
                                                                                                         -1, 5),
                        pred_conf.view(batch_size, -1, 1),
                        pred_cls.view(batch_size, -1, self.num_classes),
                    ),
                    -1,
                )  # [1,123232,8]
                Pred_cls.append(pred_cls)
                Pred_boxes.append(pred_boxes)
                outputs.append(output)

        if target is None:

            return outputs,0,0,0,0
        else:
            sum_loss = 0.
            outputs = []
            loss_reg = 0.
            loss_cls = 0.
            loss_conf = 0.


            for i in range(self.nl):
                scale_x_y = self.scale_x_y[i]
                output = self.m[i](x[i])
                stride = self.stride[i]
                masked_anchors = [(a_w / stride, a_h / stride, a) for a_w, a_h in self.anchors[i] for a in
                                  self.angles[i]]
                FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor
                #
                # output.shape-> [batch_size, num_anchors * (num_classes + 5), grid_size, grid_size]
                batch_size, grid_size = output.size(0), output.size(2)

                # prediction.shape-> torch.Size([1, num_anchors, grid_size, grid_size, num_classes + 5])
                prediction = (
                    output.view(batch_size, self.num_anchors, self.num_classes + 6, grid_size, grid_size)
                        .permute(0, 1, 3, 4, 2).contiguous()
                )  # [1,18,92,92,8]

                pred_x = torch.sigmoid(prediction[..., 0]) * scale_x_y - (
                        scale_x_y - 1) / 2  # [1,18,92,92]
                pred_y = torch.sigmoid(prediction[..., 1]) * scale_x_y - (scale_x_y - 1) / 2
                pred_w = prediction[..., 2]
                pred_h = prediction[..., 3]
                pred_a = prediction[..., 4]
                pred_conf = torch.sigmoid(prediction[..., 5])
                pred_cls = torch.sigmoid(prediction[..., 6:])  # [1,18,92,92,2]

                grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(
                    FloatTensor)
                # grid_x [1,1,92,92]
                grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(
                    FloatTensor)

                # anchor.shape-> [1, 3, 1, 1, 1]
                masked_anchors = FloatTensor(masked_anchors)  # [18,3] 18个anchor，（w,h,theta）
                anchor_w = masked_anchors[:, 0].view([1, self.num_anchors, 1, 1])
                anchor_h = masked_anchors[:, 1].view([1, self.num_anchors, 1, 1])
                anchor_a = masked_anchors[:, 2].view([1, self.num_anchors, 1, 1])  # [1,18,1,1]

                # decode
                pred_boxes = FloatTensor(prediction[..., :5].shape)
                pred_boxes[..., 0] = (pred_x + grid_x)
                pred_boxes[..., 1] = (pred_y + grid_y)
                pred_boxes[..., 2] = (torch.exp(pred_w) * anchor_w)
                pred_boxes[..., 3] = (torch.exp(pred_h) * anchor_h)
                pred_boxes[..., 4] = pred_a + anchor_a  # [1,18,92,92,5]   #  在这里的box 都是在feature map上的大小

                output = torch.cat(
                    (
                        torch.cat([pred_boxes[..., :4] * self.stride[i], pred_boxes[..., 4:]], dim=-1).view(
                            batch_size,
                            -1, 5),
                        pred_conf.view(batch_size, -1, 1),
                        pred_cls.view(batch_size, -1, self.num_classes),
                    ),
                    -1,
                )  # [1,123232,8]
                outputs.append(output)


                iou_scores, skew_iou, ciou_loss, class_mask, obj_mask, noobj_mask, ta, tcls, tconf = self.build_targets(
                    self.stride[i],masked_anchors,pred_boxes=pred_boxes, pred_cls=pred_cls, target=target
                )
                iou_const = skew_iou[obj_mask]
                angle_loss = F.smooth_l1_loss(pred_a[obj_mask], ta[obj_mask], reduction="none")
                reg_loss = angle_loss + ciou_loss[obj_mask]
                with torch.no_grad():
                    reg_const = iou_const / reg_loss
                reg_loss = (reg_loss * reg_const).mean()

                # Focal Loss for object's prediction
                FOCAL = FocalLoss(reduction=self.reduction)
                conf_loss = (
                        FOCAL(pred_conf[obj_mask], tconf[obj_mask])
                        + FOCAL(pred_conf[noobj_mask], tconf[noobj_mask])
                )

                # Binary Cross Entropy Loss for class' prediction
                cls_loss = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction)

                # Loss scaling
                reg_loss = self.lambda_coord * reg_loss
                conf_loss = self.lambda_conf_scale * conf_loss
                cls_loss = self.lambda_cls_scale * cls_loss
                loss_reg += reg_loss
                loss_cls += cls_loss
                loss_conf += conf_loss
                layer_loss = reg_loss + conf_loss + cls_loss
                sum_loss += layer_loss * batch_size

            return outputs, sum_loss,loss_reg,loss_conf,loss_cls


    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None,export=False):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        self.export = export
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     s = 128  # 2x min stride
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     check_anchor_order(m)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x,target=None,export=False):
        return self.forward_once(x,target,export=export)  # single-scale inference, train

    def forward_once(self, x, target=None,export=False):
        if export:
            y, dt = [], []  # outputs

            for m in self.model:

                if m.f != -1:  # if not from previous layer

                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(m, Detect):
                    x = m(x, export=export)


                else:
                    x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

            return x
        else:

            y, dt = [], []  # outputs
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if isinstance(m,Detect):
                    x,loss,loss_reg,loss_conf,loss_cls = m(x,target=target)

                else:
                    x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

            return x ,loss,loss_reg,loss_conf,loss_cls

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()

        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']

    na = 18
    no = na * 6  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]  # c1 是输出的通道数   c2 是输出的通道数

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type

        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    input = torch.ones((1,3,672,672))
    model = Model(opt.cfg).to(device)
    # print(model)
    model.train()
    model(input)



