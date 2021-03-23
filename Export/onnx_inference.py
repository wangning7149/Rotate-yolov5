# -*-coding: utf-8 -*-
from models import *

import torch
import numpy as np


import os, sys
import cv2
from PIL import Image
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import numpy as np


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes ,ds= self.onnx_session.run(self.output_name, {self.input_name[0]: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        layer1, layer2, layer3 = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return layer1, layer2, layer3
import torchvision.transforms as transforms
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad
import Utils
import torch.nn.functional as F
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
if __name__ == '__main__':
    my = ONNXModel('./yolov5.onnx')

    # my = ONNXModel('yolov3_original_float_model.onnx')
    origin_image = cv2.imread('../data/detect/459.jpg')
    image = transforms.ToTensor()(Image.open('../data/detect/459.jpg').convert('RGB'))
    classes = Utils.get_classes()
    num_classes = len(classes)
    img,pad = pad_to_square(image,0)
    # Resize
    image = resize(img, 672)


    # image /= 255.0
    image = np.expand_dims(image, axis=0)

    layer1, layer2, layer3 = my.forward(image)

    ww = layer1[0][0]
    # np.savetxt("./ets.txt", ww, fmt='%f', delimiter=',')

    outputs = []
    outputs.append(layer1)
    outputs.append(layer2)
    outputs.append(layer3)
    radian = np.pi / 180
    anchors_ = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]]
    ]

    angles_ = [
        [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90],
        [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90],
        [-radian * 60, -radian * 30, 0, radian * 30, radian * 60, radian * 90]
    ]

    scale_x_y_ = [1.2, 1.1, 1.05]
    ignore_thresh = 0.6
    stride_ = [8, 16, 32]

    outputs[0] = np.transpose(outputs[0], (0, 3, 1, 2)).reshape([1, 18, 8, 84, 84])
    outputs[0] = np.transpose(outputs[0], (0, 1, 3, 4, 2))
    outputs[1] = np.transpose(outputs[1], (0, 3, 1, 2)).reshape([1, 18, 8, 42, 42])
    outputs[1] = np.transpose(outputs[1], (0, 1, 3, 4, 2))
    outputs[2] = np.transpose(outputs[2], (0, 3, 1, 2)).reshape([1, 18, 8, 21, 21])
    outputs[2] = np.transpose(outputs[2], (0, 1, 3, 4, 2))
    # [1,18,84,84,8]

    res = []
    num_anchors = 18
    for i in range(2):
        scale_x_y = scale_x_y_[i]
        stride = stride_[i]
        masked_anchors = [(a_w / stride, a_h / stride, a) for a_w, a_h in anchors_[i] for a in
                          angles_[i]]
        prediction = outputs[i]
        batch_size, grid_size = prediction.shape[0], prediction.shape[2]
        pred_x = Utils.sigmoid(prediction[..., 0]) * scale_x_y - (
                scale_x_y - 1) / 2  # [1,18,92,92]
        pred_y = Utils.sigmoid(prediction[..., 1]) * scale_x_y - (scale_x_y - 1) / 2
        pred_w = prediction[..., 2]
        pred_h = prediction[..., 3]
        pred_a = prediction[..., 4]
        pred_conf = Utils.sigmoid(prediction[..., 5])
        pred_cls = Utils.sigmoid(prediction[..., 6:])  # [1,18,92,92,2]

        # grid.shape-> [1, 1, 52, 52, 1]
        # 这一步预测的(pred_x, pred_y)是相对于每一个cell左上角的点
        # 需要由左上角往右下角配合grid_size加上对应的的offset，才能计算出  正确位置
        grid_x = np.arange(grid_size).reshape([1, grid_size]).repeat(grid_size, 0)
        grid_x = np.expand_dims(grid_x, 0)
        grid_x = np.expand_dims(grid_x, 0)
        grid_x = np.array(grid_x, dtype=np.float)
        # grid_x [1,1,92,92]
        grid_y = np.arange(grid_size).reshape([1, grid_size]).repeat(grid_size, 0).T
        grid_y = np.expand_dims(grid_y, 0)
        grid_y = np.expand_dims(grid_y, 0)
        grid_y = np.array(grid_y, dtype=np.float)

        # anchor.shape-> [1, 3, 1, 1, 1]
        masked_anchors = np.array(masked_anchors, dtype=np.float)
        anchor_w = masked_anchors[:, 0].reshape([1, num_anchors, 1, 1])
        anchor_h = masked_anchors[:, 1].reshape([1, num_anchors, 1, 1])
        anchor_a = masked_anchors[:, 2].reshape([1, num_anchors, 1, 1])  # [1,18,1,1]

        # decode
        pred_boxes = np.zeros(prediction[..., :5].shape, dtype=np.float)
        pred_boxes[..., 0] = (pred_x + grid_x)

        pred_boxes[..., 1] = (pred_y + grid_y)
        pred_boxes[..., 2] = (np.exp(pred_w) * anchor_w)
        pred_boxes[..., 3] = (np.exp(pred_h) * anchor_h)
        pred_boxes[..., 4] = pred_a + anchor_a  # [1,18,92,92,5]   #  在这里的box 都是在feature map上的大小

        output = np.concatenate(
            (
                np.concatenate((pred_boxes[..., :4] * stride_[i], pred_boxes[..., 4:]), -1).reshape(batch_size, -1, 5),
                pred_conf.reshape(batch_size, -1, 1),
                pred_cls.reshape(batch_size, -1, num_classes),
            ),
            -1,
        )  # [1,123232,8]

        res.append(output)
    res = np.concatenate(res, 1)

    box = Utils.post_process(res, conf_thres=0.9, nms_thres=0.3)

    Utils.plot_boxes(origin_image, box[0], classes, [672, 672])
    #
    # layer1, layer2, layer3 = torch.tensor(layer1), torch.tensor(layer2), torch.tensor(layer3)
    # model_def = 'config/prune_0.5_yolov3-hand_0.01.cfg'
    # model = Darknet_inference(model_def)
    # out = model([layer1, layer2, layer3])
    # outputs = non_max_suppression(out, conf_thres=0.4, nms_thres=0.5)[0]
    #
    # for i in range(len(outputs)):
    #     a = int(outputs[i][0])
    #     b = int(outputs[i][1])
    #     c = int(outputs[i][2])
    #     d = int(outputs[i][3])
    #     cv2.rectangle(origin, (a, b), (c, d), (0, 200, 9), 2)
    #     cv2.putText(origin, str(int(outputs[i][-1])), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255))
    #
    # cv2.imwrite('./onnx.jpg' , origin)
    # print(outputs.shape)
