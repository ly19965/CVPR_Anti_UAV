"""
siamfc code from https://github.com/got-10k/siamfc
Thanks
"""
from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-6, momentum=0.05),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)

        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out


class TrackerSiamFC(object):

    def __init__(self, net_path=None, **kargs):
        self.name = 'Yolo_SiamFC'
        self.cfg = self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
            # yolov5 hyperparameters
            'imgsz': [640, 640],
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'max_det': 1000,
            'augment': False,
            'classes': None,
            'agnostic_nms': False,
            'hide_labels': False,
            'hide_conf': False,
            'weight': '/path/to/baseline/yolov5/runs/train/exp/weights/best.pt',
            'data': '/path/to/baseline/yolov5/data/antiuav.yaml',
            'device': '0'}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def initialize_yolo(self):
        weights = self.cfg.weight  # model.pt path(s)
        data = self.cfg.data # dataset.yaml path
        device = self.cfg.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu

        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        return model

    def init(self, image, model):
        image = np.asarray(image)

        device = select_device(self.cfg.device)
        # model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.cfg.imgsz, s=stride)  # check image size

        # resize_im
        im = letterbox(image, imgsz, stride=stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        # Run inference
        bs = 1
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
        # Inference
        with dt[1]:
            visualize = False
            pred = model(im, augment=self.cfg.augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.cfg.conf_thres, self.cfg.iou_thres, self.cfg.classes,
                                       self.cfg.agnostic_nms, max_det=self.cfg.max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = image.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        if len(det) == 0:
            return [0], im0
        else:
            pred_bbox = np.array(det[0][:4].cpu())
            target_bbox = [np.float64(pred_bbox[0]), np.float64(pred_bbox[1]), pred_bbox[2]-pred_bbox[0]+1, pred_bbox[3]-pred_bbox[1]+1]  # [x1, y1, w, h]

            # convert box to 0-indexed and center based [y, x, h, w]
            box = np.array([
                target_bbox[1] - 1 + (target_bbox[3] - 1) / 2,
                target_bbox[0] - 1 + (target_bbox[2] - 1) / 2,
                target_bbox[3], target_bbox[2]], dtype=np.float32)
            self.center, self.target_sz = box[:2], box[2:]

            # create hanning window
            self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
            self.hann_window = np.outer(
                np.hanning(self.upscale_sz),
                np.hanning(self.upscale_sz))
            self.hann_window /= self.hann_window.sum()

            # search scale factors
            self.scale_factors = self.cfg.scale_step ** np.linspace(
                -(self.cfg.scale_num // 2),
                self.cfg.scale_num // 2, self.cfg.scale_num)

            # exemplar and search sizes
            context = self.cfg.context * np.sum(self.target_sz)
            self.z_sz = np.sqrt(np.prod(self.target_sz + context))
            self.x_sz = self.z_sz * \
                        self.cfg.instance_sz / self.cfg.exemplar_sz

            # exemplar image
            self.avg_color = np.mean(image, axis=(0, 1))
            exemplar_image = self._crop_and_resize(
                image, self.center, self.z_sz,
                out_size=self.cfg.exemplar_sz,
                pad_color=self.avg_color)

            # exemplar features
            exemplar_image = torch.from_numpy(exemplar_image).to(
                self.device).permute([2, 0, 1]).unsqueeze(0).float()
            with torch.set_grad_enabled(False):
                self.net.eval()
                self.kernel = self.net.feature(exemplar_image)


            # visualize using Annotator
            annotator = Annotator(im0, line_width=3, example=str(names))
            det_disp = det
            for *xyxy, conf, cls in reversed(det_disp):
                c = int(cls)  # integer class
                label = None if self.cfg.hide_labels else (names[c] if self.cfg.hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
                # Stream results
            im_disp = annotator.result()

            return target_bbox, im_disp


    def update(self, image):
        image = np.asarray(image)

        # search images
        instance_images = [self._crop_and_resize(
            image, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            pad_color=self.avg_color) for f in self.scale_factors]
        instance_images = np.stack(instance_images, axis=0)
        instance_images = torch.from_numpy(instance_images).to(
            self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            self.net.eval()
            instances = self.net.feature(instance_images)
            responses = F.conv2d(instances, self.kernel) * 0.001
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            t, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box if response.max() > 1e-5 else np.array([0])

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch
