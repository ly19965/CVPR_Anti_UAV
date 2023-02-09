# Part of the implementation is borrowed and modified from MTTR,
# publicly available at https://github.com/mttr2021/MTTR

import os.path as osp
from typing import Any, Dict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import cv2
from collections import namedtuple

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

@MODELS.register_module(
    Tasks.video_single_object_tracking,
    module_name='siamfc')
class SimfcTrack(TorchModel):

    def __init__(self, *args, **kwargs):
        """str -- model file root."""
        super().__init__(*args, **kwargs)

        self.cfg = kwargs.get('cfg')
        self.tracker_name = 'SiamFC'
        device = kwargs.get('device')
        self.device = torch.device("cuda" if device == 'gpu' else "cpu")
        self.net = SiamFC().to(self.device)

    def initialize(self, image, info: dict):
        image = np.asarray(image)
        box = info['init_bbox']

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        box = np.array(box)
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

    def track(self, image, info: dict = None):
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

        # return 1-indexed and [x0, y0, x1, y1] bounding box
        x0 =  self.center[1] + 1 - (self.target_sz[1] - 1) / 2
        y0 = self.center[0] + 1 - (self.target_sz[0] - 1) / 2
        x1 = x0 + self.target_sz[1] - 1
        y1 = y0 + self.target_sz[0] - 1
        box = np.array([x0, y0, x1, y1])

        return {'target_bbox': box} #if response.max() > 1e-5 else np.array([0])

    def forward(self, batch, device):
        self.net.train()

        z = batch[0]
        x = batch[1]

        responses = self.net(z, x)
        labels, weights = self._create_labels(responses.size(), device=device)
        loss = F.binary_cross_entropy_with_logits(
        responses, labels, weight=weights, size_average=True)

        return {'total_loss': loss}

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

    def inference(self, **kwargs):
        window = kwargs['window']
        text_query = kwargs['text_query']
        video_metadata = kwargs['metadata']

        window = nested_tensor_from_videos_list([window])
        valid_indices = torch.arange(len(window.tensors))
        if self._device_name == 'gpu':
            valid_indices = valid_indices.cuda()
        outputs = self.model(window, valid_indices, [text_query])
        window_masks = self.postprocessor(
            outputs, [video_metadata],
            window.tensors.shape[-2:])[0]['pred_masks']
        return window_masks

    def _create_labels(self, size, device):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(device).float()
        self.weights = torch.from_numpy(weights).to(device).float()

        return self.labels, self.weights

    def _initialize_weights(self):
        self.net._initialize_weights()

    def load_pretrain(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in load_dict.items():
            new_dict['net.' + k] = v
        self.load_state_dict(new_dict)
        
    @classmethod
    def _instantiate(cls, **kwargs):
        model_file = kwargs.get('am_model_name', ModelFile.TORCH_MODEL_FILE)
        ckpt_path = os.path.join(kwargs['model_dir'], model_file)
        #logger.info(f'loading model from {ckpt_path}')
        model_dir = kwargs.pop('model_dir')
        model = cls(**kwargs)
        ckpt_path = os.path.join(model_dir, model_file)
        load_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in load_dict.items():
            new_dict['net.' + k] = v
        model.load_state_dict(new_dict)
        return model
