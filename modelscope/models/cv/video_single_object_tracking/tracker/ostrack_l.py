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
from torchvision import models 
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss

from modelscope.models.cv.video_single_object_tracking.config.ostrack import \
    cfg
from modelscope.models.cv.video_single_object_tracking.models.ostrack.ostrack import \
    build_ostrack
from modelscope.models.cv.video_single_object_tracking.utils.utils import (
    Preprocessor, clip_box, generate_mask_cond, hann2d, sample_target,
    transform_image_to_crop)
from modelscope.models.cv.video_single_object_tracking.utils.misc import NestedTensor
from modelscope.models.cv.video_single_object_tracking.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from modelscope.models.cv.video_single_object_tracking.utils.merge import merge_template_search
from modelscope.models.cv.video_single_object_tracking.utils.heapmap_utils import generate_heatmap
from modelscope.models.cv.video_single_object_tracking.utils.ce_utils import generate_mask_cond, adjust_keep_rate
from modelscope.models.cv.video_single_object_tracking.utils.box_ops import giou_loss
from modelscope.models.cv.video_single_object_tracking.utils.focal_loss import FocalLoss

logger = get_logger()

@MODELS.register_module(
    Tasks.video_single_object_tracking,
    module_name='ostrack_l')
class OsTrackL(TorchModel):

    def __init__(self, *args, **kwargs):
        """str -- model file root."""
        super().__init__(*args, **kwargs)

        self.cfg = cfg
        model_cfg = kwargs.get('model_cfg')
        #if hasattr('backbone_stride'):
        #    self.cfg.MODEL.BACKBONE.STRIDE = model_cfg.backbone_stride
        self.tracker_name = 'Ostrack'
        device = kwargs.get('device')
        self.device = torch.device("cuda")# if device == 'gpu' else "cpu")
        network = build_ostrack(cfg)
        self.net = network

        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.preprocessor = Preprocessor(self.device)
        # motion constrain
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True).to(self.device)

        self.frame_id = 0
        # for save boxes from all queries
        self.z_dict1 = {}
        focal_loss = FocalLoss()
        self.loss_objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        self.loss_weight = {'giou': 1, 'l1': 1, 'focal': 1}


    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info['init_bbox'],
            self.cfg.TEST.TEMPLATE_FACTOR,
            output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor,
                template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1,
                                                 template.tensors.device,
                                                 template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.cfg.TEST.SEARCH_FACTOR,
            output_sz=self.cfg.TEST.SEARCH_SIZE)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.net.forward(
                template=self.z_dict1.tensors,
                search=x_dict.tensors,
                ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.net.box_head.cal_bbox(response,
                                                    out_dict['size_map'],
                                                    out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.cfg.TEST.SEARCH_SIZE
                    / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        x1, y1, w, h = self.state
        x2 = x1 + w
        y2 = y1 + h
        return {'target_bbox': [x1, y1, x2, y2]}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[
            1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.TEST.SEARCH_SIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def transform_bbox_to_crop(self,
                               box_in,
                               resize_factor,
                               device,
                               box_extract=None,
                               crop_type='template'):
        if crop_type == 'template':
            crop_sz = torch.Tensor(
                [self.cfg.TEST.TEMPLATE_SIZE, self.cfg.TEST.TEMPLATE_SIZE])
        elif crop_type == 'search':
            crop_sz = torch.Tensor(
                [self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(
            box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox

    def forward(self, input_data, device):
        # forward pass
        data = {}
        data['template_images'] = input_data[0]
        data['search_images'] = input_data[1]
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, input_data[2], input_data[3])

        return status

    def forward_pass(self, data):

        search_img = data['search_images']#[0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        template_list = data['template_images']

        box_mask_z = None
        ce_keep_rate = None


        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate)
                            

        return out_dict

    def compute_losses(self, pred_dict, template_gt_dict, search_gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = search_gt_dict.unsqueeze(0)  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_bbox, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        #gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        #                                                                                                   max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        gt_boxes_vec = (box_xywh_to_xyxy(gt_bbox).repeat((1, num_queries, 1)).view(-1, 4) /  self.cfg.DATA.SEARCH.SIZE).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.loss_objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.loss_objective['l1'](pred_boxes_vec, gt_boxes_vec.to(self.device))  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.loss_objective['focal'](pred_dict['score_map'], gt_gaussian_maps.to(self.device))
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"total_loss": loss,
                      "Loss/total": loss,
                      "Loss/giou": giou_loss,
                      "Loss/l1": l1_loss,
                      "Loss/location": location_loss,
                      "IoU": mean_iou}
            return loss, status
        else:
            return loss

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrain(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in load_dict.items():
            new_dict['net.' + k] = v
        self.load_state_dict(new_dict)
        
    @classmethod
    def _instantiate(cls, **kwargs):
        model_file = kwargs.get('am_model_name', ModelFile.TORCH_MODEL_BIN_FILE)
        ckpt_path = os.path.join(kwargs['model_dir'], model_file)
        #logger.info(f'loading model from {ckpt_path}')
        model_dir = kwargs.pop('model_dir')
        model = cls(**kwargs)
        return model
        #ckpt_path = os.path.join(model_dir, model_file)
        #load_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        #new_dict = {}
        #for k, v in load_dict.items():
        #    new_dict['net.' + k] = v
        #model.load_state_dict(new_dict)
        #return model
