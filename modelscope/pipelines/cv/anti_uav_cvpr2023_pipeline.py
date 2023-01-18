# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.uav_detection import TrackerSiamFC
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_multi_object_tracking, module_name=Pipelines.uav_detection)
class UavTrakckerPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a anti-uav cvpr23 baseline model pipeline 
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        net_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        self.tracker = TrackerSiamFC(net_path=net_path)
        logger.info('tracker model loaded!')

    def preprocess(self, input) -> Input:
        self.video_path = input[0]
        return input

    def forward(self, input: Input) -> Dict[str, Any]:
        dataloader = LoadVideo(input, self.opt.img_size)
        self.tracker.set_buffer_len(dataloader.frame_rate)

        results = []
        output_timestamps = []
        frame_id = 0
        for i, (path, img, img0) in enumerate(dataloader):
            output_timestamps.append(
                timestamp_format(seconds=frame_id / dataloader.frame_rate))
            blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = self.tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                    online_tlwhs.append([
                        tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    ])
                    online_ids.append(tid)
                results.append([
                    frame_id + 1, tid, tlwh[0], tlwh[1], tlwh[0] + tlwh[2],
                    tlwh[1] + tlwh[3]
                ])
            frame_id += 1

        return {
            OutputKeys.BOXES: results,
            OutputKeys.TIMESTAMPS: output_timestamps
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
