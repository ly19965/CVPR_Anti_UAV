# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from .siamfc import TrackerSiamFC
from .detection_siamfc import TrackerSiamFC2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_multi_object_tracking, module_name=Pipelines.uav_detection_23)
class UavTrakckerPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a anti-uav cvpr23 baseline model pipeline 
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        # you can substitute net_path with your trained model path
        net_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        self.tracker = TrackerSiamFC(net_path=net_path)
        self.tracker_2 = TrackerSiamFC2(net_path=net_path)
        logger.info('tracker model loaded!')

    def preprocess(self, input) -> Input:
        self.video_path = input[0]
        return input

    def forward(self, input: Input) -> Dict[str, Any]:
        return input


    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
