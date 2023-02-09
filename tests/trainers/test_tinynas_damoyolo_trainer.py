# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import shutil
import tempfile
import unittest

import torch

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile
from modelscope.utils.test_utils import DistributedTestCase, test_level


def _setup():
    model_id = 'damo/cv_tinynas_object-detection_damoyolo'
    cache_path = snapshot_download(model_id)
    return cache_path


class TestTinynasDamoyoloTrainerSingleGPU(unittest.TestCase):

    def setUp(self):
        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo'
        self.cache_path = _setup()

    #@unittest.skip('multiGPU test is varified offline')
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_from_scratch_multiGPU(self):
        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[
                0,
            ],
            batch_size=16,
            max_epochs=3,
            num_classes=1,
            cache_path=self.cache_path,
            train_image_dir='/home/ly261666/datasets/coco/coco_2017/train2017',
            val_image_dir='/home/ly261666/datasets/coco/coco_2017/train2017',
            train_ann=
            '/home/ly261666/datasets/coco/coco_2017/annotations/instances_train2017.json',
            val_ann=
            '/home/ly261666/datasets/coco/coco_2017/annotations/instances_train2017.json')
        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()



if __name__ == '__main__':
    unittest.main()
