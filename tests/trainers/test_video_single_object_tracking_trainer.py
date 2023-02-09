# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import unittest
from functools import partial

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment
from modelscope.utils.constant import DownloadMode
from modelscope.utils.hub import read_config
from modelscope.utils.test_utils import test_level

SEGMENT_LENGTH_TEST = 640


class TestVSOTTrainer(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.model_id = 'damo/cv_alex_video-single-object-tracking_siamfc'

        self.train_root = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/got10k'
        self.cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_alex_video-single-object-tracking_siamfc'


    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer(self):
        kwargs = dict(
            cfg_file='/home/ly261666/workspace/maas/modelscope_project/cv_alex_video-single-object-tracking_siamfc/configuration.json',
            model=self.model_id, # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
            0,1
            ],
            batch_size=16, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=3, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            #cache_path=self.cache_path,
            #load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练
            base_lr_per_img=0.01, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=self.train_root, # 训练图片路径
            val_image_dir=self.train_root, # 测试图片路径
            )

        trainer = build_trainer(
            Trainers.video_single_object_tracking, default_args=kwargs)
        trainer.train()

if __name__ == '__main__':
    unittest.main()
