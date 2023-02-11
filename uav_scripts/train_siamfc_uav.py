# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import os
import json

# Step 1: 数据集准备
train_dataset = MsDataset.load('Got-10k', namespace='ly261666', split='train')

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/new_data' # 下载的数据集路径
model_id = 'damo/cv_alex_video-single-object-tracking_siamfc-uav'
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_alex_video-single-object-tracking_siamfc-uav'# 下载的modelscope模型路径
pretrain_model = '/home/ly261666/.cache/modelscope/hub/damo/cv_alex_video-single-object-tracking_siamfc/pytorch_model.pt'# 下载的modelscope预训练模型路径
cfg_file = os.path.join(cache_path, 'configuration.json')

kwargs = dict(
    cfg_file=cfg_file,
    model=model_id, # 使用siamfc_uav模型 
    gpu_ids=[  # 指定训练使用的gpu
    0,1,2,3,4,5,6,7
    ],
    batch_size=16, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
    max_epochs=300, # 总的训练epochs
    load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练, 若为True，则加载modelscope上的模型finetune
    pretrain_model=pretrain_model, # 加载pretrain_model继续训练 
    base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
    train_image_dir=data_root_dir, # 训练图片路径
    val_image_dir=data_root_dir, # 测试图片路径
    )


if __name__ == '__main__':
    # Step 3: 开启训练任务
    trainer = build_trainer(
                        name=Trainers.video_single_object_tracking, default_args=kwargs)
    trainer.train()
