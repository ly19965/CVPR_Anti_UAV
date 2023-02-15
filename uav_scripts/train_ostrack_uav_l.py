# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import json

# Step 1: 数据集准备
train_dataset = MsDataset.load('3rd_Anti-UAV', namespace='ly261666', split='train')

# Step 2: 相关参数设置
data_root_dir = '/home/ly261666/.cache/modelscope/hub/datasets/ly261666/3rd_Anti-UAV/master/data_files/extracted/7b8a88c5a8f38cced25ee619b96d924c0eea9f033bb57fc160ca2ec004d1ee6f' # 下载的数据集路径
data_root_dir = '/home/ly261666/workspace/maas/modelscope_project/Mass_env/or_data/train_data/new_data' # 下载的数据集路径
model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=model_id) # 下载modelscope上的模型，配置文件
cache_path = '/home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-uav-l'# 下载的modelscope模型路径
cfg_file = os.path.join(cache_path, 'configuration.json')
pretrain_model_id = 'damo/cv_vitb_video-single-object-tracking_ostrack-l'
video_single_object_tracking = pipeline(Tasks.video_single_object_tracking, model=pretrain_model_id) # 从modelscope上下载pretrain model
pretrain_model = '/home/ly261666/.cache/modelscope/hub/damo/cv_vitb_video-single-object-tracking_ostrack-l/pytorch_model.pt'

kwargs = dict(
    cfg_file=cfg_file,
    model=model_id, # 使用ostracker_uav模型 
    gpu_ids=[  # 指定训练使用的gpu
    0,1,2,3,4,5,6,7
    ],
    batch_size=64, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
    max_epochs=10, # 总的训练epochs
    load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练, 若为True，则加载modelscope上的模型finetune。
    pretrain_model=pretrain_model,
    base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
    train_image_dir=data_root_dir, # 训练图片路径
    val_image_dir=data_root_dir, # 测试图片路径
    workers_per_gpu=1,
    )


if __name__ == '__main__':
    # Step 3: 开启训练任务
    trainer = build_trainer(
                        name=Trainers.video_single_object_tracking, default_args=kwargs)
    trainer.train()
