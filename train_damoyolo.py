import os.path as osp
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load('person_detection_for_train', namespace="modelscope", split='train')
val_dataset = MsDataset.load('person_detection_for_train', namespace="modelscope", split='validation')

# Step 2: 相关参数设置
train_root_dir = train_dataset.config_kwargs['split_config']['train']
val_root_dir = val_dataset.config_kwargs['split_config']['validation']
train_img_dir = osp.join(train_root_dir, 'images')
val_img_dir = osp.join(val_root_dir, 'images')
train_anno_path = osp.join(train_root_dir, 'train.json')
val_anno_path = osp.join(val_root_dir, 'val.json')
kwargs = dict(
            model='damo/cv_tinynas_human-detection_damoyolo', # 使用DAMO-YOLO-S模型 
            gpu_ids=[  # 指定训练使用的gpu
            0
            ],
            batch_size=2, # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
            max_epochs=3, # 总的训练epochs
            num_classes=1, # 自定义数据中的类别数
            load_pretrain=True, # 是否载入预训练模型，若为False，则为从头重新训练
            base_lr_per_img=0.001, # 每张图片的学习率，lr=base_lr_per_img*batch_size
            train_image_dir=train_img_dir, # 训练图片路径
            val_image_dir=val_img_dir, # 测试图片路径
            train_ann=train_anno_path, # 训练标注文件路径
            val_ann=val_anno_path, # 测试标注文件路径
            )

# Step 3: 开启训练任务
trainer = build_trainer(
                    name=Trainers.tinynas_damoyolo, default_args=kwargs)
trainer.train()
