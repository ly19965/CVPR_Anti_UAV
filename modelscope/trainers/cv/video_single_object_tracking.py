# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import datetime
import math
import os
import os.path as osp
import time
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from easydict import EasyDict as easydict
from torch.nn.parallel import DistributedDataParallel as DDP

from modelscope.metainfo import Trainers
from modelscope.models.cv.tinynas_detection.damo.apis.detector_evaluater import \
    Evaluater
from modelscope.models.cv.tinynas_detection.damo.apis.detector_inference import \
    inference
from modelscope.models.cv.tinynas_detection.damo.base_models.losses.distill_loss import \
    FeatureLoss
from modelscope.models.cv.tinynas_detection.damo.detectors.detector import (
    build_ddp_model, build_local_model)
from modelscope.models.cv.tinynas_detection.damo.utils import (
    cosine_scheduler, ema_model)
from modelscope.msdatasets.task_datasets.damoyolo import build_sot_dataloader
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger
from modelscope.utils.metric import MeterBuffer
from modelscope.utils.torch_utils import get_rank, synchronize
from modelscope.msdatasets.task_datasets import Got10kDataset, Pairwise


@TRAINERS.register_module(module_name=Trainers.video_single_object_tracking)
class ImageDetectionDamoyoloTrainer(BaseTrainer):

    def __init__(self,
                 model: str = None,
                 cfg_file: str = None,
                 load_pretrain: bool = True,
                 cache_path: str = None,
                 *args,
                 **kwargs):
        """ 

        Args:
            model: Model id of modelscope models.
            cfg_file: Path to configuration file.
            load_pretrain: Whether load pretrain model for finetune.
                if False, means training from scratch.
            cache_path: cache path of model files.
        """
        if model is not None:
            self.cache_path = self.get_or_download_model_dir(model)
            if cfg_file is None:
                self.cfg_file = os.path.join(self.cache_path,
                                             ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None and cache_path is not None, \
                'cfg_file and cache_path is needed, if model is not provided'

        if cfg_file is not None:
            self.cfg_file = cfg_file
            if cache_path is not None:
                self.cache_path = cache_path
        super().__init__(self.cfg_file)
        cfg = self.cfg
        cfg.model.model_id = model
        if load_pretrain:
            if 'pretrain_model' in kwargs:
                cfg.train.finetune_path = kwargs['pretrain_model']
            else:
                cfg.train.finetune_path = os.path.join(self.cache_path,
                                                       self.cfg.model.weights)

        if 'framework' in self.cfg:
            cfg = self._config_transform(cfg)

        if 'gpu_ids' in kwargs:
            cfg.train.gpu_ids = kwargs['gpu_ids']
        if 'batch_size' in kwargs:
            cfg.train.batch_size = kwargs['batch_size']
        if 'max_epochs' in kwargs:
            cfg.train.total_epochs = kwargs['max_epochs']
        if 'train_image_dir' in kwargs:
            cfg.dataset.train_image_dir = kwargs['train_image_dir']
        if 'val_image_dir' in kwargs:
            cfg.dataset.val_image_dir = kwargs['val_image_dir']
        if 'base_lr_per_img' in kwargs:
            cfg.train.base_lr_per_img = kwargs['base_lr_per_img']
        if 'workers_per_gpu' in kwargs:
            cfg.train.dataloader.workers_per_gpu = kwargs['workers_per_gpu']

        self.gpu_ids = cfg.train.gpu_ids
        self.world_size = len(self.gpu_ids)

        self.cfg = cfg

    def _train(self, local_rank, world_size, cfg):
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            'nccl',
            init_method='tcp://127.0.0.1:12344',
            rank=local_rank,
            world_size=world_size)
        trainer = SotTrainer(cfg, None, None)
        trainer.train(local_rank)

    def train(self):
        if len(self.cfg.train.gpu_ids) > 1:
            mp.spawn(
                self._train,
                nprocs=self.world_size,
                args=(self.world_size, self.cfg),
                join=True)
        else:
            trainer = SotTrainer(self.cfg, None, None)
            trainer.train(local_rank=0)

    def evaluate(self,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        if checkpoint_path is not None:
            self.cfg.test.checkpoint_path = checkpoint_path
        evaluater = Evaluater(self.cfg)
        evaluater.evaluate()

    def _config_transform(self, config):
        new_config = easydict({})
        new_config.miscs = config.train.miscs
        new_config.miscs.num_workers = config.train.dataloader.workers_per_gpu
        new_config.miscs.output_dir = config.train.work_dir
        new_config.model = config.model
        new_config.dataset = config.dataset
        new_config.train = config.train


        new_config.train.warmup_start_lr = config.train.lr_scheduler.warmup_start_lr
        new_config.train.min_lr_ratio = config.train.lr_scheduler.min_lr_ratio
        new_config.train.warmup_epochs = config.train.lr_scheduler.warmup_epochs

        new_config.train.batch_size = len(
            config.train.gpu_ids) * config.train.dataloader.batch_size_per_gpu
        new_config.train.base_lr_per_img = config.train.optimizer.lr / new_config.train.batch_size
        new_config.train.momentum = config.train.optimizer.momentum
        new_config.train.weight_decay = config.train.optimizer.weight_decay
        new_config.train.total_epochs = config.train.max_epochs

        new_config.task = config.task

        del new_config['train']['miscs']
        del new_config['train']['lr_scheduler']
        del new_config['train']['optimizer']
        del new_config['train']['dataloader']

        return new_config


class SotTrainer:

    def __init__(self, cfg, args, tea_cfg=None):
        self.cfg = cfg
        self.tea_cfg = tea_cfg
        self.args = args
        self.output_dir = cfg.miscs.output_dir
        self.exp_name = cfg.miscs.exp_name
        self.device = 'cuda'

        if len(self.cfg.train.gpu_ids) > 1:
            self.distributed = True
        else:
            self.distributed = False
        # metric record
        self.meter = MeterBuffer(window_size=cfg.miscs.print_interval_iters)
        self.file_name = os.path.join(cfg.miscs.output_dir, cfg.miscs.exp_name)

        # setup logger
        if get_rank() == 0:
            os.makedirs(self.file_name, exist_ok=True)
        self.logger = get_logger(os.path.join(self.file_name, 'train_log.txt'))

        # logger
        self.logger.info('args info: {}'.format(self.args))
        self.logger.info('cfg value:\n{}'.format(self.cfg))

    def get_data_loader(self, cfg, distributed=False):

        if hasattr(cfg.dataset, 'dataset_type'):
            seq_train_dataset = Got10kDataset(
                root_dir=cfg.dataset.train_image_dir,
                subset='train',
                dataset_type=cfg.dataset.dataset_type)
        else:
            seq_train_dataset = Got10kDataset(
                root_dir=cfg.dataset.train_image_dir,
                subset='train')
        pair_train_dataset = Pairwise(seq_train_dataset, exemplar_sz=cfg.dataset.exemplar_size, instance_sz=cfg.dataset.instance_sz)

        train_loader = build_sot_dataloader(
            pair_train_dataset,
            batch_size=cfg.train.batch_size,
            start_epoch=self.start_epoch,
            total_epochs=cfg.train.total_epochs,
            num_workers=cfg.miscs.num_workers,
            is_train=True,
            size_div=32,
            distributed=distributed)

        iters_per_epoch = math.ceil(
            len(pair_train_dataset)
            / cfg.train.batch_size)  

        return train_loader, iters_per_epoch

    def setup_iters(self, iters_per_epoch, start_epoch, total_epochs,
                    warmup_epochs, no_aug_epochs, eval_interval_epochs,
                    ckpt_interval_epochs, print_interval_iters):
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.start_iter = start_epoch * iters_per_epoch
        self.total_iters = total_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.no_aug_iters = no_aug_epochs * iters_per_epoch
        self.no_aug = self.start_iter >= self.total_iters - self.no_aug_iters
        self.eval_interval_iters = eval_interval_epochs * iters_per_epoch
        self.ckpt_interval_iters = ckpt_interval_epochs * iters_per_epoch
        self.print_interval_iters = print_interval_iters

    def build_optimizer(self, momentum, weight_decay):

        bn_group, weight_group, bias_group = [], [], []

        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                bias_group.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
                bn_group.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                weight_group.append(v.weight)

        optimizer = torch.optim.SGD(
            bn_group,
            lr=1e-3,  # only used to init optimizer,
            # and will be overwrited
            momentum=momentum,
            nesterov=True)
        optimizer.add_param_group({
            'params': weight_group,
            'weight_decay': weight_decay
        })
        optimizer.add_param_group({'params': bias_group})
        self.optimizer = optimizer

        return self.optimizer

    def train(self, local_rank):
        # build model
        from modelscope.models import Model
        self.model = Model.from_pretrained(self.cfg.model.model_id, cfg_dict=self.cfg).to(self.device)

        self.optimizer = self.build_optimizer(self.cfg.train.momentum,
                                              self.cfg.train.weight_decay)
        # resume model
        if self.cfg.train.finetune_path is not None:
            self.logger.info(f'finetune from {self.cfg.train.finetune_path}')
            self.model.load_pretrain(self.cfg.train.finetune_path)
            self.epoch = 0
            self.start_epoch = 0
        elif self.cfg.train.resume_path is not None:
            resume_epoch = self.resume_model(
                self.cfg.train.resume_path, need_optimizer=True)
            self.epoch = resume_epoch
            self.start_epoch = resume_epoch
            self.logger.info('Resume Training from Epoch: {}'.format(
                self.epoch))
        else:
            self.epoch = 0
            self.start_epoch = 0
            self.model._initialize_weights()
            self.logger.info('Start Training...')

        # dataloader
        self.train_loader, iters = self.get_data_loader(
            self.cfg, self.distributed)

        # setup iters according epochs and iters_per_epoch
        self.setup_iters(iters, self.start_epoch, self.cfg.train.total_epochs,
                         self.cfg.train.warmup_epochs,
                         self.cfg.train.no_aug_epochs,
                         self.cfg.miscs.eval_interval_epochs,
                         self.cfg.miscs.ckpt_interval_epochs,
                         self.cfg.miscs.print_interval_iters)

        self.lr_scheduler = cosine_scheduler(
            self.cfg.train.base_lr_per_img, self.cfg.train.batch_size,
            self.cfg.train.min_lr_ratio, self.total_iters, self.no_aug_iters,
            self.warmup_iters, self.cfg.train.warmup_start_lr)


        # distributed model init
        if self.distributed:
            self.model = build_ddp_model(self.model, local_rank)
        else:
            self.model = self.model.to('cuda')

        self.logger.info('Training start...')

        # ----------- start training ------------------------- #
        self.model.train()
        iter_start_time = time.time()
        iter_end_time = time.time()
        for data_iter, (inps, inps_pair, inps_gt, inps_pair_gt) in enumerate(self.train_loader):
            cur_iter = self.start_iter + data_iter

            lr = self.lr_scheduler.get_lr(cur_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            inps = inps.to(self.device)  
            inps_pair = inps_pair.to(self.device) 

            model_start_time = time.time()


            outputs = self.model([inps, inps_pair, inps_gt, inps_pair_gt], device=self.device)
            loss = outputs['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            iter_start_time = iter_end_time
            iter_end_time = time.time()

            outputs_array = {_name: _v.item() for _name, _v in outputs.items()}
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                model_time=iter_end_time - model_start_time,
                lr=lr,
                **outputs_array,
            )


            # log needed information
            if (cur_iter + 1) % self.print_interval_iters == 0 and local_rank == 0:
                left_iters = self.total_iters - (cur_iter + 1)
                eta_seconds = self.meter['iter_time'].global_avg * left_iters
                eta_str = 'ETA: {}'.format(
                    datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = 'epoch: {}/{}, iter: {}/{}'.format(
                    self.epoch + 1, self.total_epochs,
                    (cur_iter + 1) % self.iters_per_epoch,
                    self.iters_per_epoch)
                loss_meter = self.meter.get_filtered_meter('loss')
                loss_str = ', '.join([
                    '{}: {:.4f}'.format(k, v.avg)
                    for k, v in loss_meter.items()
                ])

                time_meter = self.meter.get_filtered_meter('time')
                time_str = ', '.join([
                    '{}: {:.3f}s'.format(k, v.avg)
                    for k, v in time_meter.items()
                ])

                self.logger.info('{}, {}, {}, lr: {:.3e}'.format(
                    progress_str,
                    time_str,
                    loss_str,
                    self.meter['lr'].latest,
                    ) +  (', giou_loss: {:.4f}, l1_loss: {:.4f}, location_loss: {:.4f}'.format(outputs_array['Loss/giou'], outputs_array['Loss/l1'], outputs_array['Loss/location'])) +
                (', size: ({:d}, {:d}), {}'.format(
                    inps.shape[2], inps.shape[3], eta_str)))
                self.meter.clear_meters()

            if (cur_iter + 1) % self.ckpt_interval_iters == 0:
                self.save_ckpt(
                    'epoch_%d_ckpt.pth' % (self.epoch + 1),
                    local_rank=local_rank)

            if (cur_iter + 1) % self.eval_interval_iters == 0:
                time.sleep(0.003)
                self.model.train()
            synchronize()

            if (cur_iter + 1) % self.iters_per_epoch == 0:
                self.epoch = self.epoch + 1

        self.save_ckpt(ckpt_name='latest_ckpt.pth', local_rank=local_rank)

    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):
        if local_rank == 0:
            if isinstance(self.model, DDP):
                save_model = self.model.module.net
            else:
                save_model = self.model.net
            ckpt_name = os.path.join(self.file_name, ckpt_name)
            self.logger.info('Save weights to {}'.format(ckpt_name))
            meta = {'epoch': self.epoch + 1}
            save_checkpoint(
                model=save_model,
                filename=ckpt_name,
                optimizer=self.optimizer,
                meta=meta,
                with_meta=True)

    def resume_model(self, resume_path, load_optimizer=False):
        ckpt_file_path = resume_path
        ckpt = torch.load(ckpt_file_path, map_location=self.device)
        if 'state_dict' in ckpt:
            self.model.load_state_dict(ckpt['state_dict'])
        elif 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'])

        if load_optimizer:
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.distill:
                if 'meta' in ckpt:
                    self.feature_loss.load_state_dict(
                        ckpt['meta']['feature_loss'])
                elif 'feature_loss' in ckpt:
                    self.feature_loss.load_state_dict(ckpt['feature_loss'])
        if 'meta' in ckpt:
            resume_epoch = ckpt['meta']['epoch']
        elif 'epoch' in ckpt:
            resume_epoch = ckpt['epoch']
        return resume_epoch

    def evaluate(self, local_rank, val_ann):
        if self.ema_model is not None:
            evalmodel = self.ema_model.model
        else:
            evalmodel = self.model
            if isinstance(evalmodel, DDP):
                evalmodel = evalmodel.module

        output_folder = os.path.join(self.output_dir, self.exp_name,
                                     'inference')
        if local_rank == 0:
            os.makedirs(output_folder, exist_ok=True)

        for data_loader_val in self.val_loader:
            inference(
                evalmodel,
                data_loader_val,
                device=self.device,
                output_folder=output_folder,
            )
