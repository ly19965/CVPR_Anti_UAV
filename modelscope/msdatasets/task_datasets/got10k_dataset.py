# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

import os

import json
import numpy as np
import torch
import six
import json
import glob
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps


from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset


class Got10kDataset(object):
    def __init__(self, root_dir, subset='test', return_meta=False,
                 list_file=None, check_integrity=True, dataset_type='Got10k'):
        super(Got10kDataset, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'
        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False #if subset == 'test' else return_meta
        
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')
        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(root_dir, subset, s)
                         for s in self.seq_names]
        self.dataset_type = dataset_type
        if dataset_type == 'Got10k':
            self.anno_files = [os.path.join(d, 'groundtruth.txt')
                               for d in self.seq_dirs]
        elif dataset_type == '3rd_Anti-UAV':
            self.anno_files = [os.path.join(d, 'IR_label.json')
                               for d in self.seq_dirs]
    
    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        if self.dataset_type == 'Got10k':
            anno = np.loadtxt(self.anno_files[index], delimiter=',')
        elif self.dataset_type == '3rd_Anti-UAV':
            with open(self.anno_files[index], 'r') as f:
                gt_rect = json.load(f)['gt_rect']
                anno = np.zeros((len(gt_rect), 4))
                for idx in range(len(gt_rect)):
                    if len(gt_rect[idx]) == 4:
                        anno[idx] = gt_rect[idx]
                

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta


class RandomStretch(object):

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)



class Pairwise(Dataset):

    def __init__(self, seq_dataset, pairs_per_seq=10, max_dist=100, exemplar_sz=127, instance_sz=255, context=0.5):
        super(Pairwise, self).__init__()
        self.pairs_per_seq = pairs_per_seq
        self.max_dist = max_dist
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.seq_dataset = seq_dataset
        self.indices = np.random.permutation(len(seq_dataset))
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            #RandomStretch(max_stretch=0.05),
            #CenterCrop(self.instance_sz - 8),
            #RandomCrop(self.instance_sz - 2 * 8),
            #CenterCrop(self.exemplar_sz),
            ToTensor()])
        self.transform_x = Compose([
            #RandomStretch(max_stretch=0.05),
            #CenterCrop(self.instance_sz - 8),
            #RandomCrop(self.instance_sz - 2 * 8),
            ToTensor()])


    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[-1]
        index = self.indices[index % len(self.seq_dataset)]
        img_files, anno = self.seq_dataset[index]

        # remove too small objects
        valid = anno[:, 2:].prod(axis=1) >= 10
        img_files = np.array(img_files)[valid]
        anno = anno[valid, :]

        rand_z, rand_x = self._sample_pair(len(img_files))

        exemplar_image = Image.open(img_files[rand_z])
        instance_image = Image.open(img_files[rand_x])
        exemplar_image, exemplar_image_box = self._crop_and_resize(exemplar_image, anno[rand_z], self.exemplar_sz)
        instance_image, instance_image_box  = self._crop_and_resize(instance_image, anno[rand_x], self.instance_sz)
        exemplar_image = 255.0 * self.transform_z(exemplar_image)
        instance_image = 255.0 * self.transform_x(instance_image)

        return exemplar_image, instance_image, exemplar_image_box, instance_image_box

    def __len__(self):
        return self.pairs_per_seq * len(self.seq_dataset)

    def _sample_pair(self, n):
        assert n > 0
        if n == 1:
            return 0, 0
        elif n == 2:
            return 0, 1
        else:
            max_dist = min(n - 1, self.max_dist)
            rand_dist = np.random.choice(max_dist) + 1
            rand_z = np.random.choice(n - rand_dist)
            rand_x = rand_z + rand_dist

        return rand_z, rand_x

    def _crop_and_resize(self, image, box, target_size):
        # convert box to 0-indexed and center based
        or_box = box.copy()
        box = np.array([
            box[0] - 1 + (box[2] - 1) / 2,
            box[1] - 1 + (box[3] - 1) / 2,
            box[2], box[3]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]


        # exemplar and search sizes
        import random
        fix_th = 0.2
        context = self.context - fix_th + random.random() * fix_th * 2

        context = context * np.sum(target_sz)
        z_sz = np.sqrt(np.prod(target_sz + context))
        x_sz = z_sz * self.instance_sz / self.exemplar_sz

        # convert box to corners (0-indexed)
        size = round(x_sz)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.size))
        npad = max(0, int(pads.max()))
        if npad > 0:
            avg_color = ImageStat.Stat(image).mean
            # PIL doesn't support float RGB image
            avg_color = tuple(int(round(c)) for c in avg_color)
            image = ImageOps.expand(image, border=npad, fill=avg_color)

        # crop image patch
        corners = tuple((corners + npad).astype(int))
        patch = image.crop(corners)

        resize_ratio =  target_size / patch.size[0]
        x0 = max(or_box[0] - corners[0] + npad, 0) * resize_ratio
        y0 = max(or_box[1] - corners[1] + npad, 0) * resize_ratio
        ret_box = [x0 , y0, target_sz[0] * resize_ratio, target_sz[1] * resize_ratio]
        

        # resize to target_size
        out_size = (target_size, target_size)
        patch = patch.resize(out_size, Image.BILINEAR)

        #gt = patch.crop(ret_box)
        #import time
        #gt.save('./tmp_img/{}_tmp_1_{}.png'.format(0, time.time()))

        return patch, torch.Tensor(ret_box)
