import os.path
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from data.base_dataset import BaseDataset
from configurations import *
import copy
from albumentations.augmentations.functional import grid_distortion
import matplotlib.pyplot as plt


def get_2d_paths(dir):
    arrays = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.pkl'):
                path = os.path.join(root, fname)
                arrays.append(path)

    return arrays


def augmentations(data, ratio, opt):
    height, width = data['mask'].shape[:2]  # height/y for first axis, width/x for second axis
    for axis in [1, 0]:
        if random.random() < 0.3:
            for modality in MODALITIES + ['mask']:
                data[modality] = np.flip(data[modality], axis).copy()

    if random.random() < 0.5:
        height, width = width, height
        for modality in MODALITIES:
            data[modality] = np.transpose(data[modality], (1, 0, 2))
        data['mask'] = np.transpose(data['mask'], (1, 0))
    need_resize = False
    if random.random() < 0:
        crop_size = random.randint(int(opt.trainSize / 1.5), min(height, width))
        need_resize = True
    else:
        crop_size = opt.trainSize

    mask = data['mask']
    if np.sum(mask) == 0 or random.random() < 0.005:
        x_min = random.randint(0, width - crop_size)
        y_min = random.randint(0, height - crop_size)
    else:
        non_zero_yx = np.argwhere(mask)
        y, x = random.choice(non_zero_yx)
        x_min = x - random.randint(0, crop_size - 1)
        y_min = y - random.randint(0, crop_size - 1)
        x_min = np.clip(x_min, 0, width - crop_size)
        y_min = np.clip(y_min, 0, height - crop_size)

    for modality in MODALITIES + ['mask']:
        interpolation = cv2.INTER_LINEAR
        data[modality] = data[modality][y_min:y_min + crop_size, x_min:x_min + crop_size]
        if need_resize:
            data[modality] = cv2.resize(data[modality], (opt.trainSize, opt.trainSize), interpolation)

    data['mask'] = (data['mask'] > 0.5).astype(np.float32)
    return data


#=======================================================================================================================
# Code description
# This class is used for:
# (1) independent models (fixed combination of modalities)
# (2) ModDrop: regular modality dropout
# (3) ModDrop+: dynamic filter network ONLY (without intra-subject co-training)

"""
class MsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.use_modality_dropout = opt.use_modality_dropout
        self.all_paths = []
        for dataset_name in DATASETS:
            self.dir_data = os.path.join(opt.dataroot, dataset_name, opt.phase)
            self.all_paths += sorted(get_2d_paths(self.dir_data))

    def __getitem__(self, index):
        path_this_sample = self.all_paths[index]
        data_all_modalities = np.load(path_this_sample, allow_pickle=True)
        # store the available modalities in a list
        data_return = {'paths': path_this_sample}  
        available = []

        for modality in MODALITIES:
            if modality in data_all_modalities:
                available.append(modality)
                data_return[modality] = data_all_modalities[modality]
            else:
                data_return[modality] = np.zeros(data_all_modalities['t1'].shape)

        data_return['mask'] = data_all_modalities['mask'][:, :, 0]

        # augmentation
        data_return = augmentations(data_return, data_all_modalities['ratio'], self.opt)

        # preprocessing
        for modality in available:
            data_return[modality] = data_return[modality] / 2 - 1
        data_return['mask'] = data_return['mask'] * 2 - 1
        data_return['dc'] = data_all_modalities['dc']
        data_return['mc'] = data_all_modalities['mc']

        for modality in MODALITIES:
            data_return[modality] = transforms.ToTensor()(data_return[modality]).float()

        # ======== modality dropout ========
        if self.use_modality_dropout:
            mc_idx = list(np.where(data_return['mc'] == 1)[0])
            zero_idx = random.sample(mc_idx, random.randint(0, len(mc_idx)-1))
            for idx in zero_idx:
                # image set as zero tensor
                data_return[MODALITIES[idx]] = torch.zeros(data_return[MODALITIES[idx]].size())  
                data_return['mc'][idx] = 0  # modality code set as 0
                
        return data_return

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'MsDataset'
"""


#=======================================================================================================================
# Code description:
# This class is used for ModDrop++: (1) dynamic filter network and (2) intra-subject co-training. This dataloader
# returns both (1) full-modality data and (2) missing modality data (randomly dropped) from the same subject.


class MsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.use_modality_dropout = opt.use_modality_dropout
        self.all_paths = []
        for dataset_name in DATASETS:
            self.dir_data = os.path.join(opt.dataroot, dataset_name, opt.phase)
            self.all_paths += sorted(get_2d_paths(self.dir_data))

    def __getitem__(self, index):
        path_this_sample = self.all_paths[index]
        data_all_modalities = np.load(path_this_sample, allow_pickle=True)
        # store the available modalities in a list
        data_full = {'paths': path_this_sample}
        available = []

        for modality in MODALITIES:
            if modality in data_all_modalities:
                available.append(modality)
                data_full[modality] = data_all_modalities[modality]
            else:
                data_full[modality] = np.zeros(data_all_modalities['t1'].shape)
        data_full['mask'] = data_all_modalities['mask'][:, :, 0]

        # augmentation
        data_full = augmentations(data_full, data_all_modalities['ratio'], self.opt)

        # preprocessing
        for modality in available:
            data_full[modality] = data_full[modality] / 2 - 1

        data_full['mask'] = data_full['mask'] * 2 - 1
        data_full['dc'] = data_all_modalities['dc']
        data_full['mc'] = data_all_modalities['mc']

        for modality in MODALITIES:
            data_full[modality] = transforms.ToTensor()(data_full[modality]).float()
        data_miss = copy.deepcopy(data_full)
        # === modality dropout ===
        if self.use_modality_dropout:
            mc_idx = list(np.where(data_miss['mc'] == 1)[0])
            zero_idx = random.sample(mc_idx, random.randint(0, len(mc_idx) - 1))
            for idx in zero_idx:
                data_miss[MODALITIES[idx]] = torch.zeros(data_miss[MODALITIES[idx]].size())  # image set as zero tensor
                data_miss['mc'][idx] = 0  # modality code set as 0

        return data_full, data_miss

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'MsDataset'