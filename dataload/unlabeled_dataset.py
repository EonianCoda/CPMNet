# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import json
import copy
import logging
import numpy as np
from typing import List
from .utils import load_series_list, load_image, load_label, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, normalize_processed_image, normalize_raw_image, DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class UnLabeledDataset(Dataset):
    def __init__(self, series_list_path: str, image_spacing: List[float], weak_aug=None, strong_aug = None, crop_fn=None, 
                 use_bg=False, min_d=0, min_size: int = 0, norm_method='scale', mmap_mode=None, labels = None):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.min_size = int(min_size)
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        self.series_infos = load_series_list(series_list_path)
        self.labels = labels

        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode

    def update_labels(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.series_infos)

    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        
        samples = {}
        samples['image'] = image
        samples['all_loc'] = label['all_loc'] # z, y, x
        samples['all_rad'] = label['all_rad'] # d, h, w
        samples['all_cls'] = label['all_cls']
        samples['file_name'] = series_name
        samples = self.crop_fn(samples, image_spacing)
        random_samples = []

        weak_samples = []
        strong_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'])
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            sample['ctr_transform'] = []
    
            weak_samples.append(self.weak_aug(copy.deepcopy(sample)))
            strong_samples.append(self.strong_aug(sample))
        
        random_samples = dict()
        random_samples['weak'] = weak_samples
        random_samples['strong'] = strong_samples
        
        return random_samples