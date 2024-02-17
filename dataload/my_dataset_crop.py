# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import json
import numpy as np
from typing import List
from .utils import load_series_list, load_image
from torch.utils.data import Dataset

BBOXES = 'bboxes'
USE_BG = False

class TrainDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        roots (list): list of dirs of the dataset
        transform_post: transform object after cropping
        crop_fn: cropping function
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.series_names = []
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count_crop.json')
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            
            with open(label_path, 'r') as f:
                info = json.load(f)
                
            bboxes = info[BBOXES]
            bboxes = np.array(bboxes)

            if len(bboxes) == 0:
                if not USE_BG:
                    continue
                self.dicom_paths.append(dicom_path)
                self.series_names.append(series_name)
                label = {'all_loc': np.zeros((0, 3), dtype=np.float32),
                        'all_rad': np.zeros((0, 3), dtype=np.float32),
                        'all_cls': np.zeros((0,), dtype=np.int32)}
            else:
                self.dicom_paths.append(dicom_path)
                self.series_names.append(series_name)
                # calculate center of bboxes
                all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
                all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

                all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
                all_rad = all_rad[:, [2, 0, 1]] # (z, y, x)
                all_rad = all_rad * self.image_spacing # (z, y, x)
                all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)
                
                label = {'all_loc': all_loc, 
                        'all_rad': all_rad,
                        'all_cls': all_cls}
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        
        data = {}
        data['image'] = image
        data['all_loc'] = label['all_loc'] # z, y, x
        data['all_rad'] = label['all_rad'] # d, h, w
        data['all_cls'] = label['all_cls']
        data['file_name'] = series_name
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            if self.transform_post:
                sample['ctr_transform'] = []
                sample = self.transform_post(sample)
            sample['image'] = (sample['image'] * 2.0 - 1.0) # normalized to -1 ~ 1
            random_samples.append(sample)

        return random_samples

class DetDataset(Dataset):
    """Detection dataset for inference
    """

    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.series_names = []
        self.series_folders = []
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        
        series_infos = load_series_list(series_list_path)
        for folder, series_name in series_infos:
            dicom_path = os.path.join(folder, 'npy', f'{series_name}_crop.npy')
            self.dicom_paths.append(dicom_path)
            self.series_names.append(series_name)
            self.series_folders.append(folder)
        self.splitcomb = SplitComb
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_name = self.series_names[idx]
        series_folder = self.series_folders[idx]
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x

        data = {}
        # convert to -1 ~ 1  note ste pad_value to -1 for SplitComb
        image = image * 2.0 - 1.0
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data