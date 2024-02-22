# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List
from .utils import load_series_list, load_image, load_label, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, normalize_processed_image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class TrainDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        series_list_path (str): Path to the series list file.
        image_spacing (List[float]): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.
        use_bg (bool, optional): Flag indicating whether to use background or not.

    Attributes:
        labels (List): List of labels.
        dicom_paths (List): List of DICOM file paths.
        series_list_path (str): Path to the series list file.
        series_names (List): List of series names.
        image_spacing (ndarray): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, 
                 crop_fn=None, use_bg=False, min_d=0, norm_method='scale', window_levels: List[int] = [-300, -600], window_widths: List[int] = [1400, 1500]):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.window_levels = window_levels
        self.window_widths = window_widths
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        self.series_infos = load_series_list(series_list_path)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d)
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                continue
            
            self.dicom_paths.append(dicom_path)
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        images = []
        for window_level, window_width in zip(self.window_levels, self.window_widths):
            image = load_image(dicom_path, window_level, window_width) # z, y, x
            images.append(image)
        image = np.stack(images, axis=0)
        
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
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            random_samples.append(sample)

        return random_samples

class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale',
                 window_levels: List[int] = [-300, 1400], window_widths: List[int] = [-600, 1500]):
        self.series_list_path = series_list_path
        self.window_levels = window_levels
        self.window_widths = window_widths
        
        self.labels = []
        self.dicom_paths = []
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.series_infos = load_series_list(series_list_path)
        
        for folder, series_name in self.series_infos:
            dicom_path = gen_dicom_path(folder, series_name)
            self.dicom_paths.append(dicom_path)
        self.splitcomb = SplitComb
        if self.norm_method == 'none' and self.splitcomb.pad_value != 0:
            logger.warning('SplitComb pad_value should be 0 when norm_method is none, and it is set to 0 now')
            self.splitcomb.pad_value = 0.0
            
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        
        images = []
        for window_level, window_width in zip(self.window_levels, self.window_widths):
            images.append(load_image(dicom_path, window_level, window_width)) # z, y, x
        images = np.stack(images, axis=0)
        images = normalize_processed_image(images, self.norm_method)

        data = {}
        # split_images [N, 1, crop_z, crop_y, crop_x]
        
        all_split_images = []
        for i in range(images.shape[0]):
            image = images[i]
            split_images, nzhw = self.splitcomb.split(image) # N, 1, h, d ,w
            all_split_images.append(split_images)
        
        split_images = np.concatenate(all_split_images, axis=1)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data