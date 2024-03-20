# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List, Tuple
from .utils import load_series_list, load_image, load_label, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, normalize_processed_image, normalize_raw_image
from torch.utils.data import Dataset
import torchvision
import copy
logger = logging.getLogger(__name__)

from transform.ctr_transform import OffsetMinusCTR
from transform.feat_transform import FlipFeatTransform

class FlipTransform():
    def __init__(self, flip_depth=True, flip_height=True, flip_width=True):
        """
            flip_depth (bool) : random flip along depth axis or not, only used for 3D images
            flip_height (bool): random flip along height axis or not
            flip_width (bool) : random flip along width axis or not
        """
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        flip_axes = []
        
        if self.flip_width:
            flip_axes.append(-1)
        if self.flip_height:
            flip_axes.append(-2)
        if self.flip_depth:
            flip_axes.append(-3)

        if len(flip_axes) > 0:
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            image_t = np.flip(image, flip_axes).copy()
            sample['image'] = image_t

            offset = np.array([0, 0, 0]) # (z, y, x)
            for axis in flip_axes:
                offset[axis] = input_shape[axis] - 1
            sample['ctr_transform'].append(OffsetMinusCTR(offset))
            sample['feat_transform'].append(FlipFeatTransform(flip_axes))
        return sample

class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale'):
        self.series_list_path = series_list_path
        
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
            
        
        transforms = [[FlipTransform(flip_depth=False, flip_height=False, flip_width=True)],
                        [FlipTransform(flip_depth=False, flip_height=True, flip_width=False)],
                        [FlipTransform(flip_depth=True, flip_height=False, flip_width=False)]]
        
        self.transforms = []
        for i in range(len(transforms)):
            self.transforms.append(torchvision.transforms.Compose(transforms[i]))
            
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)

        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        split_images = np.squeeze(split_images, axis=1) # (N, crop_z, crop_y, crop_x)
        
        sample = {'image': split_images.copy(), 'ctr_transform': [], 'feat_transform': []}
        all_samples = [sample]
        for i, transform_fn in enumerate(self.transforms):
            all_samples.append(transform_fn(copy.deepcopy(sample)))
        
        # Stack the split images
        split_images = np.stack([s['image'] for s in all_samples], axis=1) # (N, num_aug, crop_z, crop_y, crop_x)
        split_images = np.expand_dims(split_images, axis=2) # (N, num_aug, 1, crop_z, crop_y, crop_x)
        split_images = np.ascontiguousarray(split_images)
        
        ctr_transforms = [s['ctr_transform'] for s in all_samples] # (num_aug,)
        feat_transforms = [s['feat_transform'] for s in all_samples] # (num_aug,)
        
                
        all_samples = {'split_images': split_images, 
                       'ctr_transform': ctr_transforms, 
                       'feat_transform': feat_transforms}
        all_samples['nzhw'] = nzhw
        all_samples['spacing'] = image_spacing
        all_samples['series_name'] = series_name
        all_samples['series_folder'] = series_folder
        
        return all_samples