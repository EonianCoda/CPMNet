# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List
from .utils import load_series_list, load_image, load_label, load_lobe, ALL_RAD, ALL_LOC, ALL_CLS, ALL_PROB, gen_dicom_path, gen_label_path, \
                    gen_lobe_path, normalize_processed_image, normalize_raw_image
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
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, 
                 use_bg=False, min_d=0, min_size: int = 0, norm_method='scale', mmap_mode=None):
        self.labels = []
        self.dicom_paths = []
        self.series_names = []
        self.series_folders = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.min_size = int(min_size)
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        if self.min_size != 0:
            logger.info('When training, ignore nodules with size less than {}'.format(min_size))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        if use_bg:
            logger.info('Using background(healthy lung) as training data')
        
        self.series_infos = load_series_list(series_list_path)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d, min_size)
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                continue
            
            label[ALL_PROB] = np.zeros_like(label[ALL_CLS], dtype=np.float32)
            
            self.series_folders.append(folder)
            self.series_names.append(series_name)
            self.dicom_paths.append(dicom_path)
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode

    def __len__(self):
        return len(self.labels)
    
    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def update_nodule_probs(self, series_names, nodule_indices, nodule_probs):
        update_dict = {}
        
        for series_name, nodule_index, nodule_prob in zip(series_names, nodule_indices, nodule_probs):
            if series_name not in update_dict:
                update_dict[series_name] = dict()
            
            if nodule_index not in update_dict[series_name]:
                update_dict[series_name][nodule_index] = nodule_prob
                continue
            
            update_dict[series_name][nodule_index] = max(update_dict[series_name][nodule_index], nodule_prob)    

        series_names = set(series_names)
        for series_name in series_names:
            label_i = self.series_names.index(series_name)
            # update nodule prob
            for nodule_index, nodule_prob in update_dict[series_name].items():
                self.labels[label_i][ALL_PROB][nodule_index] = nodule_prob
            # penalize not appearing nodules
            non_appear_nodule_indices = [nodule_index for nodule_index in range(self.labels[label_i][ALL_PROB].shape[0]) if nodule_index not in update_dict[series_name]]
            self.labels[label_i][ALL_PROB][non_appear_nodule_indices] *= 0.9
            self.labels[label_i][ALL_PROB] = np.clip(self.labels[label_i][ALL_PROB], 0, 1)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_folders[idx]
        series_name = self.series_names[idx]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        
        samples = {}
        samples['image'] = image
        samples['all_loc'] = label['all_loc'] # z, y, x
        samples['all_rad'] = label['all_rad'] # d, h, w
        samples['all_cls'] = label['all_cls']
        samples['all_prob'] = label['all_prob']
        samples['series_name'] = series_name
        samples = self.crop_fn(samples, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'])
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            if self.transform_post:
                sample['ctr_transform'] = []
                sample['feat_transform'] = []
                sample = self.transform_post(sample)
            random_samples.append(sample)

        return random_samples

class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale', apply_lobe=False, out_stride = 4):
        self.series_list_path = series_list_path
        self.apply_lobe = apply_lobe
        self.norm_method = norm_method
        self.out_stride = out_stride
        
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.series_infos = load_series_list(series_list_path)

        self.dicom_paths = []
        self.lobe_paths = []
        for folder, series_name in self.series_infos:
            self.dicom_paths.append(gen_dicom_path(folder, series_name))
            self.lobe_paths.append(gen_lobe_path(folder, series_name))
        self.splitcomb = SplitComb
        
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)

        data = {}
        if self.apply_lobe: # load lobe mask
            lobe_path = self.lobe_paths[idx]
            lobe_mask = load_lobe(lobe_path)
            split_images, split_lobes, nzhw, image_shape = self.splitcomb.split(image, lobe_mask, self.out_stride) # split_images [N, 1, crop_z, crop_y, crop_x]
            data['split_lobes'] = np.ascontiguousarray(split_lobes)
        else: 
            split_images, nzhw, image_shape = self.splitcomb.split(image) # split_images [N, 1, crop_z, crop_y, crop_x]
        
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['image_shape'] = image_shape
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data