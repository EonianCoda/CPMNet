# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List
from .utils import load_series_list, load_image, load_label, load_lobe, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, \
                    gen_lobe_path, normalize_processed_image, normalize_raw_image
from torch.utils.data import Dataset
import itertools
logger = logging.getLogger(__name__)

def get_mixup_weight() -> float:
    resolution = 0.01
    values = np.arange(0.8, 1.01, resolution)
    probs = np.exp(-(values - 0.9)**2 / (2 * 0.01**2))
    probs /= probs.sum()
    random_numbers = np.random.choice(values, size=1, p=probs)
    return random_numbers[0]

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
        self.all_labels = []
        self.all_dicom_paths = []
        self.all_series_names = []
        self.all_series_folders = []
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
            
            self.all_series_folders.append(folder)
            self.all_series_names.append(series_name)
            self.all_dicom_paths.append(dicom_path)
            self.all_labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode

        self.shuffle_group()

    def __len__(self):
        return len(self.group_indices)
    
    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def shuffle_group(self):
        # Every 2 sample construct a group
        indices = np.arange(len(self.all_labels))
        np.random.shuffle(indices)
        self.group_indices = []
        for i in range(0, len(indices), 2):
            idx1 = indices[i]
            idx2 = indices[i+1]
            self.group_indices.append([idx1, idx2])
    
    def __getitem__(self, idx):
        idx1, idx2 = self.group_indices[idx]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        
        dicom_path1 = self.all_dicom_paths[idx1]
        series_folder1 = self.series_infos[idx1][0]
        series_name1 = self.series_infos[idx1][1]
        label1 = self.all_labels[idx1]
        image1 = self.load_image(dicom_path1) # z, y, x
        
        dicom_path2 = self.all_dicom_paths[idx2]
        series_folder2 = self.series_infos[idx2][0]
        series_name2 = self.series_infos[idx2][1]
        label2 = self.all_labels[idx2]
        image2 = self.load_image(dicom_path2) # z, y, x
        
        samples1 = {}
        samples1['image'] = image1
        samples1['all_loc'] = label1['all_loc'] # z, y, x
        samples1['all_rad'] = label1['all_rad'] # d, h, w
        samples1['all_cls'] = label1['all_cls']
        samples1 = self.crop_fn(samples1, image_spacing)
        
        samples2 = {}
        samples2['image'] = image2
        samples2['all_loc'] = label2['all_loc'] # z, y, x
        samples2['all_rad'] = label2['all_rad'] # d, h, w
        samples2['all_cls'] = label2['all_cls']
        samples2 = self.crop_fn(samples2, image_spacing)
        
        # Mixup nodule between two samples
        fg_samples1 = []
        fg_samples2 = []
        bg_samples = []
        for i in range(len(samples1)):
            sample = samples1[i]
            if len(sample['all_cls']) == 0:
                bg_samples.append(sample)
            else:
                fg_samples1.append(sample)
        
        for i in range(len(samples2)):
            sample = samples2[i]
            if len(sample['all_cls']) == 0:
                bg_samples.append(sample)
            else:
                fg_samples2.append(sample)

        #TODO if nodule length > 12, then not mixup
        # Mixup 
        indcies1 = np.arange(len(fg_samples1))
        indcies2 = np.arange(len(fg_samples2))
        k = 2
        if len(indcies1) != 0 and len(indcies2) == 0:
            # Combimnation
            combs = list(itertools.product(indcies1, indcies2))
            combs = combs[:k]
            for idx1, idx2 in combs:
                sample1 = fg_samples1[idx1]
                sample2 = fg_samples2[idx2]
                mixup_weight = get_mixup_weight()
                
                # bg_samples.append(sample1)
            
        
                
        # random_samples = []
        # for i in range(len(samples)):
        #     sample = samples[i]
        #     sample['image'] = normalize_raw_image(sample['image'])
        #     sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
        #     if self.transform_post:
        #         sample['ctr_transform'] = []
        #         sample['feat_transform'] = []
        #         sample = self.transform_post(sample)
        #     random_samples.append(sample)

        # return random_samples

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