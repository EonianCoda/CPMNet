# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List, Dict, Tuple
from .utils import load_series_list, load_image, load_label, load_lobe, ALL_RAD, ALL_LOC, ALL_CLS, NODULE_SIZE, gen_dicom_path, gen_label_path, \
                    gen_lobe_path, normalize_processed_image, normalize_raw_image, compute_bbox3d_iou
from torch.utils.data import Dataset

import math
from analysis.utils import load_predictions
logger = logging.getLogger(__name__)

def get_self_refined_labels(series_predictions: dict, image_spacing: np.ndarray, conf_threshold: float = 0.6, 
                            nodule_size_threshold: int = 512, expanded_offset: int = 3):
    series_names = list(series_predictions.keys())
    series_name = series_names[0]

    self_refined_labels = dict()
    for series_name in series_names:
        gt_nodules = []
        pred_valid_nodules = []
        pred_nodules = series_predictions[series_name]

        for n in pred_nodules:
            if n.is_gt == 1 and n.prob != -1:
                gt_nodules.append(n.match_nodule_finding)
            elif n.is_gt == 1 and n.prob == -1:
                gt_nodules.append(n)
            else:
                box = n.get_box() # [[z1, y1, x1], [z2, y2, x2]]
                conf = n.prob
                d, h, w = box[1][0] - box[0][0], box[1][1] - box[0][1], box[1][2] - box[0][2]
                nodule_sizes =  d * h * w
                if nodule_sizes >= nodule_size_threshold or conf < conf_threshold:
                    continue
                pred_valid_nodules.append(n)
        if len(pred_valid_nodules) != 0:
            if len(gt_nodules) != 0: # Check the iou with gt nodules
                gt_boxes = [n.get_box() for n in gt_nodules]
                gt_boxes = np.array(gt_boxes) # (n, 2, 3)
                gt_boxes[:, 0, :] -= expanded_offset
                gt_boxes[:, 1, :] += expanded_offset
                
                pred_boxes = [n.get_box() for n in pred_valid_nodules]
                pred_boxes = np.array(pred_boxes)
                pred_boxes[:, 0, :] -= expanded_offset
                pred_boxes[:, 1, :] += expanded_offset
                
                iou = compute_bbox3d_iou(gt_boxes, pred_boxes)
                pred_match_gt_ious = np.max(iou, axis=0)
                valid_mask = pred_match_gt_ious < 0.0
                
                pred_valid_nodules = [n for n, valid in zip(pred_valid_nodules, valid_mask) if valid]
            if len(pred_valid_nodules) != 0:
                all_boxes = [n.get_box() for n in pred_valid_nodules]
                all_boxes = np.array(all_boxes)
                
                all_loc = (all_boxes[:, 0, :] + all_boxes[:, 1, :]) / 2 # (n, 3)
                all_rad = (all_boxes[:, 1, :] - all_boxes[:, 0, :]) # (n, 3)
                nodule_size = (4/3 * math.pi * np.prod(all_rad, axis=1) / 6)
                all_rad = all_rad * image_spacing
                
                label = {ALL_LOC: all_loc, 
                        ALL_RAD: all_rad,
                        ALL_CLS: np.ones(len(all_rad)),
                        NODULE_SIZE: nodule_size}
                
                self_refined_labels[series_name] = label
    return self_refined_labels

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
    def __init__(self, series_list_path: str, pred_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, 
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
        series_predictions, _ = load_predictions(pred_path, series_list_path)
        refined_labels = get_self_refined_labels(series_predictions, image_spacing, conf_threshold=0.6, nodule_size_threshold=512, expanded_offset=3)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d, min_size)
            
            if series_name in refined_labels:
                refined_label = refined_labels[series_name]
                label[ALL_LOC] = np.concatenate([label[ALL_LOC], refined_label[ALL_LOC]], axis=0)
                label[ALL_RAD] = np.concatenate([label[ALL_RAD], refined_label[ALL_RAD]], axis=0)
                label[ALL_CLS] = np.concatenate([label[ALL_CLS], refined_label[ALL_CLS]], axis=0)
                label[NODULE_SIZE] = np.concatenate([label[NODULE_SIZE], refined_label[NODULE_SIZE]], axis=0)
                
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                    continue
            
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
        samples['file_name'] = series_name
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