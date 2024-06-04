import pickle
import os
import torch
import numpy as np
from dataload.utils import ALL_LOC, ALL_RAD, ALL_PROB, compute_bbox3d_iou, load_series_list, gen_label_path, load_label, gen_dicom_path, load_image

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_pseudo_labels(root: str):
    ema_update_labels_root = os.path.join(root, 'ema_update_labels')
    history_psuedo_labels_root = os.path.join(root, 'history_psuedo_labels')
    
    num_epoch = len(os.listdir(ema_update_labels_root))
    ema_update_labels = [load_pickle(os.path.join(ema_update_labels_root, f'ema_updated_labels_{epoch}.pkl')) for epoch in range(num_epoch)]
    history_psuedo_labels = [load_pickle(os.path.join(history_psuedo_labels_root, f'history_psuedo_labels_{epoch}.pkl')) for epoch in range(num_epoch)]
    
    return ema_update_labels, history_psuedo_labels

def load_gt_labels(series_list_path: str):
    series_info = load_series_list(series_list_path)
    image_spacing = np.array([1.0, 1.0, 1.0])

    all_gt_labels = dict()
    dicom_paths = dict()
    for series_folder, series_name in series_info:
        label_path = gen_label_path(series_folder, series_name)
        label = load_label(label_path, image_spacing, min_size=27)
        all_gt_labels[series_name] = label

        dicom_path = gen_dicom_path(series_folder, series_name)
        dicom_paths[series_name] = dicom_path

    return all_gt_labels, dicom_paths