import os
import json
import numpy as np

from typing import Dict, List

HU_MIN, HU_MAX = -1000, 400
BBOXES = 'bboxes'
ALL_LOC = 'all_loc'
ALL_RAD = 'all_rad'
ALL_CLS = 'all_cls'

def gen_dicom_path(folder: str, series_name: str) -> str:
    return os.path.join(folder, 'npy', f'{series_name}_crop.npy')

def gen_label_path(folder: str, series_name: str) -> str:
    return os.path.join(folder, 'mask', f'{series_name}_nodule_count_crop.json')

def load_series_list(series_list_path: str) -> List[List[str]]:
    """
    Return:
        series_list: list of tuples (series_folder, file_name)

    """
    with open(series_list_path, 'r') as f:
        lines = f.readlines()
        lines = lines[1:] # Remove the line of description
        
    series_list = []
    for series_info in lines:
        series_info = series_info.strip()
        series_folder, file_name = series_info.split(',')
        series_list.append([series_folder, file_name])
    return series_list

def normalize(image: np.ndarray) -> np.ndarray: 
    image = np.clip(image, HU_MIN, HU_MAX)
    image = image - HU_MIN
    image = image.astype(np.float32) / (HU_MAX - HU_MIN)
    return image

def load_image(dicom_path: str) -> np.ndarray:
    """
    Return:
        A 3D numpy array with dimension order [D, H, W] (z, y, x)
    """
    image = np.load(dicom_path)
    image = np.transpose(image, (2, 0, 1))
    image = normalize(image)
    return image

def load_label(label_path: str, image_spacing: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return:
        A dictionary with keys 'all_loc', 'all_rad', 'all_cls'
    """
    with open(label_path, 'r') as f:
        info = json.load(f)
    
    bboxes = np.array(info[BBOXES]) # (n, 2, 3)
    if len(bboxes) == 0:
        label = {ALL_LOC: np.zeros((0, 3), dtype=np.float32),
                ALL_CLS: np.zeros((0, 3), dtype=np.float32),
                ALL_RAD: np.zeros((0,), dtype=np.int32)}
    else:
        bboxes[:, 0, :] = np.maximum(bboxes[:, 0, :], 0) # clip to 0
        if (bboxes < 0).any():
            print(f'Warning: {label_path} has negative values')
        # calculate center of bboxes
        all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
        all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)
        
        all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad * image_spacing # (z, y, x)
        all_cls = np.zeros((all_loc.shape[0],), dtype=np.int32)
        
        label = {ALL_LOC: all_loc, 
                ALL_RAD: all_rad,
                ALL_CLS: all_cls}
    return label