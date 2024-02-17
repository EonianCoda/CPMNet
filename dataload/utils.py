import os
import numpy as np

HU_MIN, HU_MAX = -1000, 400

def load_series_list(series_list_path: str):
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