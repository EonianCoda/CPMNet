import cv2
import os
import numpy as np
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt

from dataload.utils import ALL_LOC, ALL_RAD, load_image, load_label, gen_dicom_path, gen_label_path
from evaluationScript.nodule_finding import NoduleFinding
from .convert import output2nodulefinding, label2nodulefinding, nodule2cude

MAX_IMAGE_IN_ROW = 6
SUBPLOT_WIDTH = 3.5
SUBPLOT_HEIGHT = 9

def draw_bbox(image: np.ndarray, bboxes: np.ndarray, color = (255, 0, 0)) -> np.ndarray:
    """
    Args:
        image: a 3D image with shape [Z, Y, X, 3]
    """
    for z1, y1, x1, z2, y2, x2 in bboxes:
        for z in range(int(z1), int(z2)):
            image[z] = cv2.rectangle(image[z].copy(), (x1, y1), (x2, y2), color, 1)
    return image

def draw_bbox_on_image(image: np.ndarray, bboxes: np.ndarray, color = (255, 0, 0)) -> np.ndarray:
    """
    Args:
        image: a 3D image with shape [Z, Y, X, 3]
        bbox: a 2D array with shape [N, 6] where N is the number of bounding boxes and 6 is [z1, y1, x1, z2, y2, x2]
    """
    if len(image.shape) == 3:
        image = image[..., np.newaxis]
        image = np.repeat(image, 3, axis=-1)
    
    for bbox in bboxes:
        z1, y1, x1, z2, y2, x2 = bbox
        center_x = (x1 + x2) / 2
        bboxed_image = draw_bbox(image.copy(), bbox[np.newaxis, ...], color)
        
        # Crop image to save drawing time and space
        if center_x < image.shape[2] / 2: # left
            bboxed_image = bboxed_image[:, :, :bboxed_image.shape[2] // 2]
        else: # right
            bboxed_image = bboxed_image[:, :, bboxed_image.shape[2] // 2:]
        
        if (z2 - z1) > MAX_IMAGE_IN_ROW:
            zs = list(range(z1, z2, (z2 - z1) // MAX_IMAGE_IN_ROW))
        else:
            zs = list(range(z1, z2))
            
        # Draw
        plt.figure(figsize=(int(len(zs) * SUBPLOT_WIDTH), SUBPLOT_HEIGHT))
        for i, z in enumerate(zs):
            ax = plt.subplot(1, len(zs), i+1)
            ax.imshow(bboxed_image[z], cmap='gray')
            ax.set_title(f'z={z}')
            ax.axis('off')
        plt.tight_layout()
        
def draw_bbox_on_image_with_label(img_path: str, label_path: str, color = (255, 0, 0)) -> np.ndarray:
    """
    Args:
        img_path: path to the image
        label_path: path to the label
    """
    image = load_image(img_path)
    image = image * 2 - 1 # normalize to [-1, 1]
    
    label = load_image(load_label(label_path, np.array([1.0, 1.0, 1.0]))) # use pixel rather than physical spacing
    
    nodule_findings = label2nodulefinding(label)
    bboxes = nodule2cude(nodule_findings, image.shape)
    draw_bbox_on_image(image, bboxes, color)
    
    
if __name__ == '__main__':
    # draw FN
    ME_ROOT = r'D:\workspace\medical_dataset\ME_dataset'
    LDCT_ROOT = r'D:\workspace\medical_dataset\LDCT_test_dataset'

    def get_dicom_and_label_path(series_name: str) -> Tuple[str, str]:
        """
        Return:
            A tuple of (dicom_path, label_path)
        """
        if 'Test' in series_name:
            root = LDCT_ROOT
        else:
            root = ME_ROOT
        folder = os.path.join(root, series_name) 
        # load dicom and label
        dicom_path = gen_dicom_path(folder, series_name)
        label_path = gen_label_path(folder, series_name)
        return dicom_path, label_path

    # load FN series list
    # csv_path = './save/[2024-02-14-1706]_all_ME_bs3_ns5/annotation/[2024-02-14-1706]_all_ME_bs3_ns5/annotation/predict_epoch_160/FN_0.1.txt'
    csv_path = r'D:\workspace\python\My_CPM_Net\save\[2024-02-14-1706]_all_ME_bs3_ns5\val_temp\annotation\epoch_0\FN_0.1.csv'
    with open(csv_path, 'r') as f:
        if csv_path.endswith('.csv'):
            lines = f.readlines()[1:] # skip header
            fn_nodules_infos = [line.strip().split(',') for line in lines]
        else:
            lines = f.readlines()
            fn_nodules_infos = [line.strip().split(',') for line in lines]
            temp = []
            for n in fn_nodules_infos:
                temp.append([n[0].replace('_crop.npy', ''), n[2], n[3], n[4], n[5], n[6], n[7]])
            fn_nodules_infos = temp
    fn_dicom_paths = []
    fn_nodule_findings = []
    for i, nodule in enumerate(fn_nodules_infos):
        series_name, x, y, z, w, h, d = nodule
        dicom_path, label_path = get_dicom_and_label_path(series_name)
        fn_dicom_paths.append(dicom_path)
        nodule_finding = NoduleFinding(coordX=x, coordY=y, coordZ=z, w=w, h=h, d=d)
        fn_nodule_findings.append(nodule_finding)
        
    for idx in range(len(fn_dicom_paths)):
        fn_nodule_finding = fn_nodule_findings[idx]
        if fn_nodule_finding.d >= 3:
            continue
        image = load_image(fn_dicom_paths[idx])
        image = image * 2 - 1

        print(str(fn_nodule_finding))
        fn_bboxes = nodule2cude(fn_nodule_finding, image.shape)

        mapped_image = ((image + 1) * 127.5).astype(np.uint8)
        # copy 3D image to 3 channels
        mapped_image = np.stack([mapped_image, mapped_image, mapped_image], axis=-1) # [Z, Y, X, 3]
        draw_bbox_on_image(mapped_image, fn_bboxes, color=(255, 0, 0))