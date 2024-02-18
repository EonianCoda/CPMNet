import json
import os
import math
import logging
import numpy as np
from dataload.utils import load_series_list
import argparse
from typing import List

BBOXES = 'bboxes'
NODULE_SIZE = 'nodule_size'

logger = logging.getLogger(__name__)

def generate_series_uids_csv(series_list_path: str, save_path: str):
    series_infos = load_series_list(series_list_path)
    header = 'seriesuid\n'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(header)
        for folder, series_name in series_infos:
            f.write(series_name + '\n')

class NoduleTyper:
    def __init__(self, spacing: List[float]):
        self.diamters = {'benign': [0,4], 
                        'probably_benign': [4, 6],
                        'probably_suspicious': [6, 8],
                        'suspicious': [8, -1]}
        
        self.spacing = np.array(spacing)
        self.voxel_volume = np.prod(self.spacing)
        
        self.areas = {}
        for key in self.diamters:
            self.areas[key] = [round(self.compute_sphere_volume(self.diamters[key][0]) / self.voxel_volume),
                               round(self.compute_sphere_volume(self.diamters[key][1]) / self.voxel_volume)]
        # logging.info('nodule areas: {}'.format(self.areas))
        
    @staticmethod
    def compute_sphere_volume(diameter: float) -> float:
        if diameter == 0:
            return 0
        elif diameter == -1:
            return 100000000
        else:
            radius = diameter / 2
            return 4/3 * math.pi * radius**3
        
    def get_nodule_type(self, nodule_size: float) -> str:
        for key in self.areas:
            if nodule_size >= self.areas[key][0] and (nodule_size < self.areas[key][1] or self.areas[key][1] == -1):
                return key
        return 'benign'

def generate_annot_csv(series_list_path: str,
                       save_path: str,
                       spacing: List[float] = None,
                       min_d: float = 0):
    spacing = np.array(spacing)
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'nodule_type']
    
    nodule_typer = NoduleTyper(spacing)    
    all_locs = []
    all_rads = []
    all_types = []
    series_infos = load_series_list(series_list_path)
    for folder, series_name in series_infos:
        label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count_crop.json')
        with open(label_path, 'r') as f:
            info = json.load(f)
            
        bboxes = info[BBOXES]
        bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            all_locs.append([])
            all_rads.append([])
            all_types.append([])
            continue
        
        # calculate center of bboxes
        all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
        all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

        all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad[:, [2, 0, 1]] # (d, h, w)
        
        valid_mask = all_rad[:, 0] >= min_d
        
        all_rad = all_rad * spacing
        
        if np.sum(valid_mask) == 0:
            all_locs.append([])
            all_rads.append([])
            all_types.append([])
            continue
        else:
            all_loc = all_loc[valid_mask]
            all_rad = all_rad[valid_mask]
            nodule_sizes = info[NODULE_SIZE]
            nodule_sizes = np.array(nodule_sizes)[valid_mask]
            
            all_locs.append(all_loc)
            all_rads.append(all_rad)

            all_types.append([nodule_typer.get_nodule_type(nodule_size) for nodule_size in nodule_sizes])
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(','.join(column_order) + '\n')
        for series_i in range(len(series_infos)):
            for loc, rad, nodule_type in zip(all_locs[series_i], all_rads[series_i], all_types[series_i]):
                z, y, x = loc
                d, h, w = rad
                series_name = series_infos[series_i][1]
                row = [series_name]
                
                for value in [x, y, z, w, h, d]:
                    row.append('{:.2f}'.format(value))
                
                row.append(nodule_type)
                f.write(','.join(row) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_list_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./annotation.csv')
    args = parser.parse_args()
    series_list_path = args.series_list_path
    save_path = args.save_path
    generate_annot_csv(series_list_path, save_path)
    