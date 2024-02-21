import json
import os
import math
import logging
import numpy as np
from dataload.utils import load_series_list, load_label, gen_label_path, ALL_CLS, ALL_LOC, ALL_RAD, NODULE_SIZE
import argparse
from typing import List

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
                       min_d: int = 0):
    spacing = np.array(spacing)
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'nodule_type']
    
    nodule_typer = NoduleTyper(spacing)    
    all_locs = []
    all_rads = []
    all_types = []
    series_infos = load_series_list(series_list_path)
    for series_info in series_infos:
        folder = series_info[0]
        series_name = series_info[1]
        
        label_path = gen_label_path(folder, series_name)
        label = load_label(label_path, spacing, min_d)
        all_locs.append(label[ALL_LOC].tolist())
        all_rads.append(label[ALL_RAD].tolist())
        all_types.append([nodule_typer.get_nodule_type(s) for s in label[NODULE_SIZE]])
        
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
    