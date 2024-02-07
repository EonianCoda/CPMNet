import json
import os
import numpy as np
from dataload.my_dataset import load_series_list
import argparse
from typing import List

BBOXES = 'bboxes'
# SPACING = [1.0, 0.8, 0.8] # (z, y, x)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_list_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./annotation.csv')
    args = parser.parse_args()
    return args

def generate_annot_csv(series_list_path: str,
                       save_path: str,
                       spacing: List[float] = None):
    spacing = np.array(spacing)
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd']

    all_locs = []
    all_rads = []
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
            continue
        # calculate center of bboxes
        all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
        all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

        all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
        all_rad = all_rad[:, [2, 0, 1]] # (d, h, w)
        all_rad = all_rad * spacing
        
        all_locs.append(all_loc)
        all_rads.append(all_rad)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(','.join(column_order) + '\n')
        for series_i in range(len(series_infos)):
            for i in range(len(all_locs[series_i])):
                loc = all_locs[series_i][i]
                rad = all_rads[series_i][i]
                z, y, x = loc
                d, h, w = rad
                
                row = [series_infos[series_i][1] + '_crop.npy']
                for i in [x, y, z, w, h, d]:
                    row.append('{:.2f}'.format(i))
                f.write(','.join(row) + '\n')
    
if __name__ == '__main__':
    args = get_args()
    series_list_path = args.series_list_path
    save_path = args.save_path
    generate_annot_csv(series_list_path, save_path)
    