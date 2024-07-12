from evaluationScript.nodule_finding import NoduleFinding
from collections import defaultdict
from evaluationScript.nodule_typer import compute_nodule_volume
import time
import os
import json
import numpy as np
from typing import Dict
from dataload.utils import ALL_LOC, ALL_RAD, ALL_CLS, BBOXES, NODULE_SIZE, load_series_list
import shutil

NODULE_START_SLICE_IDS = 'nodule_start_slice_ids'
LAST_MODIFIED_TIME = 'last_modified_time'
TRUE_NODULE_TYPE = 'true_nodule_type'
cur_time = time.time()

def pred2nodulefinding(line: str) -> NoduleFinding:
    pred = line.strip().split(',')
    series_name, _ , x, y, z, w, h, d, prob, true_nodule_type = pred
    nodule = NoduleFinding(series_name, x, y, z, w, h, d, true_nodule_type, prob)
    return nodule

def convert_true_nodule_type(true_nodule_type: str) -> str:
    if true_nodule_type == 'benign nodule':
        return 'benign'
    elif true_nodule_type == 'ground glass nodule':
        return 'tp'
    elif true_nodule_type == 'confuse':
        return 'confuse'
    else:
        raise ValueError(f'Unknown true nodule type: {true_nodule_type}')

save_folder = './data/patch_jsons'
pred_path = './hard_FP_client_prob06_07.csv'

if __name__ == '__main__':
    with open(pred_path, 'r') as f:
        lines = f.readlines()[1:] # skip header
        
    series_infos = load_series_list('./data/all.txt')
        
    # Get series name to folder mapping
    series_names_to_folder = {s[1]: s[0] for s in series_infos}
    series_names = [s[1] for s in series_infos]
        
    series_nodules = defaultdict(list)
    all_nodules = []
    for line in lines:
        nodule = pred2nodulefinding(line)
        if nodule.nodule_type == 'not':
            continue
        series_nodules[nodule.series_name].append(nodule)
        all_nodules.append(nodule)
        
    for series_name, nodules in series_nodules.items():
        bboxes = []
        nodule_sizes = []
        true_nodule_types = []
        for nodule in nodules:
            bbox = nodule.get_box()
            bbox = np.array(bbox) # [(z1, y1, x1), (z2, y2, x2)]
            bbox = np.round(bbox).astype(np.int32)
            # Convert bbox to [(y1, x1, z1), (y2, x2, z2)]
            bbox = bbox[:, [1, 2, 0]]
            bboxes.append(bbox)
            # Compute nodule size
            nodule_size = compute_nodule_volume(nodule.d, nodule.h, nodule.w) 
            nodule_sizes.append(nodule_size)
            true_nodule_types.append(convert_true_nodule_type(nodule.nodule_type))
        bboxes = np.concatenate(bboxes, axis=0) # (n, 2, 6)
        bboxes = np.reshape(bboxes, (-1, 6))
        nodule_sizes = np.array(nodule_sizes).round().astype(np.int32)
        # Sort min z of each bbox
        min_zs = bboxes[:, 0]
        sorted_indices = np.argsort(min_zs)
        bboxes = bboxes[sorted_indices]
        nodule_sizes = nodule_sizes[sorted_indices]
        true_nodule_types = [true_nodule_types[i] for i in sorted_indices]
        # Reshape bboxes to (n, 2, 3)
        bboxes = np.reshape(bboxes, (-1, 2, 3))
        nodule_start_slice_ids = bboxes[:, 0, 2] # z1
        
        nodule_count = {LAST_MODIFIED_TIME: cur_time, 
                        NODULE_SIZE: nodule_sizes.tolist(),
                        BBOXES: bboxes.tolist(),
                        TRUE_NODULE_TYPE: true_nodule_types,
                        NODULE_START_SLICE_IDS: nodule_start_slice_ids.tolist()}
        
        save_path = os.path.join(save_folder, f'{series_name}.json')
        
        if os.path.exists(save_path):
            old_nodule_count = json.load(open(save_path, 'r'))
            nodule_count = {LAST_MODIFIED_TIME: cur_time,
                            NODULE_SIZE: old_nodule_count[NODULE_SIZE] + nodule_sizes.tolist(),
                            BBOXES: old_nodule_count[BBOXES] + bboxes.tolist(),
                            TRUE_NODULE_TYPE: old_nodule_count[TRUE_NODULE_TYPE] + true_nodule_types,
                            NODULE_START_SLICE_IDS: old_nodule_count[NODULE_START_SLICE_IDS] + nodule_start_slice_ids.tolist()}
        
        with open(save_path, 'w') as f:
            json.dump(nodule_count, f)
            
    for f in os.listdir(save_folder):
        src_path = os.path.join(save_folder, f)
        series_name = f.split('.')[0]
        dst_path = os.path.join(series_names_to_folder[series_name], 'mask', f'{series_name}_nodule_count_patch_crop.json')
        shutil.copy(src_path, dst_path)
        print(f'Copy {src_path} to {dst_path}')