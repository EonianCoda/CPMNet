import os
import pickle
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from dataload.utils import load_label, gen_label_path, gen_dicom_path, ALL_CLS, ALL_RAD, ALL_LOC, ALL_PROB, load_image, load_series_list
from evaluationScript.nodule_finding import NoduleFinding
from visualize.draw import draw_bbox_on_image, draw_pseu_bbox_and_label_on_image
from visualize.convert import noduleFinding2cude, gtNoduleFinding2cube
from analysis.utils import pred2nodulefinding
from utils.utils import get_progress_bar

def write_csv(series_nodules: Dict[str, NoduleFinding], save_path: str):
    header = ['series_name', 'nodule_idx', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd', 'probability']
    header = ','.join(header)
    lines = []
    for series_name, nodule_findings in series_nodules.items():
        for i, n in enumerate(nodule_findings):
            lines.append(f'{series_name},{i},{n.ctr_x},{n.ctr_y},{n.ctr_z},{n.w},{n.h},{n.d},{n.prob}')
    with open(save_path, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(lines))    
            
def get_args():
    parser = argparse.ArgumentParser(description='Visualize Hard False Positive')
    parser.add_argument('--val_set', type=str, default='./data/all.txt')
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
   
    parser.add_argument('--offset', type=int, default=3)
    parser.add_argument('--bbox_offset', type=int, default=3)
    parser.add_argument('--z_offset', type=int, default=0)
    
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--half_image', action='store_true', default=False)
    
    parser.add_argument('--single_series', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.single_series != None:
        args.save_folder = os.path.join(args.save_folder, args.single_series)
    
    os.makedirs(args.save_folder, exist_ok=True)    
    series_infos = load_series_list(args.val_set)
    
    # Get series name to folder mapping
    series_names_to_folder = {s[1]: s[0] for s in series_infos}
    series_names = [s[1] for s in series_infos]
    
    # Load all nodule findings
    with open(args.pred_path, 'rb') as f:
        pseudo_labels = pickle.load(f)
        
    with get_progress_bar('Visualizing Nodule', len(pseudo_labels)) as pbar:
        for series_name, nodules in pseudo_labels.items():
            valid_mask = nodules[ALL_PROB] > args.thresh
            if np.sum(valid_mask) == 0:
                continue
            
            all_loc = nodules[ALL_LOC][valid_mask]
            all_rad = nodules[ALL_RAD][valid_mask]
            all_prob = nodules[ALL_PROB][valid_mask]
            img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
            image = (load_image(img_path) * 255).astype(np.uint8)

            for i, (loc, rad, prob) in enumerate(zip(all_loc, all_rad, all_prob)):
                box = np.array([loc - rad, loc + rad])
                box = np.expand_dims(box, axis=0)
                extra_sup_title = '{}, Prob: {:.2f}'.format(series_name, prob)
                save_path = os.path.join(args.save_folder, f'{series_name}_{i}.png')
                draw_bbox_on_image(image, box, (0, 255, 0), half_image=args.half_image, save_path=save_path, extra_sup_title=extra_sup_title, offset=args.offset, bbox_offset=args.bbox_offset, z_offset=args.z_offset)
            pbar.update(1)