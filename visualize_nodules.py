import os
import math
import argparse
import numpy as np
from typing import List, Tuple
from collections import defaultdict

from dataload.utils import load_label, gen_label_path, gen_dicom_path, ALL_CLS, ALL_RAD, ALL_LOC, ALL_PROB, load_image, load_series_list
from evaluationScript.nodule_finding import NoduleFinding
from visualize.draw import draw_bbox_on_image, draw_pseu_bbox_and_label_on_image
from visualize.convert import noduleFinding2cude, gtNoduleFinding2cube
from utils.utils import get_progress_bar

def get_args():
    parser = argparse.ArgumentParser(description='Visualize Hard False Positive')
    parser.add_argument('--val_set', type=str, default='./data/all.txt')
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--hard_FP_thresh', type=float, default=0.7)
    parser.add_argument('--half_image', action='store_true', default=False)
    args = parser.parse_args()
    return args

def str2bool(v):
    return v.lower() in ('true', '1')

def pred2nodulefinding(line: str) -> NoduleFinding:
    pred = line.strip().split(',')
    if len(pred) == 11: # no ground truth
        series_name, x, y, z, w, h, d, prob, nodule_type, match_iou, is_gt = pred
        gt_x = None
    else:
        series_name, x, y, z, w, h, d, prob, nodule_type, match_iou, is_gt, gt_x, gt_y, gt_z, gt_w, gt_h, gt_d = pred
    is_gt = str2bool(is_gt)
    nodule = NoduleFinding(series_name, x, y, z, w, h, d, nodule_type, prob, is_gt)
    if gt_x is not None:
        gt_nodule = NoduleFinding(series_name, gt_x, gt_y, gt_z, gt_w, gt_h, gt_d, nodule_type, prob, is_gt)
        nodule.set_match(match_iou, gt_nodule)
    return nodule

if __name__ == '__main__':
    args = get_args()
    hard_FP_save_folder = os.path.join(args.save_folder, 'hard_FP')
    TP_save_folder = os.path.join(args.save_folder, 'TP')
    FN_save_folder = os.path.join(args.save_folder, 'FN')
    
    os.makedirs(hard_FP_save_folder, exist_ok=True)
    os.makedirs(TP_save_folder, exist_ok=True)
    os.makedirs(FN_save_folder, exist_ok=True)
    series_infos = load_series_list(args.val_set)
    
    # Get series name to folder mapping
    series_names_to_folder = {s[1]: s[0] for s in series_infos}
    series_names = [s[1] for s in series_infos]
    
    # Load all nodule findings
    with open(args.pred_path, 'r') as f:
        lines = f.readlines()[1:] # skip header
    nodules = [pred2nodulefinding(line) for line in lines]
    
    # Draw hard false positive nodules
    # hard_FP_nodules = defaultdict(list)
    # for n in nodules:
    #     if n.prob >= args.hard_FP_thresh and not n.is_gt:
    #         hard_FP_nodules[n.series_name].append(n)
    # with get_progress_bar('Visualizing Hard FP', len(hard_FP_nodules)) as pbar:
    #     for series_name, nodule_findings in hard_FP_nodules.items():
    #         img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
    #         image = (load_image(img_path) * 255).astype(np.uint8)
    #         for i, nodule in enumerate(nodule_findings):
    #             save_path = os.path.join(hard_FP_save_folder, f'{series_name}_{i}.png')
    #             bboxes = noduleFinding2cude([nodule], image.shape)
    #             draw_bbox_on_image(image, bboxes, (0, 255, 0), half_image=args.half_image, save_path=save_path)
    #         pbar.update(1)
        
    # Draw TP nodules
    TP_nodules = defaultdict(list)
    for n in nodules:
        if n.is_gt and n.prob > 0:
            TP_nodules[n.series_name].append(n)
            
    with get_progress_bar('Visualizing TP', len(TP_nodules)) as pbar:
        for series_name, nodule_findings in TP_nodules.items():
            img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
            image = (load_image(img_path) * 255).astype(np.uint8)
            for i, nodule in enumerate(nodule_findings):
                save_path = os.path.join(TP_save_folder, f'{series_name}_{i}.png')
                pred_bboxes, gt_bboxes = gtNoduleFinding2cube([nodule], image.shape)
                draw_pseu_bbox_and_label_on_image(image, gt_bboxes, pred_bboxes, save_path=save_path, half_image=args.half_image)
            pbar.update(1)
            
    # Draw FN nodules
    FN_nodules = defaultdict(list)
    for n in nodules:
        if n.is_gt and n.prob == -1:
            FN_nodules[n.series_name].append(n)
        
    with get_progress_bar('Visualizing FN', len(FN_nodules)) as pbar:
        for series_name, nodule_findings in FN_nodules.items():
            img_path = gen_dicom_path(series_names_to_folder[series_name], series_name)
            image = (load_image(img_path) * 255).astype(np.uint8)
            for i, nodule in enumerate(nodule_findings):
                save_path = os.path.join(FN_save_folder, f'{series_name}_{i}.png')
                bboxes = noduleFinding2cude([nodule], image.shape)
                draw_bbox_on_image(image, bboxes, (0, 255, 0), half_image=args.half_image, save_path=save_path)
            pbar.update(1)