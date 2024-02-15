# coding:utf-8
import os
import json
import logging
from typing import Tuple, List, Any, Dict
from itertools import product
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dataload.my_dataset import load_series_list
from .nodule_finding import NoduleFinding

logger = logging.getLogger(__name__)

BBOXES = 'bboxes'
DEFAULT_IOU_THRESHOLDS = [0.0, 0.01, 0.1, 0.2, 0.5, 0.75, 0.9, 0.95]
DEFAULT_PROB_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_FP_RATIOS = [0.125, 0.25, 0.5, 1, 2, 4, 8]

NUM_INTERPOLATION_POINTS = 10001

def compute_bbox3d_iou(box1: npt.NDArray[np.int32], box2: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """ 
    Args:
        box1:
            shape = [N, 2, 3] = [[x1, y1, z1], [x2, y2, z2]]
        box2:
            shape = [M, 2, 3] = [[x1, y1, z1], [x2, y2, z2]]
    Return:
        the IoU of the area of the intersection between box1 and box2, shape = [N, M]
    """
    a1, a2 = box1[:,np.newaxis, 0,:], box1[:,np.newaxis, 1,:] # [N, 1, 3]
    b1, b2 = box2[np.newaxis,:, 0,:], box2[np.newaxis,:, 1,:] # [1, N, 3]
    inter_area = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod(axis=2) # [N, M]
   
    area_a = np.prod(box1[:, 1, :] - box1[:, 0, :], axis=1) # [N,]
    area_b = np.prod(box2[:, 1, :] - box2[:, 0, :], axis=1) # [M,]
    # [N, M]
    area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area 
    iou = inter_area / area
    return iou

def match_pred_with_gt(all_gt_nodules: Dict[str, List[NoduleFinding]],
                        all_pred_nodules: Dict[str, List[NoduleFinding]]):
    for series_name in all_gt_nodules.keys():
        gt_nodules = all_gt_nodules[series_name]
        pred_nodules = all_pred_nodules.get(series_name, list)
        
        gt_bboxes = np.array([n.bbox for n in gt_nodules])
        pred_bboxes = np.array([n.bbox for n in pred_nodules])
        gt_bboxes = gt_bboxes.reshape(-1, 2, 3)
        pred_bboxes = pred_bboxes.reshape(-1, 2, 3)
        
        iou = compute_bbox3d_iou(gt_bboxes, pred_bboxes)
        gt_ious = np.max(iou, axis=1)
        argmax_gt_ious = np.argmax(iou, axis=1)
        
        pred_ious = np.max(iou, axis=0)
        
        for iou, gt_nodule, match_i in zip(gt_ious, gt_nodules, argmax_gt_ious):
            gt_nodule.iou = iou
            if iou > 0:
                gt_nodule.pred_prob = pred_nodules[match_i].pred_prob
            else:
                gt_nodule.pred_prob = 0
        for iou, pred_nodule in zip(pred_ious, pred_nodules):
            pred_nodule.iou = iou

def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Return:
        A tuple of (sensitivity, precision, f1_score)
    """
    sensitivity = tp / max(tp + fn, 1e-6)
    precision = tp / max(tp + fp, 1e-6)
    f1_score = 2 * precision * sensitivity / max(precision + sensitivity, 1e-6)
    return sensitivity, precision, f1_score

def compute_froc_btp(scan_qualified_tp_fp_fn: List[List[int]],
                    scan_unqualified_tp_fp_fn: List[List[int]],
                    scan_probs: List[List[float]],
                    indices: List[int],
                    froc_fp_ratios: List[float]):
    qualified_tp_fp_fn = []
    unqualified_tp_fp_fn = []
    probs = []
    for i in indices:
        qualified_tp_fp_fn.extend(scan_qualified_tp_fp_fn[i])
        unqualified_tp_fp_fn.extend(scan_unqualified_tp_fp_fn[i])
        probs.extend(scan_probs[i])
    
    qualified_tp_fp_fn = np.array(qualified_tp_fp_fn) # [N, 3]
    unqualified_tp_fp_fn = np.array(unqualified_tp_fp_fn) # [N, 3]
    probs = np.array(probs) # [N,]
    num_of_pos = sum(qualified_tp_fp_fn[:, 0])
    
    thresholded_tp_fp_fn = []
    froc_prob_thresholds = np.linspace(0, 1, num=NUM_INTERPOLATION_POINTS)
    if num_of_pos == 0:
        thresholded_tp_fp_fn = np.zeros((len(froc_prob_thresholds), 3))
    else:
        for prob_threshold in froc_prob_thresholds:
            thresholded_tp_fp_fn.append(np.where(np.array(probs) >= prob_threshold, qualified_tp_fp_fn, unqualified_tp_fp_fn).sum(axis=0))
        thresholded_tp_fp_fn = np.array(thresholded_tp_fp_fn)
        
    fp_per_scan = thresholded_tp_fp_fn[:, 1] / len(indices)
    sens = thresholded_tp_fp_fn[:, 0] / num_of_pos # sensitivity
    precs = thresholded_tp_fp_fn[:, 0] / np.maximum(thresholded_tp_fp_fn[:, 0] + thresholded_tp_fp_fn[:, 1], 1e-6) # precision
    f1_scores = 2 * precs * sens / np.maximum(precs + sens, 1e-6)
    
    fp_per_scan_interp = np.linspace(froc_fp_ratios[0], froc_fp_ratios[-1], num=NUM_INTERPOLATION_POINTS)
    sens_interp = np.interp(fp_per_scan_interp, fp_per_scan, sens)
    prec_interp = np.interp(fp_per_scan_interp, fp_per_scan, precs)
    f1_interp = np.interp(fp_per_scan_interp, fp_per_scan, f1_scores)
    
    return sens_interp, prec_interp, f1_interp
    
def compute_froc(all_gt_nodules: Dict[str, List[NoduleFinding]],
                    all_pred_nodules: Dict[str, List[NoduleFinding]],
                    froc_fp_ratios = DEFAULT_FP_RATIOS,
                    froc_iou_threshold: float = 0.1,
                    bootstrapping_times: int = 2000):
    num_of_scan = len(all_gt_nodules)
    
    scan_qualified_tp_fp_fn = []
    scan_unqualified_tp_fp_fn = []
    scan_probs = []
    
    for series_name in all_gt_nodules.keys():
        qualified_tp_fp_fn = []
        unqualified_tp_fp_fn = []
        probs = []
        for gt_nodule in all_gt_nodules[series_name]:
            probs.append(gt_nodule.pred_prob)
            if gt_nodule.iou >= froc_iou_threshold:
                qualified_tp_fp_fn.append([1, 0, 0])
            else:
                unqualified_tp_fp_fn.append([0, 0, 1])
        
        for pred_nodule in all_pred_nodules[series_name]:
            probs.append(pred_nodule.pred_prob)
            if pred_nodule.iou >= froc_iou_threshold:
                qualified_tp_fp_fn.append([0, 0, 0])
            else:
                unqualified_tp_fp_fn.append([0, 1, 0])

        scan_qualified_tp_fp_fn.append(qualified_tp_fp_fn)
        scan_unqualified_tp_fp_fn.append(unqualified_tp_fp_fn)
        scan_probs.append(probs)
    
    sens_interp, prec_interp, f1_interp = compute_froc_btp(scan_qualified_tp_fp_fn, scan_unqualified_tp_fp_fn, scan_probs, list(range(num_of_scan)), froc_fp_ratios)
    
    # Bootstrapping sampling to get the confidence interval
    sens_interp_btp = []
    prec_interp_btp = []
    f1_interp_btp = []
    for i in range(bootstrapping_times):
        sample_indices = np.random.choice(num_of_scan, num_of_scan, replace=True)
        sens, prec, f1 = compute_froc_btp(scan_qualified_tp_fp_fn, scan_unqualified_tp_fp_fn, scan_probs, sample_indices, froc_fp_ratios)
        sens_interp_btp.append(sens)
        prec_interp_btp.append(prec)
        f1_interp_btp.append(f1)
    
    sens_interp_btp = np.array(sens_interp_btp)
    prec_interp_btp = np.array(prec_interp_btp)
    f1_interp_btp = np.array(f1_interp_btp)
    
    sens_interp_btp_mean = np.mean(sens_interp_btp, axis=0)
    prec_interp_btp_mean = np.mean(prec_interp_btp, axis=0)
    f1_interp_btp_mean = np.mean(f1_interp_btp, axis=0)
    
    # Get sensitivity, precision, f1_score at different fp_per_scan
    fp_per_scan_interp_btp = np.linspace(froc_fp_ratios[0], froc_fp_ratios[-1], num=NUM_INTERPOLATION_POINTS)
    sens_points = []
    prec_points = []
    f1_points = []
    for i in range(froc_fp_ratios):
        index = np.argmin(abs(fp_per_scan_interp_btp - froc_fp_ratios[i]))
        sens_points.append(sens_interp_btp_mean[index])
        prec_points.append(prec_interp_btp_mean[index])
        f1_points.append(f1_interp_btp_mean[index])
    
    return (sens_interp_btp_mean, prec_interp_btp_mean, f1_interp_btp_mean), \
            (sens_points, prec_points, f1_points), \

def read_nodule_info(folder: str, series_name: str, spacing: np.ndarray) -> List[NoduleFinding]:
    label_path = os.path.join(folder, 'mask', f'{series_name}_nodule_count_crop.json')
    with open(label_path, 'r') as f:
        info = json.load(f)
        
    bboxes = info[BBOXES]
    bboxes = np.array(bboxes)
    
    if len(bboxes) == 0:
        return []
    
    # calculate center of bboxes
    all_loc = ((bboxes[:, 0] + bboxes[:, 1] - 1) / 2).astype(np.float32) # (y, x, z)
    all_rad = (bboxes[:, 1] - bboxes[:, 0]).astype(np.float32) # (y, x, z)

    all_loc = all_loc[:, [2, 0, 1]] # (z, y, x)
    all_rad = all_rad[:, [2, 0, 1]] # (d, h, w)
    all_rad = all_rad * spacing
    
    nodules = []
    for loc, rad in zip(all_loc, all_rad):
        z, y, x = loc
        d, h, w = rad
        
        nodule = NoduleFinding(coord_x=x, coord_y=y, coord_z=z, w=w, h=h, d=d)
        nodules.append(nodule)
    
    return nodules
                
class Evaluation(object):
    def __init__(self, 
                 series_list_path: str,
                 image_spacing: np.ndarray,
                 froc_iou_threshold: float = 0.1,
                 max_num_of_nodule_candidate_in_series: int = 100,
                 bootstrapping_times: int = 2000):
        self.series_list_path = series_list_path
        self.image_spacing = image_spacing
        self.froc_iou_threshold = froc_iou_threshold
        self.max_num_of_candidate = max_num_of_nodule_candidate_in_series
        self.bootstrapping_times = bootstrapping_times
        self.series_names = []
        
        self.gt_nodules = []
        for folder, series_name in load_series_list(series_list_path):
            self.series_names.append(series_name)
            self.gt_nodules[series_name] = read_nodule_info(folder, series_name, image_spacing)
    
    def evaluate(self,
                all_gt_nodules: Dict[str, List[NoduleFinding]],
                all_pred_nodules: Dict[str, List[NoduleFinding]],
                output_dir: str,
                froc_fp_ratios = DEFAULT_FP_RATIOS,
                iou_thresholds = DEFAULT_IOU_THRESHOLDS,
                prob_thresholds = DEFAULT_PROB_THRESHOLDS,
                froc_iou_threshold: float = 0.1) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
        match_pred_with_gt(all_gt_nodules, all_pred_nodules)
        
        metric_output_txt = open(os.path.join(output_dir, 'metrics.txt'), 'w')
        metric_bst_output_txt = open(os.path.join(output_dir, 'metrics_bst.txt'), 'w')
        header = 'iou_threshold,prob_threshold,sensitivity,precision,f1_score,tp,fp,fn\n'
        metric_template = '{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:d},{:d},{:d}\n'
        metric_output_txt.write(header)
        metric_bst_output_txt.write(header)
        
        key_template = '{:.3f}_{:.3f}'
        all_metrics = dict()
        # Compute sensitivity, precision, f1_score based on the different iou and prob thresholds
        for iou_threshold, prob_threshold in product(iou_thresholds, prob_thresholds):
            key = key_template.format(iou_threshold, prob_threshold)
            
            # Collect the number of true positive, false positive and false negative for each scan
            scan_tp_fp_fn = []
            for series_name in all_gt_nodules.keys():
                tp, fp, fn = 0, 0, 0
                gt_nodules = all_gt_nodules[series_name]
                pred_nodules = all_pred_nodules.get(series_name, list)
                for gt_nodule in gt_nodules:
                    if ((gt_nodule.iou >= iou_threshold) or (iou_threshold == 0 and gt_nodule.iou > 0)) \
                        and (gt_nodule.pred_prob >= prob_threshold):
                        tp += 1
                    else:
                        fn += 1
                for pred_nodule in pred_nodules:
                    if pred_nodule.iou < iou_threshold and pred_nodule.pred_prob >= prob_threshold:
                        fp += 1
                scan_tp_fp_fn.append([tp, fp, fn])
            
            scan_tp_fp_fn = np.array(scan_tp_fp_fn)
            tp_fp_fn = scan_tp_fp_fn.sum(axis=0)
            tp, fp, fn = tp_fp_fn
            sensitivity, precision, f1_score = compute_metrics(tp, fp, fn)
            result = [sensitivity, precision, f1_score, tp, fp, fn]
            metric_output_txt.write(metric_template.format(iou_threshold, prob_threshold, sensitivity, precision, f1_score, tp, fp, fn))
            
            # Bootstrapping sampling to get the confidence interval
            bst_result = [] 
            for i in range(self.bootstrapping_times):
                sample_indices = np.random.choice(len(scan_tp_fp_fn), len(scan_tp_fp_fn), replace=True)
                tp_fp_fn = scan_tp_fp_fn[sample_indices].sum(axis=0)
                tp, fp, fn = tp_fp_fn
                sensitivity, precision, f1_score = compute_metrics(tp, fp, fn)
                bst_result.append([sensitivity, precision, f1_score, tp, fp, fn])
            
            bst_result = np.array(bst_result)
            bst_result_mean = np.mean(bst_result, axis=0)
            line = metric_template.format(iou_threshold, prob_threshold, *bst_result_mean.tolist())
            metric_bst_output_txt.write(line)
            logger.info(f'====> iou_threshold:{iou_threshold:.3f}, prob_threshold:{prob_threshold:.3f} sensitivity:{bst_result_mean[0]:.3f}, precision:{bst_result_mean[1]:.3f}, f1_score:{bst_result_mean[2]:.3f}')
            all_metrics[key] = [result, bst_result_mean]
        # Compute FROC
        inter_btp_mean, inter_points = compute_froc(all_gt_nodules = all_gt_nodules,
                                                    all_pred_nodules = all_pred_nodules, 
                                                    froc_fp_ratios = froc_fp_ratios,
                                                    froc_iou_threshold = froc_iou_threshold,
                                                    bootstrapping_times=self.bootstrapping_times)
        sens_interp_btp_mean, prec_interp_btp_mean, f1_interp_btp_mean = inter_btp_mean
        sens_points, prec_points, f1_points = inter_points
        
        inter_btp_mean = {'sens': sens_interp_btp_mean, 'prec': prec_interp_btp_mean, 'f1': f1_interp_btp_mean}
        inter_points = {'sens': sens_points, 'prec': prec_points, 'f1': f1_points}
        
        return all_metrics, inter_btp_mean, inter_points
        
    def run(self, pred_nodules: Dict[str, List[NoduleFinding]], output_dir: str):
        if self.max_num_of_candidate > 0:
            for series_name, nodules in pred_nodules.items():
                # If number of candidate in a series of prediction is larger than max_num_of_nodule_candidate_in_series, keep 
                # the top max_num_of_nodule_candidate_in_series candidates
                if len(nodules) > self.max_num_of_candidate:
                    # sort the candidates by their probability
                    pred_nodules[series_name] = sorted(nodules, key=lambda x: x.pred_prob, reverse=True)[:self.max_num_of_candidate]
                    sorted_nodules = sorted(nodules.items(), key=lambda x: x[1].pred_prob, reverse=True) 
                
                    keep_nodules = dict()
                    for i in range(self.max_num_of_candidate):
                        keep_nodules[sorted_nodules[i][0]] = sorted_nodules[i][1]
                    
                    pred_nodules[series_name] = keep_nodules
            
        all_metrics, inter_btp_mean, inter_points = self.evaluate(all_gt_nodules = self.gt_nodules.copy(),
                                                                all_pred_nodules = pred_nodules.copy(),
                                                                output_dir = output_dir,
                                                                froc_iou_threshold = self.froc_iou_threshold)      
        return all_metrics, inter_btp_mean, inter_points