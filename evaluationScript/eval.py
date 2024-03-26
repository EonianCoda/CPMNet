# coding:utf-8
import os
import logging
from typing import Tuple, List, Any, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np


import math
from collections import defaultdict
from dataload.utils import load_label, load_series_list, gen_label_path, ALL_CLS, ALL_RAD, ALL_LOC, NODULE_SIZE, compute_bbox3d_iou
from functools import cmp_to_key
from .markdown_writer import MarkDownWriter
from .nodule_finding import NoduleFinding
from .nodule_typer import NoduleTyper

logger = logging.getLogger(__name__)

# Evaluation settings
PERFORMBOOTSTRAPPING = True
NUMBEROFBOOTSTRAPSAMPLES = 1000
BOTHERNODULESASIRRELEVANT = True
CONFIDENCE = 0.95

SERIESUID = 'seriesuid'
COORDX = 'coordX'
COORDY = 'coordY'
COORDZ = 'coordZ'
WW = 'w'
HH = 'h'
DD = 'd'
NODULE_TYPE = 'nodule_type'
CADProbability_label = 'probability'

# plot settings
FROC_MINX = 0.125 # Mininum value of x-axis of FROC curve
FROC_MAXX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

NODULE_TYPE_TEMPLATE = '{:20s}: Recall={:.3f}, Precision={:.3f}, F1={:.3f}, TP={:4d}, FP={:4d}, FN={:4d}'

def gen_bootstrap_set(scan_to_cands_dict: Dict[str, np.ndarray], seriesUIDs_np: np.ndarray) -> np.ndarray:
    """
    Generates bootstrapped version of set(bootstrapping is sampling method with replacement)
    """
    num_scans = seriesUIDs_np.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_indices = np.random.randint(num_scans, size=num_scans)
    seriesUIDs_rand = seriesUIDs_np[rand_indices]
    
    # get a new list of candidates
    candidatesExists = False
    for series_uid in seriesUIDs_rand:
        if series_uid not in scan_to_cands_dict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scan_to_cands_dict[series_uid])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scan_to_cands_dict[series_uid]),axis = 1)

    return candidates

def compute_FROC_bootstrap(FROC_gt_list: List[float],
                          FROC_prob_list: List[float],
                          FROC_series_uids: List[str],
                          seriesUIDs: List[str],
                          FROC_is_FN_list: List[bool],
                          numberOfBootstrapSamples: int = 1000, 
                          confidence = 0.95):
    
    set1 = np.concatenate(([FROC_gt_list], [FROC_prob_list], [FROC_is_FN_list]), axis=0) # 3 x N, N is the number of candidates
    fp_scans_list = []
    sens_list = []
    precision_list = []
    thresholds_list = []
    
    FROC_series_uids_np = np.asarray(FROC_series_uids)
    seriesUIDs_np = np.asarray(seriesUIDs)
    # Make a dict with all candidates of all scans
    scan_to_cands_dict = {}
    for i in range(len(FROC_series_uids_np)):
        series_uid = FROC_series_uids_np[i]
        candidate = set1[:, i:i+1]

        if series_uid not in scan_to_cands_dict:
            scan_to_cands_dict[series_uid] = np.copy(candidate)
        else:
            scan_to_cands_dict[series_uid] = np.concatenate((scan_to_cands_dict[series_uid],candidate),axis = 1)
    
    for i in range(numberOfBootstrapSamples):
        # Generate a bootstrapped set
        btpsamp = gen_bootstrap_set(scan_to_cands_dict, seriesUIDs_np)
        fp_scans, sens, precisions, thresholds = compute_FROC(btpsamp[0,:], btpsamp[1,:],len(seriesUIDs_np),btpsamp[2,:])
    
        fp_scans_list.append(fp_scans)
        sens_list.append(sens)
        precision_list.append(precisions)
        thresholds_list.append(thresholds)

    # compute statistic
    all_fp_scans = np.linspace(FROC_MINX, FROC_MAXX, num=10000) # shape (10000,)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    interp_precisions = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    interp_thresholds = np.zeros((numberOfBootstrapSamples, len(all_fp_scans)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fp_scans, fp_scans_list[i], sens_list[i])
        interp_precisions[i,:] = np.interp(all_fp_scans, fp_scans_list[i], precision_list[i])
        interp_thresholds[i,:] = np.interp(all_fp_scans, fp_scans_list[i], thresholds_list[i])
    # compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence = confidence)
    prec_mean, prec_lb, prec_up = compute_mean_ci(interp_precisions, confidence = confidence)
    thresholds_mean, thresholds_lb, thresholds_up = compute_mean_ci(interp_thresholds, confidence = confidence)
    
    return (all_fp_scans, thresholds_mean), (sens_mean, sens_lb, sens_up), (prec_mean, prec_lb, prec_up)

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]), dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]), dtype = 'float32') # lower bound
    sens_up   = np.zeros((interp_sens.shape[1]), dtype = 'float32') # upper bound
    
    Pz = (1.0 - confidence) / 2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(np.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(np.floor((1.0-Pz)*len(vec)))]

    return sens_mean, sens_lb, sens_up

def compute_FROC(FROC_is_pos_list: List[float], 
                FROC_prob_list: List[float], 
                total_num_of_series: int,
                FROC_is_FN_list: List[bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        FROC_is_pos_list:
            each element is 1 if the sample is positive, 0 otherwise
        FROC_prob_list: 
            each element is the probability of the corresponding sample
        total_num_of_series: 
            total number of series
        FROC_is_FN_list:
            each element is True if the sample is a false negative, False otherwise
    Returns:
        A tuple of (fp_per_scan, sens, precisions, thresholds)
    """
    # Remove FNs
    FROC_is_pos_list_local = []
    FROC_prob_list_local = []
    for i in range(len(FROC_is_FN_list)):
        if FROC_is_FN_list[i] == False:
            FROC_is_pos_list_local.append(FROC_is_pos_list[i])
            FROC_prob_list_local.append(FROC_prob_list[i])
    
    num_of_detected_pos = sum(FROC_is_pos_list_local)
    num_of_gt_pos = sum(FROC_is_pos_list)
    num_of_cand = len(FROC_prob_list_local)
    
    if num_of_detected_pos == 0:
        fp_ratio = np.zeros((5,), dtype=np.float32)
        tp_ratio = np.zeros((5,), dtype=np.float32)
        thresholds = np.array([np.inf, 0.8, 0.4, 0.2, 0.1])
    else:
        fp_ratio, tp_ratio, thresholds = skl_metrics.roc_curve(FROC_is_pos_list_local, FROC_prob_list_local)
    
    # Compute false positive per scan along different thresholds
    if sum(FROC_is_pos_list) == len(FROC_is_pos_list): #  Handle border case when there are no false positives and ROC analysis give nan values.
        fp_per_scans = np.zeros(len(fp_ratio))
    else:
        fp_per_scans = fp_ratio * (num_of_cand - num_of_detected_pos) / total_num_of_series # shape (len(fp_ratio),)
    
    sens = (tp_ratio * num_of_detected_pos) / num_of_gt_pos # sensitivity
    precisions = (tp_ratio * num_of_detected_pos) / np.maximum(1, tp_ratio * num_of_detected_pos + fp_ratio * (num_of_cand - num_of_detected_pos)) # precision
    return fp_per_scans, sens, precisions, thresholds

def evaluateCAD(seriesUIDs: List[str], 
                results_path: str,
                output_dir: str,
                all_gt_nodules: Dict[str, List[NoduleFinding]],
                max_num_of_nodule_candidate_in_series: int = -1,
                iou_threshold = 0.1,
                fixed_prob_threshold = 0.8):
    """
    function to evaluate a CAD algorithm
    """
    # compute FROC
    fps, sens, precisions, thresholds = compute_FROC(FROC_is_pos_list = FROC_is_pos_list, 
                                                    FROC_prob_list = FROC_prob_list, 
                                                    total_num_of_series = len(seriesUIDs), 
                                                    FROC_is_FN_list = FROC_is_FN_list)
    
    if PERFORMBOOTSTRAPPING:  # True
        (fps_bs_itp, thresholds_mean), senstitivity_info, precision_info = compute_FROC_bootstrap(FROC_gt_list = FROC_is_pos_list,
                                                                                                FROC_prob_list = FROC_prob_list,
                                                                                                FROC_series_uids = FROC_series_uids,
                                                                                                seriesUIDs = seriesUIDs,
                                                                                                FROC_is_FN_list = FROC_is_FN_list,
                                                                                                numberOfBootstrapSamples = NUMBEROFBOOTSTRAPSAMPLES, 
                                                                                                confidence = CONFIDENCE)
        sens_bs_mean, sens_bs_lb, sens_bs_up = senstitivity_info
        prec_bs_mean, prec_bs_lb, prec_bs_up = precision_info
        f1_score_mean = 2 * prec_bs_mean * sens_bs_mean / np.maximum(1e-6, prec_bs_mean + sens_bs_mean)
        
        best_f1_index = np.argmax(f1_score_mean)
        best_f1_threshold = thresholds_mean[best_f1_index]
        best_f1_sens = sens_bs_mean[best_f1_index]
        best_f1_prec = prec_bs_mean[best_f1_index]
        best_f1_score = f1_score_mean[best_f1_index]
        logger.info('Best F1 score: {:.4f} at threshold: {:.3f}, Sens: {:.3f}, Prec: {:.3f}'.format(best_f1_score, best_f1_threshold, best_f1_sens, best_f1_prec))
        # Write FROC curve
        with open(os.path.join(output_dir, "froc_{}.txt".format(iou_threshold)), 'w') as f:
            f.write("FPrate,Sensivity,Precision,f1_score,Threshold\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.5f,%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], prec_bs_mean[i], f1_score_mean[i], thresholds_mean[i]))
    # Write FROC vectors to disk as well
    with open(os.path.join(output_dir, "froc_gt_prob_vectors_{}.csv".format(iou_threshold)), 'w') as f:
        f.write("is_pos, prob\n")
        for i in range(len(FROC_is_pos_list)):
            f.write("%d,%.4f\n" % (FROC_is_pos_list[i], FROC_prob_list[i]))

    fps_itp = np.linspace(FROC_MINX, FROC_MAXX, num=10001)
    
    sens_itp = np.interp(fps_itp, fps, sens)
    prec_itp = np.interp(fps_itp, fps, precisions)
    
    sens_points = []
    prec_points = []
    
    if PERFORMBOOTSTRAPPING: # True
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(output_dir, "froc_bootstrapping_{}.csv".format(iou_threshold)), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
            FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            total_sens = 0
            nodule_output_file.write('-'*20 + '\n')
            nodule_output_file.write("FP/Scan, Sensitivity, Precision\n")
            for fp_point in FPS:
                index = np.argmin(abs(fps_bs_itp - fp_point))
                nodule_output_file.write('{:.3f}, {:.3f}, {:.3f}\n'.format(fp_point, sens_bs_mean[index], prec_bs_mean[index]))
                sens_points.append(sens_bs_mean[index])
                prec_points.append(prec_bs_mean[index])
                total_sens += sens_bs_mean[index]
            nodule_output_file.write("\n")
            nodule_output_file.write("Froc_mean = {:.2f}\n".format(total_sens / len(FPS)))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(total_num_of_nodules) > 0:
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, lw=2)
        if PERFORMBOOTSTRAPPING:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_MINX
        xmax = FROC_MAXX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.title('FROC performance')
        
        if bLogPlot:
            plt.xscale('log')
            ax.xaxis.set_major_locator(plt.FixedLocator([0.125,0.25,0.5,1,2,4,8]))
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "froc_{}.png".format(iou_threshold)), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points), (fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score), (best_f1_score, best_f1_threshold)
    
def compute_sphere_volume(diameter: float) -> float:
    if diameter == 0:
        return 0
    elif diameter == -1:
        return 100000000
    else:
        radius = diameter / 2
        return 4/3 * math.pi * radius**3

def compute_nodule_volume(w: float, h: float, d: float) -> float:
    # We assume that the shape of the nodule is approximately spherical. The original formula for calculating its 
    # volume is 4/3 * math.pi * (w/2 * h/2 * d/2) = 4/3 * math.pi * (w * h * d) / 8. However, when comparing the 
    # segmentation volume with the nodule volume, we discovered that using 4/3 * math.pi * (w * h * d) / 8 results 
    # in a nodule volume smaller than the segmentation volume. Therefore, we opted to use 4/3 * math.pi * (w * h * d) / 6 
    # to calculate the nodule volume.
    volume = 4/3 * math.pi * ((w * h * d) / 6)
    return volume
class FROC:
    def __init__(self, 
                 nodule_types: List[str] = ['benign', 'probably_benign', 'probably_suspicious', 'suspicious']):
        self.is_pos_list = []
        self.is_FN_list = []
        self.prob_list = []
        self.series_names = []
        
        self.nodules_list = [] # nodules list contains ground truth nodules and candidates
        self.canidates_list = [] # candidates list only contains candidates, if it is FN, then it is None
    
        self.tp_count = 0
        self.fp_count = 0
        self.fn_count = 0
        
        self.nodule_types = nodule_types
        
    def add(self, is_pos: bool, is_FN: bool, prob: float, series_name: str, nodule: NoduleFinding):
        self.is_pos_list.append(is_pos)
        self.is_FN_list.append(is_FN)
        self.prob_list.append(prob)
        self.series_names.append(series_name)
        
        if is_FN:
            self.fn_count += 1
        elif is_pos:
            self.tp_count += 1
        else:
            self.fp_count += 1
        
        self.nodules_list.append(nodule)
        if is_FN:
            self.canidates_list.append(None)
        else:
            self.canidates_list.append(nodule)

    def get_metrics(self, prob_threshold: float = 0.0) -> Tuple[np.ndarray, float, float]:
        ##TODO make prob_threshold into list
        logger.info('Prob threshold: {:.3f}'.format(prob_threshold))
        classified_metrics = dict()
        series_metric = dict()
        for nodule_type in self.nodule_types:
            classified_metrics[nodule_type] = np.zeros(3, dtype=np.int32) # tp, fp, fn
        for is_pos, is_FN, prob, nodule, series_name in zip(self.is_pos_list, self.is_FN_list, self.prob_list, self.nodules_list, self.series_names):
            if series_name not in series_metric:
                series_metric[series_name] = np.zeros(3, dtype=np.int32)
            
            if is_FN or (is_pos and prob < prob_threshold): # fn
                classified_metrics[nodule.nodule_type][2] += 1
                series_metric[series_name][2] += 1
            elif is_pos and prob >= prob_threshold: # tp
                classified_metrics[nodule.nodule_type][0] += 1
                series_metric[series_name][0] += 1
            elif not is_pos and prob >= prob_threshold: # fp
                classified_metrics[nodule.nodule_type][1] += 1
                series_metric[series_name][1] += 1
        
        # Compute metrics for all types
        classified_metrics['all'] = np.zeros(3, dtype=np.int32)
        for nodule_type in self.nodule_types:
            classified_metrics['all'] += classified_metrics[nodule_type]
        
        for nodule_type in self.nodule_types + ['all']:
            tp, fp, fn = classified_metrics[nodule_type]
            recall = tp / max(tp + fn, 1e-6)
            precision = tp / max(tp + fp, 1e-6)
            f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
            logger.info(NODULE_TYPE_TEMPLATE.format(nodule_type, recall, precision, f1_score, tp, fp, fn))
    
            classified_metrics[nodule_type] = {'recall': recall, 
                                               'precision': precision, 
                                               'f1_score': f1_score, 
                                               'tp': tp, 
                                               'fp': fp, 
                                               'fn': fn}
    
        # Compute metrics for each series
        recall_series_based = []
        for series_name, metrics in series_metric.items():
            tp, fp, fn = metrics
            if tp + fn == 0:
                recall_series_based.append(-1)
            else:
                recall_series_based.append(tp / max(tp + fn, 1e-6))
        recall_series_based = np.array(recall_series_based)
        
        recall_remove_health_series_based = np.mean(recall_series_based[recall_series_based != -1])
        recall_series_based[recall_series_based == -1] = 1
        recall_series_based = np.mean(recall_series_based)
        
        logger.info('Recall(series_based): {:.3f}'.format(recall_series_based))
        logger.info('Recall(remove_healthy_series_based): {:.3f}'.format(recall_remove_health_series_based))
                
        return classified_metrics, recall_series_based, recall_remove_health_series_based

class Evaluation:
    def __init__(self, 
                 series_list_path: str,
                 image_spacing: Tuple[float, float, float],
                 nodule_type_diameters: Dict[str, float],
                 prob_threshold: float = 0.65,
                 iou_threshold: float = 0.1,
                 nodule_size_mode = 'dhw', # or 'seg_size'
                 nodule_min_d: int = 0,
                 nodule_min_size: int = 0):
        
        self.image_spacing = np.array(image_spacing)
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.voxel_volume = np.prod(self.image_spacing)
        
        self.nodule_min_d = nodule_min_d
        self.nodule_min_size = nodule_min_size
        self.nodule_type_diameters = nodule_type_diameters
        self.nodule_size_mode = nodule_size_mode
        
        # Initialize nodule typer and collect ground truth nodules
        self.nodule_typer = NoduleTyper(nodule_type_diameters, image_spacing)
        self._init_nodule_type_volumes()
        self._collect_gt(series_list_path)
        
        self.sorted_nodule_types = ['benign', 'probably_benign', 'probably_suspicious', 'suspicious']
    
    def _init_nodule_type_volumes(self):
        """
        Initializes the nodule type volumes dictionary.

        This method calculates and stores the volumes of nodules for each nodule type based on their diameters.
        """
        self.nod_type_volumes = {}
        for key in self.nodule_type_diameters:
            min_diameter, max_diameter = self.nodule_type_diameters[key]
            self.nod_type_volumes[key] = [round(compute_sphere_volume(min_diameter) / self.voxel_volume),
                                           round(compute_sphere_volume(max_diameter) / self.voxel_volume)]
    
    def _collect_gt(self, series_list_path: str):
        """
        Collects all ground truth nodules from the series list file and stores them in a dictionary.
        """
        self.all_gt_nodules = defaultdict(list)
        self.num_of_all_gt_nodules = 0
        for info in load_series_list(series_list_path):
            series_dir = info[0]
            series_name = info[1]

            label_path = gen_label_path(series_dir, series_name)
            label = load_label(label_path, self.image_spacing, self.nodule_min_d, self.nodule_min_size)
            
            # If there are no nodules in the series, skip it
            if len(label[ALL_LOC]) == 0:
                self.all_gt_nodules[series_name] = []
                continue
            
            self.num_of_all_gt_nodules = self.num_of_all_gt_nodules + len(label[ALL_LOC])
            for ctrs, dhws, seg_size in zip(label[ALL_LOC], label[ALL_RAD], label[NODULE_SIZE]):
                ctr_z, ctr_y, ctr_x = ctrs
                d, h, w = dhws
                d, h, w = d / self.image_spacing[0], h / self.image_spacing[1], w / self.image_spacing[2]
                self.all_gt_nodules[series_name].append(self._build_nodule_finding(series_name, ctr_x, ctr_y, ctr_z, w, h, d, nodule_size=seg_size, is_gt=True))
            
    def _build_nodule_finding(self, series_name: str, ctr_x: float, ctr_y: float, ctr_z: float, 
                              w: float, h: float, d: float, nodule_size: float = None, **kwargs) -> NoduleFinding:
        if nodule_size is not None and self.nodule_size_mode == 'seg_size':
            nodule_type = self.nodule_typer.get_nodule_type_by_seg_size(nodule_size)
        else:
            nodule_type = self.nodule_typer.get_nodule_type_by_dhw(d, h, w)
        
        return NoduleFinding(series_name, ctr_x, ctr_y, ctr_z, w, h, d, nodule_type = nodule_type, **kwargs)
    
    def evaluation(self, preds: List[List[Any]], save_dir: str):
        """
        Args:
            preds: list of predicted nodules in format of [series_name, ctr_x, ctr_y, ctr_z, w, h, d, prob]
        """
        # Collect predicted nodules
        num_of_all_pred_cands = len(preds)
        all_pred_cands = defaultdict(list)
        for pred in preds:
            series_name, ctr_x, ctr_y, ctr_z, prob, w, h, d = pred
            all_pred_cands[series_name].append(self._build_nodule_finding(series_name, ctr_x, ctr_y, ctr_z, w, h, d, prob = prob))
        if len(all_pred_cands) > len(self.all_gt_nodules):
            raise ValueError('Number of predicted series is {} which is larger than the number of ground truth series {}'.format(len(all_pred_cands), len(self.all_gt_nodules)))
        
        # Match predicted nodules with ground truth nodules
        # tp_count, fp_count, fn_count = 0, 0, 0
        FN_gt_nodules = []
        all_series_names = list(set(list(self.all_gt_nodules.keys()) + list(all_pred_cands.keys())))
        froc = FROC()
        
        for series_name in all_series_names:
            pred_cands = all_pred_cands[series_name]
            gt_nodules = self.all_gt_nodules[series_name]
            
            # Compute the iou between all ground truth nodules and all predicted nodules
            gt_bboxes = np.array([gt_nodule.get_box() for gt_nodule in gt_nodules]) # [M, 2, 3], 3 is for [x, y, z]
            pred_bboxes = np.array([cand.get_box() for cand in pred_cands]) # [N, 2, 3], 3 is for [x, y, z]
            
            if len(gt_bboxes) != 0 and len(pred_bboxes) != 0:
                all_ious = compute_bbox3d_iou(gt_bboxes, pred_bboxes) # [M, N]
            else:
                all_ious = np.zeros((0,))
            
            # Compute TP and FN
            if len(gt_nodules) != 0:
                matched_masks = (all_ious >= self.iou_threshold)
                for gt_idx, gt_nodule in enumerate(gt_nodules):
                    ious = all_ious[gt_idx]
                    match_mask = matched_masks[gt_idx]
                    if np.any(match_mask): # TP
                        # Select the candidate with the highest probability
                        match_cand_ids = np.where(match_mask == True)[0]
                        max_prob_idx = np.argmax([pred_cands[cand_id].prob for cand_id in match_cand_ids])
                        match_cand = pred_cands[match_cand_ids[max_prob_idx]]
                        max_prob = match_cand.prob
                        
                        max_prob_iou = ious[match_cand_ids[max_prob_idx]] # iou of the matched candidate with the highest probability
                        match_cand.set_match(max_prob_iou, gt_nodule)
                        gt_nodule.set_match(max_prob_iou, match_cand)
                        gt_nodule.prob = max_prob
                        froc.add(is_pos=True, is_FN=False, prob=match_cand.prob, series_name=series_name, nodule=gt_nodule)
                    else: # FN
                        FN_gt_nodules.append(gt_nodule)
                        gt_nodule.set_match(np.max(ious), None)
                        froc.add(is_pos=True, is_FN=True, prob=-1, series_name=series_name, nodule=gt_nodule)
            
            # Compute FP
            if len(pred_cands) != 0:
                if len(all_ious) != 0:
                    pred_ious = np.max(all_ious, axis=0)
                    for iou, cand in zip(pred_ious, pred_cands):
                        if iou >= self.iou_threshold:
                            continue
                        cand.set_match(iou, None)
                        froc.add(is_pos=False, is_FN=False, prob=cand.prob, series_name=series_name, nodule=cand)
                else:
                    for cand in pred_cands:
                        cand.set_match(0, None)
                        froc.add(is_pos=False, is_FN=False, prob=cand.prob, series_name=series_name, nodule=cand)

        self._write_predicitions(save_dir, froc)
        self._write_FN_csv(FN_gt_nodules, save_dir)
        classified_metrics, recall_series_based, recall_remove_health_series_based = froc.get_metrics(self.prob_threshold)
        self._write_stats(froc, save_dir, classified_metrics, recall_series_based, recall_remove_health_series_based, num_of_all_pred_cands)
        ##TODO: refactor this part
        # compute FROC
        FROC_is_pos_list = froc.is_pos_list
        FROC_prob_list = froc.prob_list
        seriesUIDs = list(set(froc.series_names))
        FROC_is_FN_list = froc.is_FN_list
        FROC_series_uids = froc.series_names
        
        fps, sens, precisions, thresholds = compute_FROC(FROC_is_pos_list = FROC_is_pos_list, 
                                                        FROC_prob_list = FROC_prob_list, 
                                                        total_num_of_series = len(seriesUIDs), 
                                                        FROC_is_FN_list = FROC_is_FN_list)
        
        if PERFORMBOOTSTRAPPING:  # True
            (fps_bs_itp, thresholds_mean), senstitivity_info, precision_info = compute_FROC_bootstrap(FROC_gt_list = FROC_is_pos_list,
                                                                                                    FROC_prob_list = FROC_prob_list,
                                                                                                    FROC_series_uids = FROC_series_uids,
                                                                                                    seriesUIDs = seriesUIDs,
                                                                                                    FROC_is_FN_list = FROC_is_FN_list,
                                                                                                    numberOfBootstrapSamples = NUMBEROFBOOTSTRAPSAMPLES, 
                                                                                                    confidence = CONFIDENCE)
            sens_bs_mean, sens_bs_lb, sens_bs_up = senstitivity_info
            prec_bs_mean, prec_bs_lb, prec_bs_up = precision_info
            f1_score_mean = 2 * prec_bs_mean * sens_bs_mean / np.maximum(1e-6, prec_bs_mean + sens_bs_mean)
            
            best_f1_index = np.argmax(f1_score_mean)
            best_f1_threshold = thresholds_mean[best_f1_index]
            best_f1_sens = sens_bs_mean[best_f1_index]
            best_f1_prec = prec_bs_mean[best_f1_index]
            best_f1_score = f1_score_mean[best_f1_index]
            logger.info('Best F1 score: {:.4f} at threshold: {:.3f}, Sens: {:.3f}, Prec: {:.3f}'.format(best_f1_score, best_f1_threshold, best_f1_sens, best_f1_prec))
            # Write FROC curve
            with open(os.path.join(save_dir, "froc_{}.txt".format(self.iou_threshold)), 'w') as f:
                f.write("FPrate,Sensivity,Precision,f1_score,Threshold\n")
                for i in range(len(fps_bs_itp)):
                    f.write("%.5f,%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], prec_bs_mean[i], f1_score_mean[i], thresholds_mean[i]))
        # Write FROC vectors to disk as well
        with open(os.path.join(save_dir, "froc_gt_prob_vectors_{}.csv".format(self.iou_threshold)), 'w') as f:
            f.write("is_pos, prob\n")
            for i in range(len(FROC_is_pos_list)):
                f.write("%d,%.4f\n" % (FROC_is_pos_list[i], FROC_prob_list[i]))

        fps_itp = np.linspace(FROC_MINX, FROC_MAXX, num=10001)
        
        sens_itp = np.interp(fps_itp, fps, sens)
        prec_itp = np.interp(fps_itp, fps, precisions)
        
        sens_points = []
        prec_points = []
        
        if PERFORMBOOTSTRAPPING: # True
            # Write mean, lower, and upper bound curves to disk
            with open(os.path.join(save_dir, "froc_bootstrapping_{}.csv".format(self.iou_threshold)), 'w') as f:
                f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
                for i in range(len(fps_bs_itp)):
                    f.write("%.5f,%.5f,%.5f,%.5f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
                FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
                total_sens = 0
                # nodule_output_file.write('-'*20 + '\n')
                # nodule_output_file.write("FP/Scan, Sensitivity, Precision\n")
                for fp_point in FPS:
                    index = np.argmin(abs(fps_bs_itp - fp_point))
                    # nodule_output_file.write('{:.3f}, {:.3f}, {:.3f}\n'.format(fp_point, sens_bs_mean[index], prec_bs_mean[index]))
                    sens_points.append(sens_bs_mean[index])
                    prec_points.append(prec_bs_mean[index])
                    total_sens += sens_bs_mean[index]
                # nodule_output_file.write("\n")
                # nodule_output_file.write("Froc_mean = {:.2f}\n".format(total_sens / len(FPS)))
        else:
            fps_bs_itp = None
            sens_bs_mean = None
            sens_bs_lb = None
            sens_bs_up = None

        # create FROC graphs
        # if int(total_num_of_nodules) > 0:
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, lw=2)
        if PERFORMBOOTSTRAPPING:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_MINX
        xmax = FROC_MAXX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.title('FROC performance')
        
        if bLogPlot:
            plt.xscale('log')
            ax.xaxis.set_major_locator(plt.FixedLocator([0.125,0.25,0.5,1,2,4,8]))
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, "froc_{}.png".format(self.iou_threshold)), bbox_inches=0, dpi=300)

        fixed_tp = classified_metrics['all']['tp']
        fixed_fp = classified_metrics['all']['fp']
        fixed_fn = classified_metrics['all']['fn']
        fixed_recall = classified_metrics['all']['recall']
        fixed_precision = classified_metrics['all']['precision']
        fixed_f1_score = classified_metrics['all']['f1_score']
        plt.close()
        return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points), (fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score), (best_f1_score, best_f1_threshold)
    
    def _write_predicitions(self, save_dir: str, froc: FROC):
        def sort_by_x(n1, n2):
            if n1.ctr_x == n2.ctr_x:
                return 0
            elif n1.ctr_x > n2.ctr_x:
                return 1
            else:
                return -1
        
        save_path = os.path.join(save_dir, "predictions.csv")
        
        header = "series_name,ctr_x,ctr_y,ctr_z,w,h,d,prob, nodule_type, match_iou, is_gt\n"
        lines = [header]
        
        series_name_cands = defaultdict(list)
        for cand in froc.nodules_list:
            series_name = cand.series_name
            series_name_cands[series_name].append(cand)
        
        for series_name in sorted(series_name_cands.keys()):
            for cand in sorted(series_name_cands[series_name], key=cmp_to_key(sort_by_x)):
                series_name = cand.series_name
                ctr_x, ctr_y, ctr_z = cand.ctr_x, cand.ctr_y, cand.ctr_z
                w, h, d = cand.w, cand.h, cand.d
                prob = cand.prob
                nodule_type = cand.nodule_type
                match_iou = cand.match_iou
                is_gt = cand.is_gt
                lines.append("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{:.3f},{}\n".format(series_name, ctr_x, ctr_y, ctr_z, w, h, d, prob, nodule_type, match_iou, is_gt))
        
        with open(save_path, 'w') as f:
            f.writelines(lines)
            
    def _write_stats(self, froc: FROC, save_dir: str, classified_metrics: Dict[str, Dict[str, float]], recall_series_based: float, 
                     recall_remove_health_series_based: float, num_of_all_pred_cands: int): 
        
        def generate_prob_iou_table(stats: Dict[str, Dict[str, float]], nodule_type: str) -> List[str]:
            prob_and_iou_stats_template = "{:20s}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n"
            prob_stats_template = "{:20s}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n"
            
            number_text = str(stats['number'])
            if 'ratio' in stats:
                number_text = number_text + '<br>({:.1f}%)'.format(stats['ratio'] * 100)
            if len(stats['iou']) == 0:
                text = prob_stats_template.format(nodule_type, number_text, stats['prob']['mean'], stats['prob']['std'], stats['prob']['min'], stats['prob']['max'])
            else:
                text = prob_and_iou_stats_template.format(nodule_type, number_text, stats['prob']['mean'], stats['prob']['std'], stats['prob']['min'], stats['prob']['max'])
            
            return text.split(",")    
            
        stats_file_path = os.path.join(save_dir, "stats.md".format(self.iou_threshold))
        
        prob_and_iou_stats = self._analyze_prob_and_iou(froc)
        
        with MarkDownWriter(stats_file_path) as f:
            # Write the evaluation statistics
            f.write_header("Evaluation statistics", 1)
        
            f.write("TP: {}".format(froc.tp_count))
            f.write("FP: {}".format(froc.fp_count))
            f.write("FN: {}".format(froc.fn_count))
            
            f.write("Number of all predicted candidates: {}".format(num_of_all_pred_cands))
            f.write("Average number of candidates per series: {:.2f}".format(num_of_all_pred_cands / len(self.all_gt_nodules)))
            f.write("Number of all ground truth nodules: {}".format(self.num_of_all_gt_nodules))
            
            recall = froc.tp_count / max(froc.tp_count + froc.fn_count, 1e-6)
            f.write("Recall(Threshold = 0): {:.3f}".format(recall))
            f.write("Recall(series_based): {:.3f}".format(recall_series_based))
            f.write("Recall(remove_healthy_series_based): {:.3f}".format(recall_remove_health_series_based))
            
            # Write the statistics of the nodules
            f.write_header("Nodule type statistics", 1)
            header = "Nodule type, Recall, Precision, F1 score, TP, FP, FN\n"
            values = []
            for nodule_type, metrics in classified_metrics.items():
                text = NODULE_TYPE_TEMPLATE.format(nodule_type, metrics['recall'], metrics['precision'], metrics['f1_score'], metrics['tp'], metrics['fp'], metrics['fn'])
                values.append(text.split(","))
            f.write_table(header, values)
            
            # Write the statistics of the probability and iou of the matched candidates
            f.write_header("Prob and IoU statistics", 1)
            
            f.write_header("TP", 2)
            header = "Nodule type, Number, Prob(mean), Prob(std), Prob(min), Prob(max), IoU(mean), IoU(std), IoU(min), IoU(max)\n"
            values = []
            for nodule_type in self.sorted_nodule_types + ['all']:
                values.append(generate_prob_iou_table(prob_and_iou_stats['TP'][nodule_type], nodule_type))
            f.write_table(header, values)
            
            f.write_header("FP", 2)
            header = "Nodule type, Number, Prob(mean), Prob(std), Prob(min), Prob(max)\n"
            values = []
            for nodule_type in self.sorted_nodule_types + ['all']:
                values.append(generate_prob_iou_table(prob_and_iou_stats['FP'][nodule_type], nodule_type))
            f.write_table(header, values)
            
            f.write_header("FN", 2)
            header = "Nodule type, Number, Prob(mean), Prob(std), Prob(min), Prob(max)\n"
            values = []
            for nodule_type in self.sorted_nodule_types + ['all']:
                values.append(generate_prob_iou_table(prob_and_iou_stats['FN'][nodule_type], nodule_type))
            f.write_table(header, values)
            
    def _analyze_prob_and_iou(self, froc: FROC) -> Dict[str, Dict[str, float]]:
        """
        Analyze the mean, std, min, max of the probability and iou of the matched candidates of different nodule types.
        
        Returns: A dictionary containing the statistics of the probability and iou of the matched candidates of different nodule types.
        """
        def get_stats(nodules_by_type: Dict[str, List[NoduleFinding]], analyse_iou: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
            stats = {}
            total_number_of_nodules = sum([len(nodules) for nodules in nodules_by_type.values()])
            for nodule_type, nodules in nodules_by_type.items():
                probs = [n.prob for n in nodules]
                stats[nodule_type] = {'number': len(nodules),
                                      'ratio': len(nodules) / total_number_of_nodules,
                                        'prob': {'mean': np.mean(probs), 'std': np.std(probs), 'min': np.min(probs), 'max': np.max(probs)},
                                        'iou': {}}
                if analyse_iou:
                    ious = [n.match_iou for n in nodules]
                    stats[nodule_type]['iou'] = {'mean': np.mean(ious), 'std': np.std(ious), 'min': np.min(ious), 'max': np.max(ious)}
            # add All type
            probs = [n.prob for nodules in nodules_by_type.values() for n in nodules]
            stats['all'] = {'number': total_number_of_nodules,
                            'ratio': 1.0,
                            'prob': {'mean': np.mean(probs), 'std': np.std(probs), 'min': np.min(probs), 'max': np.max(probs)},
                            'iou': {}}
            if analyse_iou:
                ious = [n.match_iou for nodules in nodules_by_type.values() for n in nodules]
                stats['all']['iou'] = {'mean': np.mean(ious), 'std': np.std(ious), 'min': np.min(ious), 'max': np.max(ious)}
            
            return stats
            
        tps_by_type = defaultdict(list)
        fps_by_type = defaultdict(list)
        fns_by_type = defaultdict(list)
        for nod, cand in zip(froc.nodules_list, froc.canidates_list):
            if cand is None:
                fns_by_type[nod.nodule_type].append(nod)
            elif nod.is_gt:
                tps_by_type[nod.nodule_type].append(nod)
            else:
                fps_by_type[nod.nodule_type].append(nod)
        
        tp_stats = get_stats(tps_by_type, analyse_iou=True)
        fp_stats = get_stats(fps_by_type, analyse_iou=False)
        fn_stats = get_stats(fns_by_type, analyse_iou=False)
        
        return {'TP': tp_stats, 'FP': fp_stats, 'FN': fn_stats}
        
    def _write_FN_csv(self, FN_gt_nodules: List[NoduleFinding], save_dir: str):
        if save_dir is None:
            return
            
        FN_file_path = os.path.join(save_dir, "FN_{}.csv".format(self.iou_threshold))
        os.makedirs(os.path.dirname(FN_file_path), exist_ok=True)
        header = "seriesuid,ctr_x,ctr_y,ctr_z,w,h,d,nodule_type,match_iou\n"
        with open(FN_file_path, 'w') as f:
            f.write(header)
            for nodule in FN_gt_nodules:
                f.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{},{:.2f}\n".format(nodule.series_name, nodule.ctr_x, nodule.ctr_y, nodule.ctr_z, nodule.w, nodule.h, nodule.d, nodule.nodule_type, nodule.match_iou))