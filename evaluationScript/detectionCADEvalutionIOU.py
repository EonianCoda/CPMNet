# coding:utf-8
import os
import math
import logging
from typing import Tuple, List, Any, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np
from .NoduleFinding import NoduleFinding

from .tools import csvTools

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
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

def box_iou_union_3d(boxes1: List[float], boxes2: List[float], eps: float = 0.001) -> float:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.
    
    Args:
        boxes1: boxes [x1, x2, y1, y2, z1, z2]
                boxes2: boxes [x1, x2, y1, y2, z1, z2]
        eps: optional small constant for numerical stability
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        Tensor[N, M]: the nxM matrix containing the pairwise union
            values
    """
    vol1 = (boxes1[1] - boxes1[0]) * (boxes1[3] - boxes1[2]) * (boxes1[5] - boxes1[4])
    vol2 = (boxes2[1] - boxes2[0]) * (boxes2[3] - boxes2[2]) * (boxes2[5] - boxes2[4])

    x1 = max(boxes1[0], boxes2[0])
    x2 = min(boxes1[1], boxes2[1])
    y1 = max(boxes1[2], boxes2[2])
    y2 = min(boxes1[3], boxes2[3]) 
    z1 = max(boxes1[4], boxes2[4]) 
    z2 = min(boxes1[5], boxes2[5])

    inter = (max((x2 - x1), 0) * max((y2 - y1), 0) * max((z2 - z1), 0)) + eps
    union = (vol1 + vol2 - inter)
    return inter / union

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(np.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(np.floor((1.0-Pz)*len(vec)))]

    return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROC_gt_list: List[float],
                          FROC_prob_list: List[float],
                          FROC_series_uids: List[str],
                          FROCImList,
                          FROC_is_exclude_list: List[bool],
                          numberOfBootstrapSamples: int = 1000, 
                          confidence = 0.95):
    set1 = np.concatenate(([FROC_gt_list], [FROC_prob_list], [FROC_is_exclude_list]), axis=0)
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FROC_series_uids_np = np.asarray(FROC_series_uids)
    FROCImList_np = np.asarray(FROCImList)
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
        btpsamp = generateBootstrapSet(scan_to_cands_dict,FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

    return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROC_gt_list: List[float], 
                FROC_prob_list: List[float], 
                total_num_of_series: int,
                excludeList: List[bool]):
    # Remove excluded candidates
    FROC_gt_list_local = []
    FROC_prob_list_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROC_gt_list_local.append(FROC_gt_list[i])
            FROC_prob_list_local.append(FROC_prob_list[i])
    
    numberOfDetectedLesions = sum(FROC_gt_list_local)
    totalNumberOfLesions = sum(FROC_gt_list)
    totalNumberOfCandidates = len(FROC_prob_list_local)
    
    if numberOfDetectedLesions == 0:
        fpr = np.zeros((5,), dtype=np.float32)
        tpr = np.zeros((5,), dtype=np.float32)
        thresholds = np.array([np.inf, 0.8, 0.4, 0.2, 0.1])
    else:
        fpr, tpr, thresholds = skl_metrics.roc_curve(FROC_gt_list_local, FROC_prob_list_local)
    if sum(FROC_gt_list) == len(FROC_gt_list): #  Handle border case when there are no false positives and ROC analysis give nan values.
    #   print ("WARNING, this system has no false positives..")
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / total_num_of_series
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs: List[str], 
                results_path: str,
                output_dir: str,
                all_gt_nodules: Dict[str, List[NoduleFinding]],
                CADSystemName: str,
                maxNumberOfCADMarks=-1,
                iou_threshold = 0.1):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''
    logger.info('IOU threshold: {}'.format(iou_threshold))
    nodule_output_file = open(os.path.join(output_dir,'CADAnalysis_{}.txt'.format(iou_threshold)),'w')
    nodule_output_file.write("\n")
    nodule_output_file.write((60 * "*") + "\n")
    nodule_output_file.write("CAD Analysis: %s\n" % CADSystemName)
    nodule_output_file.write((60 * "*") + "\n")
    nodule_output_file.write("\n")

    pred_results = csvTools.readCSV(results_path)
    all_pred_cands = {}
    
    # collect candidates from prediction result file
    for series_uid in seriesUIDs:
        nodules = {}
        header = pred_results[0]
        i = 0
        for result in pred_results[1:]:
            nodule_seriesuid = result[header.index(SERIESUID)]
            
            if series_uid == nodule_seriesuid:
                nodule = get_nodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:  
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.items():  
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                
                keep_nodules = {}
                cur_num_of_keep_nodules = 0
                for keytemp, noduletemp in nodules.items():
                    if cur_num_of_keep_nodules >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        keep_nodules[keytemp] = noduletemp
                        cur_num_of_keep_nodules += 1

                nodules = keep_nodules  
        
        all_pred_cands[series_uid] = nodules  
        
    # open output files
    nodules_wo_candidate_file = open(os.path.join(output_dir, "nodulesWithoutCandidate_{}_{}.txt".format(CADSystemName, iou_threshold)), 'w')

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate
    # initialize some variables to be used in the loop
    cand_TPs, cand_FPs, cand_FNs, cand_TNs = 0, 0, 0, 0
    total_num_of_cands = 0
    total_num_of_nodules = 0
    double_candidates_ignored = 0
    num_of_irrelevant_cands = 0
    min_prob_value = -1000000000.0  # minimum value of a float
    FROC_gt_list = []
    FROC_prob_list = []
    FROC_series_uids = []
    FROC_is_exclude_list = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    FN_diameter = []
    seriesuid_save = []
    for series_uid in seriesUIDs:
        # get the candidates for this case
        pred_cands = all_pred_cands.get(series_uid, dict()) # A dict of candidates with key as candidateID and value as NoduleFinding
        total_num_of_cands += len(pred_cands.keys())  
        pred_cands_copy = pred_cands.copy() # make a copy in which items will be deleted

        # get the nodule annotations for this case
        gt_nodules = all_gt_nodules.get(series_uid, list()) # A list of NoduleFinding

        # - loop over each nodule annotation and determine whether it is covered by a candidate
        for gt_nodule in gt_nodules:
            if gt_nodule.state == "Included":
                total_num_of_nodules += 1 

            x, y, z = float(gt_nodule.coordX), float(gt_nodule.coordY), float(gt_nodule.coordZ)
            w, h, d = float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d)
            half_w, half_h, half_d = w/2, h/2, d/2

            is_found = False
            nodule_matches = []
            for cand_id, candidate in pred_cands.items():
                cand_x, cand_y, cand_z = float(candidate.coordX), float(candidate.coordY), float(candidate.coordZ)
                cand_w, cand_h, cand_d = float(candidate.w), float(candidate.h), float(candidate.d)
                cand_half_w, cand_half_h, cand_half_d = cand_w/2, cand_h/2, cand_d/2
                
                # [x1, x2, y1, y2, z1, z2]
                pred_box = [cand_x - cand_half_w, 
                        cand_x + cand_half_w, 
                        cand_y - cand_half_h, 
                        cand_y + cand_half_h, 
                        cand_z - cand_half_d, 
                        cand_z + cand_half_d]
                
                gt_box = [x - half_w, 
                          x + half_w, 
                          y - half_h, 
                          y + half_h, 
                          z - half_d, 
                          z + half_d]
                
                iou = box_iou_union_3d(pred_box, gt_box)
                if iou >= iou_threshold:
                    if (gt_nodule.state == "Included"):
                        is_found = True  
                        nodule_matches.append(candidate)  
                        if cand_id not in pred_cands_copy.keys():  
                            print("This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), series_uid, str(gt_nodule.id)))
                        else:
                            del pred_cands_copy[cand_id]
                    elif (gt_nodule.state == "Excluded"):  # an excluded nodule
                        if BOTHERNODULESASIRRELEVANT: # delete marks on excluded nodules so they don't count as false positives
                            if cand_id in pred_cands_copy.keys():
                                num_of_irrelevant_cands += 1  
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (series_uid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
                                del pred_cands_copy[cand_id]
            if len(nodule_matches) > 1:  # double detection
                double_candidates_ignored += (len(nodule_matches) - 1)  
            
            if gt_nodule.state == "Included":  
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if is_found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(nodule_matches)):
                        candidate = nodule_matches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROC_gt_list.append(1.0)  
                    FROC_prob_list.append(float(maxProb))  
                    FROC_series_uids.append(series_uid)
                    FROC_is_exclude_list.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%.9f" % (series_uid, gt_nodule.id, gt_nodule.coordX, gt_nodule.coordY, gt_nodule.coordZ, float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d), str(candidate.id), float(candidate.CADprobability)))
                    cand_TPs += 1  
                else:
                    cand_FNs += 1
                    FN_diameter.append([w, h, d])
                    seriesuid_save.append(series_uid)
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROC_gt_list.append(1.0) 
                    FROC_prob_list.append(min_prob_value)  
                    FROC_series_uids.append(series_uid)  
                    FROC_is_exclude_list.append(True)  
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%s" % (series_uid, gt_nodule.id, gt_nodule.coordX, gt_nodule.coordY, gt_nodule.coordZ, float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d), int(-1), "NA"))
                    nodules_wo_candidate_file.write("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s\n" % (series_uid, gt_nodule.id, gt_nodule.coordX, gt_nodule.coordY, gt_nodule.coordZ, float(gt_nodule.w), float(gt_nodule.h), float(gt_nodule.d), str(-1)))
                    
        # add all false positives to the vectors
        for cand_id, candidate3 in pred_cands_copy.items():  # candidates2此时剩下的是误报的，不在圆内
            cand_FPs += 1
            FROC_gt_list.append(0.0) 
            FROC_prob_list.append(float(candidate3.CADprobability))
            FROC_series_uids.append(series_uid)
            FROC_is_exclude_list.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (series_uid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

    if not (len(FROC_gt_list) == len(FROC_prob_list) and len(FROC_gt_list) == len(FROC_series_uids) and len(FROC_gt_list) == len(FROCtoNoduleMap) and len(FROC_gt_list) == len(FROC_is_exclude_list)):
        nodule_output_file.write("Length of FROC vectors not the same, this should never happen! Aborting..\n") 
    # Statistics that are computed
    nodule_output_file.write("Candidate detection results:\n")
    nodule_output_file.write("    True positives: %d\n" % cand_TPs)
    nodule_output_file.write("    False positives: %d\n" % cand_FPs)
    nodule_output_file.write("    False negatives: %d\n" % cand_FNs)
    nodule_output_file.write("    True negatives: %d\n" % cand_TNs)
    nodule_output_file.write("    Total number of candidates: %d\n" % total_num_of_cands)
    nodule_output_file.write("    Total number of nodules: %d\n" % total_num_of_nodules)

    nodule_output_file.write("    Ignored candidates on excluded nodules: %d\n" % num_of_irrelevant_cands)
    nodule_output_file.write("    Ignored candidates which were double detections on a nodule: %d\n" % double_candidates_ignored)
    if int(total_num_of_nodules) == 0:
        nodule_output_file.write("    Sensitivity: 0.0\n")
    else:
        nodule_output_file.write("    Sensitivity: %.9f\n" % (float(cand_TPs) / float(total_num_of_nodules)))
    nodule_output_file.write("    Average number of candidates per scan: %.9f\n" % (float(total_num_of_cands) / float(len(seriesUIDs))))
    nodule_output_file.write(
        "    FN_diammeter:\n")
    for idx, whd in enumerate(FN_diameter):
        nodule_output_file.write("    FN_%d: w:%.9f, h:%.9f, d:%.9f sericeuid:%s\n" % (idx+1, whd[0], whd[1], whd[2], seriesuid_save[idx]))
    # compute FROC
    fps, sens, thresholds = computeFROC(FROC_gt_list, FROC_prob_list, len(seriesUIDs), FROC_is_exclude_list)
    
    if PERFORMBOOTSTRAPPING:  # True
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROC_gt_list = FROC_gt_list,
                                                                              FROC_prob_list = FROC_prob_list,
                                                                              FROC_series_uids = FROC_series_uids,
                                                                              FROCImList = seriesUIDs,
                                                                              FROC_is_exclude_list = FROC_is_exclude_list,
                                                                              numberOfBootstrapSamples = NUMBEROFBOOTSTRAPSAMPLES, 
                                                                              confidence = CONFIDENCE)
    # Write FROC curve
    with open(os.path.join(output_dir, "froc_{}_{}.txt".format(CADSystemName, iou_threshold)), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
    
    # Write FROC vectors to disk as well
    with open(os.path.join(output_dir, "froc_gt_prob_vectors_{}_{}.csv".format(CADSystemName, iou_threshold)), 'w') as f:
        for i in range(len(FROC_gt_list)):
            f.write("%d,%.9f\n" % (FROC_gt_list[i], FROC_prob_list[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)  # FROC横坐标范围
    
    sens_itp = np.interp(fps_itp, fps, sens)  # FROC纵坐标
    forcs = []
    if PERFORMBOOTSTRAPPING: # True
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(output_dir, "froc_{}_bootstrapping_{}.csv".format(CADSystemName, iou_threshold)), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
            FPS = [0.125,0.25,0.5,1,2,4,8]
            Total_Sens = 0
            for fp_point in FPS:
                index = np.argmin(abs(fps_bs_itp-fp_point))
                nodule_output_file.write("\n")
                nodule_output_file.write(str(index))
                nodule_output_file.write(str(sens_bs_mean[index]))
                forcs.append(sens_bs_mean[index])
                Total_Sens += sens_bs_mean[index]
            print('Froc_mean:', Total_Sens / len(FPS))
            nodule_output_file.write("\n")
            nodule_output_file.write("Froc_mean")
            nodule_output_file.write(str(Total_Sens / len(FPS)))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(total_num_of_nodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if PERFORMBOOTSTRAPPING:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))
        
        if bLogPlot:
            plt.xscale('log')
            ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "froc_{}_{}.png".format(CADSystemName, iou_threshold)), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, forcs)
    
def get_nodule(annot: List[Any], 
               header: List[str],
               state = '') -> NoduleFinding:
    nodule = NoduleFinding()
    nodule.coordX = annot[header.index(COORDX)]
    nodule.coordY = annot[header.index(COORDY)]
    nodule.coordZ = annot[header.index(COORDZ)]

    nodule.w = annot[header.index(WW)]
    nodule.h = annot[header.index(HH)]
    nodule.d = annot[header.index(DD)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annot[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule

def collect_nodule_annotations(annotations: List[List[Any]],
                               annotations_excluded: List[List[Any]],
                               seriesUIDs: List[str]) -> Dict[str, List[NoduleFinding]]:
    """Collects all nodule annotations from the annotations file and returns them in a dictionary
    
    Args:
        annotations: list of annotations
        annotations_excluded: list of annotations that are excluded from analysis
        seriesUIDs: list of CT images in seriesuids
    Returns:
        Dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    """
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(SERIESUID)]
            
            if seriesuid == nodule_seriesuid:
                nodule = get_nodule(annotation, header, state = "Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(SERIESUID)]
            
            if seriesuid == nodule_seriesuid:
                nodule = get_nodule(annotation, header, state = "Excluded")
                nodules.append(nodule)
            
        allNodules[seriesuid] = nodules
        noduleCount += numberOfIncludedNodules
        noduleCountTotal += len(nodules)
    
    logger.info('Total number of included nodule annotations: {}'.format(noduleCount))
    logger.info('Total number of nodule annotations: {}'.format(noduleCountTotal))
    return allNodules
    
def collect(annot_path: str, 
            annot_excluded_path: str, 
            seriesuids_path: str) -> Tuple[Dict[str, List[NoduleFinding]], List[str]]:
    """Collects all nodule annotations from the annotations file and returns them in a dictionary
    Args:
        annot_path: path to annotations file
        annot_excluded_path: path to annotations_excluded file
        seriesuids_path: path to seriesuids file
    Returns:
        A tuple of:
        - Dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
        - List of seriesuids
    """
    annotations = csvTools.readCSV(annot_path) 
    annotations_excluded = csvTools.readCSV(annot_excluded_path)
    seriesUIDs_csv = csvTools.readCSV(seriesuids_path)
    
    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collect_nodule_annotations(annotations, annotations_excluded, seriesUIDs)  
    return allNodules, seriesUIDs
    
def noduleCADEvaluation(annot_path: str,
                        annot_excluded_path: str,
                        seriesuids_path: str,
                        results_path: str,
                        output_dir: str,
                        iou_threshold: float):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''
    logger.info('Annotation path: {}'.format(annot_path))
    all_gt_nodules, seriesUIDs = collect(annot_path, annot_excluded_path, seriesuids_path)
    
    out = evaluateCAD(seriesUIDs = seriesUIDs, 
                      results_path = results_path, 
                      output_dir = output_dir, 
                      all_gt_nodules = all_gt_nodules,
                      CADSystemName = os.path.splitext(os.path.basename(results_path))[0],
                      maxNumberOfCADMarks = 100, 
                      iou_threshold = iou_threshold)
    return out

if __name__ == '__main__':
    annotations_filename          = './annotations/annotations.csv'
    annotations_excluded_filename = './annotations/annotations_excluded.csv'
    seriesuids_filename           = './annotations/seriesuids.csv'
    results_filename              = './submission/sampleSubmission.csv'
    outputDir                     = './result'

    noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir)