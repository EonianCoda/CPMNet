import os
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from evaluationScript.eval_refactor import Evaluation

from utils.utils import get_progress_bar
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def convert_to_standard_output(output: np.ndarray, series_name: str) -> List[List[Any]]:
    """
    convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
    ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    """
    preds = []
    for j in range(output.shape[0]):
        preds.append([series_name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
    return preds

def val(args,
        model: nn.Module,
        detection_postprocess,
        val_loader: DataLoader,
        device: torch.device,
        image_spacing: List[float],
        # series_list_path: str,
        exp_folder: str,
        epoch: int = 0,
        batch_size: int = 16,
        nms_keep_top_k: int = 40,
        nodule_type_diameters : Dict[str, Tuple[float, float]] = None,
        min_d: int = 0,
        min_size: int = 0,
        nodule_size_mode: str = 'seg_size') -> Dict[str, float]:
    
    save_dir = os.path.join(exp_folder, 'annotation', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    if min_d != 0:
        logger.info('When validating, ignore nodules with depth less than {}'.format(min_d))
    if min_size != 0:
        logger.info('When validating, ignore nodules with size less than {}'.format(min_size))
    
    evaluator = Evaluation(series_list_path=val_loader.dataset.series_list_path, 
                           image_spacing=image_spacing,
                           nodule_type_diameters=nodule_type_diameters,
                           prob_threshold=args.val_fixed_prob_threshold,
                           iou_threshold = args.val_iou_threshold,
                           nodule_size_mode=nodule_size_mode,
                           nodule_min_d=min_d, 
                           nodule_min_size=min_size)
    
    
    model.eval()
    split_comber = val_loader.dataset.splitcomb
    all_preds = []
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to validate')
        
    with get_progress_bar('Validation', len(val_loader)) as progress_bar:
        for sample in val_loader:
            data = sample['split_images'].to(device, non_blocking=True, memory_format=memory_format)
            nzhws = sample['nzhws']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            outputlist = []
            
            for i in range(int(math.ceil(data.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > data.size(0):
                    end = data.size(0)
                input = data[i * batch_size:end]
                if args.val_mixed_precision:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output = model(input)
                            output = detection_postprocess(output, device=device)
                else:
                    with torch.no_grad():
                        output = model(input)
                        output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                outputlist.append(output.data.cpu().numpy())
            
            outputs = np.concatenate(outputlist, 0)
            
            start_idx = 0
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                nzhw = nzhws[i]
                output = split_comber.combine(outputs[start_idx:start_idx + n_split], nzhw)
                output = torch.from_numpy(output).view(-1, 8)
                # Remove the padding
                object_ids = output[:, 0] != -1.0
                output = output[object_ids]
                
                # NMS
                if len(output) > 0:
                    keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
                    output = output[keep]
                output = output.numpy()
            
                preds = convert_to_standard_output(output, series_names[i])  
                all_preds.extend(preds)
                start_idx += n_split
                
            progress_bar.update(1)
    # Save the results to csv
    # output_dir = os.path.join(annot_dir, f'epoch_{epoch}')
    # os.makedirs(output_dir, exist_ok=True)
    # header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    # df = pd.DataFrame(all_preds, columns=header)
    # pred_results_path = os.path.join(output_dir, 'predict_epoch_{}.csv'.format(epoch))
    # df.to_csv(pred_results_path, index=False)
    
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_out, fixed_out, (best_f1_score, best_f1_threshold) = evaluator.evaluation(preds=all_preds,
                                                                                   save_dir=save_dir)
    # froc_out, fixed_out, (best_f1_score, best_f1_threshold) = nodule_evaluation(annot_path = annot_path,
    #                                                                             series_uids_path = series_uids_path, 
    #                                                                             pred_results_path = pred_results_path,
    #                                                                             output_dir = output_dir,
    #                                                                             iou_threshold = args.val_iou_threshold,
    #                                                                             fixed_prob_threshold=args.val_fixed_prob_threshold)
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, sens_points = froc_out
    
    logger.info('==> Epoch: {}'.format(epoch))
    for i in range(len(sens_points)):
        logger.info('==> fps:{:.3f} iou 0.1 frocs:{:.4f}'.format(FP_ratios[i], sens_points[i]))
    logger.info('==> mean frocs:{:.4f}'.format(np.mean(np.array(sens_points))))
    
    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fixed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score,
                'best_f1_score': best_f1_score,
                'best_f1_threshold': best_f1_threshold}
    mean_recall = np.mean(np.array(sens_points))
    metrics['froc_mean_recall'] = float(mean_recall)
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
    
    return metrics