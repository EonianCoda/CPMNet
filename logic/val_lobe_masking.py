import os
import math
import logging
import numpy as np
from typing import List, Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from evaluationScript.eval import Evaluation

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
        series_list_path: str,
        exp_folder: str,
        epoch: str = 0,
        batch_size: int = 8,
        nms_keep_top_k: int = 40,
        nodule_type_diameters : Dict[str, Tuple[float, float]] = None,
        min_d: int = 0,
        min_size: int = 0,
        nodule_size_mode: str = 'seg_size') -> Dict[str, float]:
    if str(epoch).isdigit():
        save_dir = os.path.join(exp_folder, 'annotation', f'epoch_{epoch}')
    else:
        save_dir = os.path.join(exp_folder, 'annotation', epoch)
    os.makedirs(save_dir, exist_ok=True)
    if min_d != 0:
        logger.info('When validating, ignore nodules with depth less than {}'.format(min_d))
    if min_size != 0:
        logger.info('When validating, ignore nodules with size less than {}'.format(min_size))
    
    evaluator = Evaluation(series_list_path=series_list_path, 
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
            lobes = sample['split_lobes'].to(device, non_blocking=True, memory_format=memory_format)
            nzhws = sample['nzhws']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            image_shapes = sample['image_shapes']
            outputlist = []
            
            for i in range(int(math.ceil(data.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > data.size(0):
                    end = data.size(0)
                input = data[i * batch_size:end]
                lobe = lobes[i * batch_size:end]
                if args.val_mixed_precision:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output = model(input)
                            output = detection_postprocess(output, lobe, device=device)
                else:
                    with torch.no_grad():
                        output = model(input)
                        output = detection_postprocess(output, lobe, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                outputlist.append(output.data.cpu().numpy())
            
            outputs = np.concatenate(outputlist, 0)
            
            start_idx = 0
            del data
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                nzhw = nzhws[i]
                image_shape = image_shapes[i]
                output = split_comber.combine(outputs[start_idx:start_idx + n_split], nzhw, image_shape)
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
            torch.cuda.empty_cache()
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    froc_det_thresholds = args.froc_det_thresholds
    froc_info_list, fixed_out = evaluator.evaluation(preds=all_preds,
                                                                save_dir=save_dir,
                                                                froc_det_thresholds = froc_det_thresholds)
    sens_points, prec_points, f1_points, thresholds_points = froc_info_list[0]
    
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
                'f1_score': fixed_f1_score}
    
    mean_recall = np.mean(np.array(sens_points))
    metrics['froc_mean_recall'] = float(mean_recall)
    for fp_ratio, sens_p in zip(FP_ratios, sens_points):
        metrics['froc_{}_recall'.format(str(fp_ratio))] = sens_p
    
    return metrics