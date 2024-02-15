import os
import math
import logging
import numpy as np
from typing import List, Any
from collections import defaultdict


import torch
import torch.nn as nn

from utils.box_utils import nms_3D
from inference.evaluation import Evaluation, DEFAULT_FP_RATIOS
from inference.nodule_finding import NoduleFinding

logger = logging.getLogger(__name__)

def convert_to_standard_output(output: np.ndarray, spacing: torch.Tensor, name: str) -> List[List[Any]]:
    '''
    convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
    ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    '''
    preds = []
    spacing = np.array([spacing[0].numpy(), spacing[1].numpy(), spacing[2].numpy()]).reshape(-1, 3)
    for j in range(output.shape[0]):
        preds.append([name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
    return preds

def val(args,
        model: nn.Module,
        val_loader,
        detection_postprocess,
        device: torch.device,
        epoch: int,
        image_spacing: List[float],
        series_list_path: str,
        output_root: str,
        nms_keep_top_k: int = 40):
    model.eval()
    split_comber = val_loader.dataset.splitcomb
    batch_size = args.batch_size * args.num_samples
    all_preds = []
    for sample in val_loader:
        data = sample['split_images'][0].to(device, non_blocking=True)
        nzhw = sample['nzhw']
        name = sample['file_name'][0]
        spacing = sample['spacing'][0]
        outputlist = []
        
        for i in range(int(math.ceil(data.size(0) / batch_size))):
            end = (i + 1) * batch_size
            if end > data.size(0):
                end = data.size(0)
            input = data[i * batch_size:end]
            with torch.no_grad():
                output = model(input)
                output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputlist.append(output.data.cpu().numpy())
            
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8) # [N, 8], 8-> id, prob, z_min, y_min, x_min, d, h, w 
        
        # Remove the padding
        object_ids = output[:, 0] != -1.0
        output = output[object_ids]
        
        # NMS
        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
            output = output[keep]
        output = output.numpy()
        
        preds = convert_to_standard_output(output, spacing, name) # convert to ['seriesuid', 'coordX', 'coordY', 'coordZ', 'radius', 'probability']
        all_preds.extend(preds)
    pred_nodules = defaultdict(list)
    # Save the results to csv
    for pred in all_preds:
        series_name = pred[0]
        x, y, z = pred[1], pred[2], pred[3]
        prob = pred[4]
        w, h, d = pred[5], pred[6], pred[7]
        nodule = NoduleFinding(coord_x=x, coord_y=y, coord_z=z, pred_prob=prob, w=w, h=h, d=d) 
        pred_nodules[series_name].append(nodule)
    
    output_dir = os.path.join(output_root, f'predictions_{epoch}')
    os.makedirs(output_dir, exist_ok=True)
    evaluation = Evaluation(series_list_path=series_list_path, 
                            image_spacing=image_spacing)
    all_metrics, inter_btp_mean, inter_points = evaluation.run(pred_nodules, output_dir)
    
    return all_metrics, inter_btp_mean, inter_points