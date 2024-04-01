import math
import logging
import numpy as np
from typing import List, Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from dataload.utils import ALL_LOC, ALL_RAD, ALL_CLS, ALL_PROB, NODULE_SIZE
from utils.utils import get_progress_bar
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def pred2label(pred: np.ndarray, prob_threshold: float) -> Dict[str, np.ndarray]:
    pred = pred[..., 1:] # List of [prob, ctr_z, ctr_y, ctr_x, d, h, w]
        
    all_prob = []
    all_loc = []
    all_rad = []
    all_cls = []
    for i in range(len(pred)):
        prob, ctr_z, ctr_y, ctr_x, d, h, w = pred[i]
        if prob < prob_threshold:
            continue
            
        all_loc.append([ctr_z, ctr_y, ctr_x])
        all_rad.append([d, h, w])
        all_cls.append(0)
        all_prob.append(prob)
        
    if len(all_loc) == 0:
        label = {ALL_LOC: np.zeros((0, 3)),
                ALL_RAD: np.zeros((0,)),
                ALL_CLS: np.zeros((0, 3), dtype=np.int32),
                ALL_PROB: np.zeros((0,))}
    else:
        label = {ALL_LOC: np.array(all_loc),
                ALL_RAD: np.array(all_rad),
                ALL_CLS: np.array(all_cls, dtype=np.int32),
                ALL_PROB: np.array(all_prob)}
    return label
    
def gen_pseu_labels(model: nn.Module,
                    dataloader: DataLoader,
                    device: torch.device,
                    detection_postprocess,
                    prob_threshold: float = 0.8,
                    batch_size: int = 16,
                    nms_keep_top_k: int = 40,
                    mixed_precision: bool = False,
                    memory_format: str = None) -> Dict[str, np.ndarray]:
    """
    Return:
        A dictionary with series name as key and pseudo labels as value. The pseudo label is a dictionary with keys 'all_loc', 'all_rad', 'all_cls'.
    """
    logger.info("Generating pseudo labels")
    
    model.eval()
    split_comber = dataloader.dataset.splitcomb
    memory_format = get_memory_format(memory_format)
    
    pseudo_labels = dict()
    
    with get_progress_bar('Pseu-Label Generation', len(dataloader)) as pbar:
        for sample in dataloader:
            data = sample['split_images'].to(device, non_blocking=True, memory_format=memory_format)
            nzhws = sample['nzhws']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            image_shapes = sample['image_shapes']
            
            preds = []
            for i in range(int(math.ceil(data.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > data.size(0):
                    end = data.size(0)
                input = data[i * batch_size:end]
                
                with torch.no_grad():
                    if mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred = model(input)
                            pred = detection_postprocess(pred, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                    else:
                        pred = model(input)
                        pred = detection_postprocess(pred, device=device)
                        
                preds.append(pred.data.cpu().numpy())
            
            preds = np.concatenate(preds, 0) # [n, 8]
            
            start_index = 0
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                nzhw = nzhws[i]
                image_shape = image_shapes[i]
                series_name = series_names[i]
                pred = split_comber.combine(preds[start_index:start_index + n_split], nzhw, image_shape)
                
                pred = torch.from_numpy(pred).view(-1, 8)
                # Remove the padding
                valid_mask = (pred[:, 0] != -1.0)
                pred = pred[valid_mask]
                # NMS
                if len(pred) > 0:
                    keep = nms_3D(pred[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
                    pred = pred[keep]
                pred = pred.numpy()
                pseudo_labels[series_name] = pred2label(pred, prob_threshold)
                start_index += n_split
            pbar.update(1)
    return pseudo_labels