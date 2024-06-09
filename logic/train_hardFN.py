import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format
from dataload.utils import compute_bbox3d_iou

logger = logging.getLogger(__name__)

# def train_one_step_wrapper(memory_format):
#     def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
#         labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
#         # Compute loss
#         cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
#         cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
#         loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
#         del image, labels
#         return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss
#     return train_one_step

def train(args,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader: DataLoader,
          device: torch.device,
          detection_postprocess,
          detection_loss,
          ema = None) -> Dict[str, float]:
    model.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    iters_to_accumulate = args.iters_to_accumulate
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    total_num_steps = len(dataloader)
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    # train_one_step = train_one_step_wrapper(memory_format)
        
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        gt_annot_np = sample['annot'].numpy()
        if mixed_precision:
            with torch.cuda.amp.autocast():
                image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
                labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
                
                output = model(image)
                # Compute loss
                cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = detection_loss(output, labels, device = device)
                cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
                loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
                
                # cls_output = output['Cls'].sigmoid().detach().cpu().numpy()   
                # del output
                processed_output = detection_postprocess(output, device=device, is_logits=True) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                processed_output = processed_output.data.cpu().numpy()
                del output
                del image, labels
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
            
            # Update nodule probs
            series_names = sample['series_names']
            all_nodule_indices = sample['nodule_indices']
            
            updated_series_names = []
            updated_nodule_indices = []
            updated_nodule_probs = []
            
            for batch_i in range(len(series_names)):
                series_name = series_names[batch_i]
                nodule_indices_b = all_nodule_indices[batch_i]
                
                # Get ground truth nodule
                gt_annot_b = gt_annot_np[batch_i]
                gt_annot_b = gt_annot_b[gt_annot_b[:, -1] != -1.0]
                if len(gt_annot_b) == 0:
                    continue
                gt_ctrs = gt_annot_b[:, :3]
                gt_rads = gt_annot_b[:, 3:6]
                gt_bboxes = np.stack([gt_ctrs - gt_rads / 2, gt_ctrs + gt_rads / 2], axis=1)
                
                # Get predicted nodule
                output_b = processed_output[batch_i]
                valid_mask = (output_b[:, -1] != -1.0)
                output_b = output_b[valid_mask]
                
                if len(output_b) == 0: # No valid nodule
                    for i in range(len(nodule_indices_b)):
                        updated_series_names.append(series_name)
                        updated_nodule_indices.append(nodule_indices_b[i])
                        updated_nodule_probs.append(0.0)
                    continue
                
                pred_probs = output_b[:, 1]
                pred_ctrs = output_b[:, 2:5]
                pred_rads = output_b[:, 5:8]
                
                pred_bboxes = np.stack([pred_ctrs - pred_rads / 2, pred_ctrs + pred_rads / 2], axis=1)
                
                # Compute iou
                ious = compute_bbox3d_iou(gt_bboxes, pred_bboxes)
                
                matched_pred_iou = np.max(ious, axis=1)
                matched_pred_indices = np.argmax(ious, axis=1)
                for gt_i, (iou, pred_i) in enumerate(zip(matched_pred_iou, matched_pred_indices)):
                    if iou >= 0.1:
                        updated_series_names.append(series_name)
                        updated_nodule_indices.append(nodule_indices_b[gt_i])
                        updated_nodule_probs.append(pred_probs[pred_i])
                    else:
                        updated_series_names.append(series_name)
                        updated_nodule_indices.append(nodule_indices_b[gt_i])
                        updated_nodule_probs.append(0.0)
            del processed_output
            dataloader.dataset.update_nodule_probs(updated_series_names, updated_nodule_indices, updated_nodule_probs)
            
        # Update history
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
        avg_loss.update(loss.item() * iters_to_accumulate)
        
        # Update model
        if (iter_i + 1) % iters_to_accumulate == 0 or iter_i == total_num_steps - 1:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            progress_bar.set_postfix(loss = avg_loss.avg,
                                    pos_cls = avg_cls_pos_loss.avg,
                                    neg_cls = avg_cls_neg_loss.avg,
                                    cls_loss = avg_cls_loss.avg,
                                    shape_loss = avg_shape_loss.avg,
                                    offset_loss = avg_offset_loss.avg,
                                    iou_loss = avg_iou_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics