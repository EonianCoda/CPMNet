import os
import logging
import math
import json
import numpy as np
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.box_utils import nms_3D
from utils.utils import get_progress_bar

from utils.average_meter import AverageMeter
logger = logging.getLogger(__name__)

UNLABELED_LOSS_WEIGHT = 1.0
LABELED_LOSS_WEIGHT = 1.0

def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = sample['image'].to(device, non_blocking=True) # z, y, x
    labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
    
    # Compute loss
    cls_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
    cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
    loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
    return loss, cls_loss, shape_loss, offset_loss, iou_loss

def generate_pseudo_labels(args,
                            model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            detection_postprocess,
                            batch_size: int = 16,
                            nms_keep_top_k: int = 40) -> None:
    logger.info("Generating pseudo labels")
    model.eval()
    split_comber = dataloader.dataset.splitcomb
    
    for sample in dataloader:
        # Generate pseudo labels
        data = sample['split_images'].to(device, non_blocking=True)
        nzhws = sample['nzhws']
        series_names = sample['series_names']
        series_folders = sample['series_folders']
        num_splits = sample['num_splits']
        
        outputlist = []
        for i in range(int(math.ceil(data.size(0) / batch_size))):
            end = (i + 1) * batch_size
            if end > data.size(0):
                end = data.size(0)
            input = data[i * batch_size:end]
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = model(input)
                    output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputlist.append(output.data.cpu().numpy())
         
        outputs = np.concatenate(outputlist, 0)
        
        scan_outputs = []
        start_ind = 0
        for i in range(len(num_splits)):
            n_split = num_splits[i]
            nzhw = nzhws[i]
            output = split_comber.combine(outputs[start_ind:start_ind + n_split], nzhw)
            output = torch.from_numpy(output).view(-1, 8)
            # Remove the padding
            object_ids = output[:, 0] != -1.0
            output = output[object_ids]
            
            # NMS
            if len(output) > 0:
                keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
                output = output[keep]
            output = output.numpy()
            scan_outputs.append(output)           
            start_ind += n_split
            
        # Save pesudo labels
        for i in range(len(scan_outputs)):
            series_name = series_names[i]
            series_folder = series_folders[i]
            save_path = os.path.join(series_folder, 'pseudo_label', f"{series_name}.json")

            all_loc = scan_outputs[i][:, 2:5]
            all_rad = scan_outputs[i][:, 5:]
            
            all_prob = scan_outputs[i][:, 1]
            all_prob = all_prob.reshape(-1, 1)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({'all_loc': all_loc.tolist(), 
                           'all_prob': all_prob.tolist(), 
                           'all_rad': all_rad.tolist()}, f, indent=4)
                
def train(args,
          teacher_model: nn.modules,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          unlabeled_train_dataloader: DataLoader,
          labeled_dataloader: DataLoader,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device) -> Dict[str, float]:
    teacher_model.eval()
    model.train()
    scheduler.step()
    
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    logger.info('Number of unlabeled batch: %d', len(unlabeled_train_dataloader))
    # get_progress_bar
    progress_bar = get_progress_bar('Train', len(unlabeled_train_dataloader))
    iter_labeled_train = iter(labeled_dataloader)
    
    for unlabeled_sample in unlabeled_train_dataloader:         
        if len(unlabeled_sample['image']) == 0:
            logger.info("No unlabeled data")
            continue
        try:
            labeled_sample = next(iter_labeled_train)
        except StopIteration:
            iter_labeled_train = iter(labeled_dataloader)
            labeled_sample = next(iter_labeled_train)

        optimizer.zero_grad(set_to_none=True)
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss_u, cls_loss_u, shape_loss_u, offset_loss_u, iou_loss_u = train_one_step(args, model, unlabeled_sample, device)
                loss_l, cls_loss_l, shape_loss_l, offset_loss_l, iou_loss_l = train_one_step(args, model, labeled_sample, device)
                
            loss = loss_l * LABELED_LOSS_WEIGHT + loss_u * UNLABELED_LOSS_WEIGHT
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_u, cls_loss_u, shape_loss_u, offset_loss_u, iou_loss_u = train_one_step(args, model, unlabeled_sample, device)
            loss_l, cls_loss_l, shape_loss_l, offset_loss_l, iou_loss_l = train_one_step(args, model, labeled_sample, device)
            loss = loss_l * LABELED_LOSS_WEIGHT + loss_u * UNLABELED_LOSS_WEIGHT 
            loss.backward()
            optimizer.step()
        
        # Update teacher model by exponential moving average
        for param, teacher_param in zip(model.parameters(), teacher_model.parameters()):
            teacher_param.data.mul_(args.semi_ema_alpha).add_((1 - args.semi_ema_alpha) * param.data)
            
        cls_loss = cls_loss_l * LABELED_LOSS_WEIGHT + cls_loss_u * UNLABELED_LOSS_WEIGHT
        shape_loss = shape_loss_l * LABELED_LOSS_WEIGHT + shape_loss_u * UNLABELED_LOSS_WEIGHT
        offset_loss = offset_loss_l * LABELED_LOSS_WEIGHT + offset_loss_u * UNLABELED_LOSS_WEIGHT
        iou_loss = iou_loss_l * LABELED_LOSS_WEIGHT + iou_loss_u * UNLABELED_LOSS_WEIGHT
        
        # Update history
        avg_cls_loss.update(cls_loss.item() * args.lambda_cls)
        avg_shape_loss.update(shape_loss.item() * args.lambda_shape)
        avg_offset_loss.update(offset_loss.item() * args.lambda_offset)
        avg_iou_loss.update(iou_loss.item() * args.lambda_iou)
        avg_loss.update(loss.item())
        
        progress_bar.set_postfix(loss = avg_loss.avg,
                                cls_Loss = avg_cls_loss.avg,
                                shape_loss = avg_shape_loss.avg,
                                offset_loss = avg_offset_loss.avg,
                                giou_loss = avg_iou_loss.avg)
        progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics