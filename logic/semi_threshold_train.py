import os
import json
import math
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format

UNLABELED_LOSS_WEIGHT = 1.0
LABELED_LOSS_WEIGHT = 1.0


logger = logging.getLogger(__name__)

def train_one_step_wrapper(memory_format):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        cls_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        return loss, cls_loss, shape_loss, offset_loss, iou_loss
    return train_one_step

def model_predict_wrapper(memory_format):
    def model_predict(model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        outputs = model(image)
        return outputs
    return model_predict

def train(args,
          model_t: nn.modules,
          model_s: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader_u: DataLoader,
          dataloader_l: DataLoader,
          num_iters: int,
          device: torch.device) -> Dict[str, float]:
    model_t.eval()
    model_s.train()
    
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
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format)
    model_predict = model_predict_wrapper(memory_format)
    
    optimizer.zero_grad()
    iter_l = iter(dataloader_l)
    with get_progress_bar('Train', num_iters) as progress_bar:
        for sample_u in dataloader_u:         
            try:
                labeled_sample = next(iter_l)
            except StopIteration:
                iter_l = iter(dataloader_l)
                labeled_sample = next(iter_l)

            optimizer.zero_grad(set_to_none=True)
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs_t = model_predict(model_t, sample_u['strong'], device)
            else:
                outputs_t = model_predict(model_t, sample_u['strong'], device)
            
            
            
            
            # # Update teacher model by exponential moving average
            # for param, teacher_param in zip(model_s.parameters(), model_t.parameters()):
            #     teacher_param.data.mul_(args.semi_ema_alpha).add_((1 - args.semi_ema_alpha) * param.data)
                
            # cls_loss = cls_loss_l * LABELED_LOSS_WEIGHT + cls_loss_u * UNLABELED_LOSS_WEIGHT
            # shape_loss = shape_loss_l * LABELED_LOSS_WEIGHT + shape_loss_u * UNLABELED_LOSS_WEIGHT
            # offset_loss = offset_loss_l * LABELED_LOSS_WEIGHT + offset_loss_u * UNLABELED_LOSS_WEIGHT
            # iou_loss = iou_loss_l * LABELED_LOSS_WEIGHT + iou_loss_u * UNLABELED_LOSS_WEIGHT
            
            # # Update history
            # avg_cls_loss.update(cls_loss.item() * args.lambda_cls)
            # avg_shape_loss.update(shape_loss.item() * args.lambda_shape)
            # avg_offset_loss.update(offset_loss.item() * args.lambda_offset)
            # avg_iou_loss.update(iou_loss.item() * args.lambda_iou)
            # avg_loss.update(loss.item())
            
            # progress_bar.set_postfix(loss = avg_loss.avg,
            #                         cls_Loss = avg_cls_loss.avg,
            #                         shape_loss = avg_shape_loss.avg,
            #                         offset_loss = avg_offset_loss.avg,
            #                         giou_loss = avg_iou_loss.avg)
            # progress_bar.update()
        

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics