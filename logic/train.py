import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar

logger = logging.getLogger(__name__)

def write_metrics(metrics: Dict[str, float], epoch: int, prefix: str, writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(model: nn.Module, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                save_path: str):
    
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_structure': model}
    torch.save(save_dict, save_path)

def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = sample['image'].to(device, non_blocking=True) # z, y, x
    labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
    
    # Compute loss
    cls_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
    cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
    loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
    return loss, cls_loss, shape_loss, offset_loss, iou_loss

def train(args,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader: DataLoader,
          scheduler: torch.optim.lr_scheduler,
          device: torch.device) -> Dict[str, float]:
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
        
    # get_progress_bar
    progress_bar = get_progress_bar('Train', len(dataloader))
    for batch_idx, sample in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, sample, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, sample, device)
            loss.backward()
            optimizer.step()
        
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