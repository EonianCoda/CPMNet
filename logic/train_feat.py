import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def train(args,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader: DataLoader,
          device: torch.device,
          detection_loss,
          ema = None,) -> Dict[str, float]:
    model.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_feat_loss = AverageMeter()
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
    feat_loss_fn = nn.CosineEmbeddingLoss(reduction='none')
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        all_ctr_transforms = sample['ctr_transforms'] # (N, num_aug)
        all_feat_transforms = sample['feat_transforms'] # (N, num_aug)
            
        # Split half
        num_data = len(all_ctr_transforms)
        ctr_transforms1 = all_ctr_transforms[: num_data // 2]
        ctr_transforms2 = all_ctr_transforms[num_data // 2: ]
        
        feat_transforms1 = all_feat_transforms[: num_data // 2]
        feat_transforms2 = all_feat_transforms[num_data // 2: ]
            
        if mixed_precision:
            with torch.cuda.amp.autocast():
                image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
                labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]

                output, feats = model(image)
                
                cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = detection_loss(output, labels, device = device)
                cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
                loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
                
                feats1 = feats[: num_data // 2].clone()
                feats2 = feats[num_data // 2: ].clone()
                
                for b_i in range(len(feat_transforms1)):
                    feat_transforms = feat_transforms1[b_i]
                    if len(feat_transforms) > 0:
                        for trans in reversed(feat_transforms):
                            feats1[b_i, ...] = trans.backward(feats1[b_i, ...])
                
                for b_i in range(len(feat_transforms2)):
                    feat_transforms = feat_transforms2[b_i]
                    if len(feat_transforms) > 0:
                        for trans in reversed(feat_transforms):
                            feats2[b_i, ...] = trans.backward(feats2[b_i, ...])
                
                bs = len(feats1)
                
                f1 = feats1.contiguous().view(bs, -1)
                f2 = feats2.contiguous().view(bs, -1)
                
                # Get crop has nodule
                has_nodule = torch.any(labels[: num_data // 2, :, -1] == 1, dim = 1)
                feat_loss = feat_loss_fn(f1, f2, torch.ones(bs, device=device))
                feat_loss[has_nodule] *= 2
                feat_loss = feat_loss.mean()
                loss = loss + feat_loss * args.lambda_feat
                
                del image, labels, output, feats, feats1, feats2, f1, f2
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        
        # Update history
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
        avg_feat_loss.update(feat_loss.item())
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
                                    feat_loss = avg_feat_loss.avg,
                                    iou_loss = avg_iou_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'feat_loss': avg_feat_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics