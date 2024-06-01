from typing import Tuple, List, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

from utils.box_utils import bbox_decode, make_anchors, zyxdhw2zyxzyx, nms_3D

class DetectionLoss(nn.Module):
    def __init__(self, 
                 crop_size=[96, 96, 96],
                 pos_target_topk = 7, 
                 pos_ignore_ratio = 3,
                 cls_num_neg = 10000,
                 cls_num_hard = 100,
                 cls_fn_weight = 4.0,
                 cls_fn_threshold = 0.8,
                 cls_neg_pos_ratio = 100,
                 cls_hard_fp_thrs1 = 0.5,
                 cls_hard_fp_thrs2 = 0.7,
                 cls_hard_fp_w1 = 1.5,
                 cls_hard_fp_w2 = 2.0,
                 cls_focal_alpha = 0.75,
                 cls_focal_gamma = 2.0):
        super(DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        
        self.cls_num_neg = cls_num_neg
        self.cls_num_hard = cls_num_hard
        self.cls_fn_weight = cls_fn_weight
        self.cls_fn_threshold = cls_fn_threshold
        
        self.cls_neg_pos_ratio = cls_neg_pos_ratio
        
        self.cls_hard_fp_thrs1 = cls_hard_fp_thrs1
        self.cls_hard_fp_thrs2 = cls_hard_fp_thrs2
        self.cls_hard_fp_w1 = cls_hard_fp_w1
        self.cls_hard_fp_w2 = cls_hard_fp_w2
        
        self.cls_focal_alpha = cls_focal_alpha
        self.cls_focal_gamma = cls_focal_gamma
        
    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, neg_pos_ratio = 100, fn_weight = 4.0, fn_threshold = 0.8, 
                 hard_fp_thrs1 = 0.5, hard_fp_thrs2 = 0.7, hard_fp_w1 = 1.5, hard_fp_w2 = 2.0):
        """
        Calculates the classification loss using focal loss and binary cross entropy.

        Args:
            pred (torch.Tensor): The predicted logits of shape (b, num_points, 1)
            target: The target labels of shape (b, num_points, 1)
            mask_ignore: The mask indicating which pixels to ignore of shape (b, num_points, 1)
            alpha (float): The alpha factor for focal loss (default: 0.75)
            gamma (float): The gamma factor for focal loss (default: 2.0)
            num_neg (int): The maximum number of negative pixels to consider (default: 10000, if -1, use all negative pixels)
            num_hard (int): The number of hard negative pixels to keep (default: 100)
            ratio (int): The ratio of negative to positive pixels to consider (default: 100)
            fn_weight (float): The weight for false negative pixels (default: 4.0)
            fn_threshold (float): The threshold for considering a pixel as a false negative (default: 0.8)
            hard_fp_weight (float): The weight for hard false positive pixels (default: 2.0)
            hard_fp_threshold (float): The threshold for considering a pixel as a hard false positive (default: 0.7)
        Returns:
            torch.Tensor: The calculated classification loss
        """
        cls_pos_losses = []
        cls_neg_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            
            # Calculate the focal weight
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # Calculate the binary cross entropy loss
            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            if num_positive_pixels > 0:
                # Weight the hard false negatives(FN)
                FN_index = torch.lt(cls_prob, fn_threshold) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] *= fn_weight
                
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                    
                Positive_loss = cls_loss[record_targets == 1]
                Negative_loss = cls_loss[record_targets == 0]
                # Randomly sample negative pixels
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                    Negative_loss = Negative_loss[neg_idcs]
                    
                # Get the top k negative pixels
                _, keep_idx = torch.topk(Negative_loss, min(neg_pos_ratio * num_positive_pixels, len(Negative_loss))) 
                Negative_loss = Negative_loss[keep_idx] 
                
                # Calculate the loss
                num_positive_pixels = torch.clamp(num_positive_pixels.float(), min=1.0)
                Positive_loss = Positive_loss.sum() / num_positive_pixels
                Negative_loss = Negative_loss.sum() / num_positive_pixels
                cls_pos_losses.append(Positive_loss)
                cls_neg_losses.append(Negative_loss)
            else: # no positive pixels
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                
                # Randomly sample negative pixels
                Negative_loss = cls_loss[record_targets == 0]
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                    Negative_loss = Negative_loss[neg_idcs]
                
                # Get the top k negative pixels   
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                
                # Calculate the loss
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_neg_losses.append(Negative_loss)
                
        if len(cls_pos_losses) == 0:
            cls_pos_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_pos_loss = torch.sum(torch.stack(cls_pos_losses)) / batch_size
            
        if len(cls_neg_losses) == 0:
            cls_neg_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_neg_loss = torch.sum(torch.stack(cls_neg_losses)) / batch_size
        return cls_pos_loss, cls_neg_loss
    
    @staticmethod
    def target_proprocess(annotations: torch.Tensor, 
                          device, 
                          input_size: List[int],
                          stride: List[int],
                          mask_ignore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the annotations to generate the targets for the network.
        In this function, we remove some annotations that the area of nodule is too small in the crop box. (Probably cropped by the edge of the image)
        
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
            input_size: List[int]
                A list of length 3 containing the (z, y, x) dimensions of the input.
            mask_ignore: torch.Tensor
                A zero tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        Returns: 
            A tuple of two tensors:
                (1) annotations_new: torch.Tensor
                    A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                    (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
                (2) mask_ignore: torch.Tensor
                    A tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        """
        batch_size = annotations.shape[0]
        annotations_new = -1 * torch.ones_like(annotations, device=device)
        for sample_i in range(batch_size):
            annots = annotations[sample_i]
            gt_bboxes = annots[annots[:, -1] > -1] # -1 means ignore, it is used to make each sample has same number of bbox (pad with -1)
            bbox_annotation_target = []
            
            crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]], device=device)
            for s in range(len(gt_bboxes)):
                each_label = gt_bboxes[s] # (z_ctr, y_ctr, x_ctr, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1)
                # coordinate convert zmin, ymin, xmin, d, h, w
                z1 = (torch.max(each_label[0] - each_label[3]/2., crop_box[0]))
                y1 = (torch.max(each_label[1] - each_label[4]/2., crop_box[1]))
                x1 = (torch.max(each_label[2] - each_label[5]/2., crop_box[2]))

                z2 = (torch.min(each_label[0] + each_label[3]/2., crop_box[3]))
                y2 = (torch.min(each_label[1] + each_label[4]/2., crop_box[4]))
                x2 = (torch.min(each_label[2] + each_label[5]/2., crop_box[5]))
                
                nd = torch.clamp(z2 - z1, min=0.0)
                nh = torch.clamp(y2 - y1, min=0.0)
                nw = torch.clamp(x2 - x1, min=0.0)
                if nd * nh * nw == 0:
                    continue
                percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
                if (percent > 0.1) and (nw*nh*nd >= 15):
                    spacing_z, spacing_y, spacing_x = each_label[6:9]
                    bbox = torch.from_numpy(np.array([float(z1 + 0.5 * nd), float(y1 + 0.5 * nh), float(x1 + 0.5 * nw), float(nd), float(nh), float(nw), float(spacing_z), float(spacing_y), float(spacing_x), 0])).to(device)
                    bbox_annotation_target.append(bbox.view(1, 10))
                else:
                    z1 = int(torch.floor(z1 / stride[0]))
                    z2 = min(int(torch.ceil(z2 / stride[0])), mask_ignore.size(2))
                    y1 = int(torch.floor(y1 / stride[1]))
                    y2 = min(int(torch.ceil(y2 / stride[1])), mask_ignore.size(3))
                    x1 = int(torch.floor(x1 / stride[2]))
                    x2 = min(int(torch.ceil(x2 / stride[2])), mask_ignore.size(4))
                    mask_ignore[sample_i, 0, z1 : z2, y1 : y2, x1 : x2] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                annotations_new[sample_i, :len(bbox_annotation_target)] = bbox_annotation_target
        return annotations_new, mask_ignore
    
    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
                (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

        # Union Area
        union = w1 * h1 * d1 + w2 * h2 * d2 - inter

        # IoU
        iou = inter / union
        if DIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
            c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
            + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
            return iou - rho2 / c2  # DIoU
        return iou  # IoU
    
    @staticmethod
    def get_pos_target(annotations: torch.Tensor,
                       anchor_points: torch.Tensor,
                       stride: torch.Tensor,
                       pos_target_topk = 7, 
                       ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
        """Get the positive targets for the network.
        Steps:
            1. Calculate the distance between each annotation and each anchor point.
            2. Find the top k anchor points with the smallest distance for each annotation.
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format: (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1).
            anchor_points: torch.Tensor
                A tensor of shape (num_of_point, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride: torch.Tensor
                A tensor of shape (1,1,3) containing the strides of each dimension in format (z, y, x).
        """
        batchsize, num_of_annots, _ = annotations.size()
        # -1 means ignore, larger than 0 means positive
        # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
        # indicating whether the annotation is ignored or not.
        mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
        # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
        ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        
        sp = annotations[:, :, 6:9] # spacing, shape = (b, num_annotations, 3)
        sp = sp.unsqueeze(-2) # shape = (b, num_annotations, 1, 3)
        
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1)) # (b, num_annotation, num_of_points)
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
        mask_topk = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, :pos_target_topk], 1) # (b, num_annotation, num_of_points)
        mask_ignore = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, pos_target_topk:], -1) # (b, num_annotation, num_of_points)
        
        # the value is 1 or 0, shape= (b, num_annotations, num_of_points)
        # mask_gt is 1 mean the annotation is not ignored, 0 means the annotation is ignored
        # mask_topk is 1 means the point is assigned to positive
        # mask_topk * mask_gt.unsqueeze(-1) is 1 means the point is assigned to positive and the annotation is not ignored
        mask_pos = mask_topk * mask_gt.unsqueeze(-1) 
        
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1) # the value is -1 or 0, shape= (b, num_annotations, num_of_points)
        gt_idx = mask_pos.argmax(-2) # shape = (b, num_of_points), it indicates each point matches which annotation
        
        # Flatten the batch dimension
        batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None] # (b, 1)
        gt_idx = gt_idx + batch_ind * num_of_annots
        
        # Generate the targets of each points
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    def forward(self, 
                output: Dict[str, torch.Tensor], 
                annotations: torch.Tensor,
                device):
        """
        Args:
            output: Dict[str, torch.Tensor], the output of the model.
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
        """
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1) # (b, 1, num_points)
        pred_shapes = Shape.view(batch_size, 3, -1) # (b, 3, num_points)
        pred_offsets = Offset.view(batch_size, 3, -1)
        # (b, num_points, 1 or 3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
        # process annotations
        stride_list = [self.crop_size[0] / Cls.size()[2], self.crop_size[1] / Cls.size()[3], self.crop_size[2] / Cls.size()[4]]
        process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, stride_list, target_mask_ignore)
        target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # shape = (num_anchors, 3)
        # predict bboxes (zyxdhw)
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor) # shape = (b, num_anchors, 6)
        # assigned points and targets (target bboxes zyxdhw)
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
                                                                                                     anchor_points = anchor_points,
                                                                                                     stride = stride_tensor[0].view(1, 1, 3), 
                                                                                                     pos_target_topk = self.pos_target_topk,
                                                                                                     ignore_ratio = self.pos_ignore_ratio)
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        mask_ignore = mask_ignore.int()
        cls_pos_loss, cls_neg_loss = self.cls_loss(pred = pred_scores, 
                                                   target = target_scores, 
                                                   mask_ignore = mask_ignore, 
                                                   neg_pos_ratio = self.cls_neg_pos_ratio,
                                                   num_hard = self.cls_num_hard, 
                                                   num_neg = self.cls_num_neg,
                                                   fn_weight = self.cls_fn_weight, 
                                                   fn_threshold = self.cls_fn_threshold,
                                                   hard_fp_thrs1=self.cls_hard_fp_thrs1,
                                                   hard_fp_thrs2=self.cls_hard_fp_thrs2,
                                                    hard_fp_w1=self.cls_hard_fp_w1,
                                                    hard_fp_w2=self.cls_hard_fp_w2,
                                                    alpha=self.cls_focal_alpha,
                                                    gamma=self.cls_focal_gamma)
        
        # Only calculate the loss of positive samples                                 
        fg_mask = target_scores.squeeze(-1).bool()
        if fg_mask.sum() == 0:
            reg_loss = torch.tensor(0.0, device=device)
            offset_loss = torch.tensor(0.0, device=device)
            iou_loss = torch.tensor(0.0, device=device)
        else:
            reg_loss = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_loss = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_loss = 1 - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss
    
class Unsupervised_DetectionLoss(nn.Module):
    def __init__(self, 
                 crop_size=[96, 96, 96],
                 pos_target_topk = 7, 
                 pos_ignore_ratio = 3,
                 soft_pos_target_topk = 7,
                 soft_pos_ignore_ratio = 3,
                 cls_num_neg = 10000,
                 cls_num_hard = 100,
                 cls_fn_weight = 4.0,
                 cls_fn_threshold = 0.8,
                 cls_neg_pos_ratio = 100,
                 cls_focal_alpha = 0.75,
                 cls_focal_gamma = 2.0,
                 cls_soft_focal_alpha = 0.9,
                 cls_soft_focal_gamma = 1.5):
        super(Unsupervised_DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        self.soft_pos_target_topk = soft_pos_target_topk
        self.soft_pos_ignore_ratio = soft_pos_ignore_ratio
        
        self.cls_num_neg = cls_num_neg
        self.cls_num_hard = cls_num_hard
        self.cls_fn_weight = cls_fn_weight
        self.cls_fn_threshold = cls_fn_threshold
        
        self.cls_neg_pos_ratio = cls_neg_pos_ratio
        
        self.cls_focal_alpha = cls_focal_alpha
        self.cls_focal_gamma = cls_focal_gamma
        
        self.cls_soft_focal_alpha = cls_soft_focal_alpha
        self.cls_soft_focal_gamma = cls_soft_focal_gamma
        
    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, soft_target, soft_mask_ignore, background_mask, 
                 alpha = 0.75 , gamma = 2.0, soft_alpha = 0.9, soft_gamma = 1.5, num_neg = 10000, 
                 num_hard = 100, neg_pos_ratio = 100, fn_weight = 4.0, fn_threshold = 0.8):
        """
        Calculates the classification loss using focal loss and binary cross entropy.

        Args:
            pred (torch.Tensor): The predicted logits of shape (b, num_points, 1)
            target: The target labels of shape (b, num_points, 1)
            mask_ignore: The mask indicating which pixels to ignore of shape (b, num_points, 1)
            alpha (float): The alpha factor for focal loss (default: 0.75)
            gamma (float): The gamma factor for focal loss (default: 2.0)
            num_neg (int): The maximum number of negative pixels to consider (default: 10000, if -1, use all negative pixels)
            num_hard (int): The number of hard negative pixels to keep (default: 100)
            ratio (int): The ratio of negative to positive pixels to consider (default: 100)
            fn_weight (float): The weight for false negative pixels (default: 4.0)
            fn_threshold (float): The threshold for considering a pixel as a false negative (default: 0.8)

        Returns:
            torch.Tensor: The calculated classification loss
        """
        def ce_loss(target, pred, eps=1e-6):
            pred = pred.sigmoid()
            ce = - target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps) # shape = (b, num_points)
            return ce
        
        cls_pos_losses = []
        cls_neg_losses = []
        cls_soft_losses = []
        batch_size = pred.shape[0]
        background_mask = background_mask.unsqueeze(-1) # shape = (b, num_points, 1)
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            
            soft_target_b = soft_target[j]
            soft_mask_ignore_b = soft_mask_ignore[j]
            
            # Calculate the focal weight
            pred_prob = torch.sigmoid(pred_b.detach())
            pred_prob = torch.clamp(pred_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - pred_prob, pred_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # Calculate the focal loss
            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            
            # Get mask for pos, neg and soft mask
            pos_mask = (record_targets == 1)
            neg_mask = torch.logical_and(torch.logical_and(record_targets == 0, background_mask[j]), soft_mask_ignore_b == 0)
            # Weight the hard false negatives(FN) and false positives(FP)
            if num_positive_pixels > 0:
                # FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(pred_prob, fn_threshold) & pos_mask  # 0.9
                cls_loss[FN_index == 1] = fn_weight * cls_loss[FN_index == 1]
                
                Positive_loss = cls_loss[pos_mask]
                Negative_loss = cls_loss[neg_mask]
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                    Negative_loss = Negative_loss[neg_idcs]
                _, keep_idx = torch.topk(Negative_loss, min(neg_pos_ratio * num_positive_pixels, len(Negative_loss))) 
                Negative_loss = Negative_loss[keep_idx]
                
                # Calculate the loss
                num_positive_pixels = torch.clamp(num_positive_pixels.float(), min=1.0)
                Positive_loss = Positive_loss.sum() / num_positive_pixels
                Negative_loss = Negative_loss.sum() / num_positive_pixels
                cls_pos_losses.append(Positive_loss)
                cls_neg_losses.append(Negative_loss)
            else:
                Negative_loss = cls_loss[neg_mask]
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                    Negative_loss = Negative_loss[neg_idcs]
                assert len(Negative_loss) > num_hard
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_neg_losses.append(Negative_loss)
            
            # Calculate the soft loss
            # Compute soft focal weight
            soft_mask = torch.logical_and(soft_target_b > 0, record_targets == 0)
            if torch.sum(soft_mask) > 0:
                a1 = 1 - soft_alpha
                a2 = soft_alpha
                beta = ((soft_target_b - pred_prob).abs() + 1e-6).pow(soft_gamma)
                soft_weight = (a1 + soft_target_b * (a2 - a1)) * beta + 1e-6
                soft_loss = ce_loss(soft_target_b, pred_prob)
                soft_loss = (soft_loss * soft_weight)
                soft_loss = soft_loss[soft_mask]
                cls_soft_losses.append(soft_loss.sum())
            
        if len(cls_pos_losses) == 0:
            cls_pos_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_pos_loss = torch.sum(torch.stack(cls_pos_losses)) / batch_size
            
        if len(cls_neg_losses) == 0:
            cls_neg_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_neg_loss = torch.sum(torch.stack(cls_neg_losses)) / batch_size
        
        if len(cls_soft_losses) == 0:
            cls_soft_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_soft_loss = torch.sum(torch.stack(cls_soft_losses)) / batch_size    
        
        return cls_pos_loss, cls_neg_loss, cls_soft_loss
    
    @staticmethod
    def target_proprocess(annotations: torch.Tensor, 
                          device, 
                          input_size: List[int],
                          stride: List[int],
                          mask_ignore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the annotations to generate the targets for the network.
        In this function, we remove some annotations that the area of nodule is too small in the crop box. (Probably cropped by the edge of the image)
        
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
            input_size: List[int]
                A list of length 3 containing the (z, y, x) dimensions of the input.
            mask_ignore: torch.Tensor
                A zero tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        Returns: 
            A tuple of two tensors:
                (1) annotations_new: torch.Tensor
                    A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                    (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
                (2) mask_ignore: torch.Tensor
                    A tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        """
        batch_size = annotations.shape[0]
        annotations_new = -1 * torch.ones_like(annotations, device=device)
        for sample_i in range(batch_size):
            annots = annotations[sample_i]
            gt_bboxes = annots[annots[:, -1] > -1] # -1 means ignore, it is used to make each sample has same number of bbox (pad with -1)
            bbox_annotation_target = []
            
            crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]], device=device)
            for s in range(len(gt_bboxes)):
                each_label = gt_bboxes[s] # (z_ctr, y_ctr, x_ctr, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1)
                # coordinate convert zmin, ymin, xmin, d, h, w
                z1 = (torch.max(each_label[0] - each_label[3]/2., crop_box[0]))
                y1 = (torch.max(each_label[1] - each_label[4]/2., crop_box[1]))
                x1 = (torch.max(each_label[2] - each_label[5]/2., crop_box[2]))

                z2 = (torch.min(each_label[0] + each_label[3]/2., crop_box[3]))
                y2 = (torch.min(each_label[1] + each_label[4]/2., crop_box[4]))
                x2 = (torch.min(each_label[2] + each_label[5]/2., crop_box[5]))
                
                nd = torch.clamp(z2 - z1, min=0.0)
                nh = torch.clamp(y2 - y1, min=0.0)
                nw = torch.clamp(x2 - x1, min=0.0)
                if nd * nh * nw == 0:
                    continue
                percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
                if (percent > 0.1) and (nw*nh*nd >= 15):
                    spacing_z, spacing_y, spacing_x = each_label[6:9]
                    bbox = torch.from_numpy(np.array([float(z1 + 0.5 * nd), float(y1 + 0.5 * nh), float(x1 + 0.5 * nw), float(nd), float(nh), float(nw), float(spacing_z), float(spacing_y), float(spacing_x), float(each_label[9])])).to(device)
                    bbox_annotation_target.append(bbox.view(1, 10))
                else:
                    z1 = int(torch.floor(z1 / stride[0]))
                    z2 = min(int(torch.ceil(z2 / stride[0])), mask_ignore.size(2))
                    y1 = int(torch.floor(y1 / stride[1]))
                    y2 = min(int(torch.ceil(y2 / stride[1])), mask_ignore.size(3))
                    x1 = int(torch.floor(x1 / stride[2]))
                    x2 = min(int(torch.ceil(x2 / stride[2])), mask_ignore.size(4))
                    mask_ignore[sample_i, 0, z1 : z2, y1 : y2, x1 : x2] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                annotations_new[sample_i, :len(bbox_annotation_target)] = bbox_annotation_target
        return annotations_new, mask_ignore
    
    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
                (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

        # Union Area
        union = w1 * h1 * d1 + w2 * h2 * d2 - inter

        # IoU
        iou = inter / union
        if DIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
            c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
            + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
            return iou - rho2 / c2  # DIoU
        return iou  # IoU
    
    @staticmethod
    def get_pos_target(annotations: torch.Tensor,
                       anchor_points: torch.Tensor,
                       stride: torch.Tensor,
                       pos_target_topk = 7, 
                       ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
        """Get the positive targets for the network.
        Steps:
            1. Calculate the distance between each annotation and each anchor point.
            2. Find the top k anchor points with the smallest distance for each annotation.
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format: (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1).
            anchor_points: torch.Tensor
                A tensor of shape (num_of_point, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride: torch.Tensor
                A tensor of shape (1,1,3) containing the strides of each dimension in format (z, y, x).
        """
        batchsize, num_of_annots, _ = annotations.size()
        # -1 means ignore, larger than 0 means positive
        # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
        # indicating whether the annotation is ignored or not.
        mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
        # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
        ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        
        sp = annotations[:, :, 6:9] # spacing, shape = (b, num_annotations, 3)
        sp = sp.unsqueeze(-2) # shape = (b, num_annotations, 1, 3)
        
        # distance (b, n_max_object, anchors)
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1))
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
        mask_topk = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, :pos_target_topk], 1) # (b, num_annotation, num_of_points)
        mask_ignore = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, pos_target_topk:], -1) # (b, num_annotation, num_of_points)
        
        # the value is 1 or 0, shape= (b, num_annotations, num_of_points)
        # mask_gt is 1 mean the annotation is not ignored, 0 means the annotation is ignored
        # mask_topk is 1 means the point is assigned to positive
        # mask_topk * mask_gt.unsqueeze(-1) is 1 means the point is assigned to positive and the annotation is not ignored
        mask_pos = mask_topk * mask_gt.unsqueeze(-1) 
        
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1) # the value is -1 or 0, shape= (b, num_annotations, num_of_points)
        gt_idx = mask_pos.argmax(-2) # shape = (b, num_of_points), it indicates each point matches which annotation
        
        # Flatten the batch dimension
        batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None] # (b, 1)
        gt_idx = gt_idx + batch_ind * num_of_annots
        
        # Generate the targets of each points
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    @staticmethod
    def get_pos_soft_target(annotations: torch.Tensor,
                            anchor_points: torch.Tensor,
                            stride: torch.Tensor,
                            pos_target_topk = 7, 
                            ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
        batchsize, num_of_annots, _ = annotations.size()
        # -1 means ignore, larger than 0 means positive
        # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
        # indicating whether the annotation is ignored or not.
        mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
        # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
        ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        
        sp = annotations[:, :, 6:9] # spacing, shape = (b, num_annotations, 3)
        sp = sp.unsqueeze(-2) # shape = (b, num_annotations, 1, 3)
        
        # distance (b, n_max_object, anchors)
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1))
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
        soft_prob = annotations[:, :, -1].clone()
        soft_prob = soft_prob.unsqueeze(-1)
        soft_prob = torch.repeat_interleave(soft_prob, repeats=pos_target_topk, dim=-1)
        mask_topk = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, :pos_target_topk], soft_prob) # (b, num_annotation, num_of_points)
        mask_ignore = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, pos_target_topk:], -1) # (b, num_annotation, num_of_points)
        
        # the value is 1 or 0, shape= (b, num_annotations, num_of_points)
        # mask_gt is 1 mean the annotation is not ignored, 0 means the annotation is ignored
        # mask_topk is 1 means the point is assigned to positive
        # mask_topk * mask_gt.unsqueeze(-1) is 1 means the point is assigned to positive and the annotation is not ignored
        mask_pos = mask_topk * mask_gt.unsqueeze(-1) 
        # print(torch.max(mask_pos))
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1) # the value is -1 or 0, shape= (b, num_annotations, num_of_points)
        gt_idx = mask_pos.argmax(-2) # shape = (b, num_of_points), it indicates each point matches which annotation
        
        # Flatten the batch dimension
        batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None] # (b, 1)
        gt_idx = gt_idx + batch_ind * num_of_annots
        
        # Generate the targets of each points
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    def forward(self, 
                output: Dict[str, torch.Tensor], 
                annotations: torch.Tensor,
                soft_annotation: torch.Tensor,
                background_mask: torch.Tensor, # shape = (b, num_points)
                soft_prob: torch.Tensor,
                device):
        """
        Args:
            output: Dict[str, torch.Tensor], the output of the model.
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
        """
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        soft_target_mask_ignore = torch.zeros(Cls.size()).to(device)
        
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1) # (b, 1, num_points)
        pred_shapes = Shape.view(batch_size, 3, -1) # (b, 3, num_points)
        pred_offsets = Offset.view(batch_size, 3, -1)
        soft_prob = soft_prob.view(batch_size, 1, -1)
        # (b, num_points, 1 or 3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        soft_prob = soft_prob.permute(0, 2, 1).contiguous()
        
        # process annotations
        stride_list = [self.crop_size[0] / Cls.size()[2], self.crop_size[1] / Cls.size()[3], self.crop_size[2] / Cls.size()[4]]
        process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, stride_list, target_mask_ignore)
        target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
        
        # process soft annotations
        process_soft_annotations, soft_target_mask_ignore = self.target_proprocess(soft_annotation, device, self.crop_size, stride_list, soft_target_mask_ignore)
        soft_target_mask_ignore = soft_target_mask_ignore.view(batch_size, 1,  -1)
        soft_target_mask_ignore = soft_target_mask_ignore.permute(0, 2, 1).contiguous()
        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # shape = (num_anchors, 3)
        # predict bboxes (zyxdhw)
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor) # shape = (b, num_anchors, 6)
        # assigned points and targets (target bboxes zyxdhw)
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
                                                                                                     anchor_points = anchor_points,
                                                                                                     stride = stride_tensor[0].view(1, 1, 3), 
                                                                                                     pos_target_topk = self.pos_target_topk,
                                                                                                     ignore_ratio = self.pos_ignore_ratio)
        soft_target_offset, soft_target_shape, soft_target_bboxes, soft_target_scores, soft_mask_ignore = self.get_pos_soft_target(annotations = process_soft_annotations,
                                                                                                                                    anchor_points = anchor_points,
                                                                                                                                    stride = stride_tensor[0].view(1, 1, 3),
                                                                                                                                    pos_target_topk = self.soft_pos_target_topk,
                                                                                                                                    ignore_ratio = self.soft_pos_ignore_ratio)
        
        
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        fg_mask = target_scores.squeeze(-1).bool()
        
        soft_mask_ignore = soft_mask_ignore.bool() | soft_target_mask_ignore.bool()
        
        cls_pos_loss, cls_neg_loss, cls_soft_loss = self.cls_loss(pred = pred_scores, 
                                                                target = target_scores, 
                                                                mask_ignore = mask_ignore,
                                                                soft_target= soft_target_scores,
                                                                soft_mask_ignore = soft_mask_ignore,
                                                                background_mask = background_mask, 
                                                                neg_pos_ratio = self.cls_neg_pos_ratio,
                                                                num_hard = self.cls_num_hard, 
                                                                num_neg = self.cls_num_neg,
                                                                fn_weight = self.cls_fn_weight, 
                                                                fn_threshold = self.cls_fn_threshold,
                                                                alpha=self.cls_focal_alpha,
                                                                gamma=self.cls_focal_gamma,
                                                                soft_alpha=self.cls_soft_focal_alpha,
                                                                soft_gamma=self.cls_soft_focal_gamma)
                                              
        if fg_mask.sum() == 0:
            reg_loss = torch.tensor(0).float().to(device)
            offset_loss = torch.tensor(0).float().to(device)
            iou_loss = torch.tensor(0).float().to(device)
        else:
            reg_loss = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_loss = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_loss = 1 - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return cls_pos_loss, cls_neg_loss, cls_soft_loss, reg_loss, offset_loss, iou_loss