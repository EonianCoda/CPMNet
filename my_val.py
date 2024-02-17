# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
from typing import List, Tuple, Any, Dict

from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess, DetectionLoss
### data ###
from dataload.my_dataset_crop import DetDatasetCSVRTest
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import torch.nn as nn
import transform as transform

from logic.val import val

from utils.logs import setup_logging
from utils.utils import init_seed

SAVE_ROOT = './save'
DEFAULT_CROP_SIZE = [64, 128, 128]
OVERLAY_RATIO = 0.25
IMAGE_SPACING = [1.0, 0.8, 0.8]
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='use pin memory')
    parser.add_argument('--num_workers', type=int, default=1, metavar='S', help='num_workers (default: 1)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=DEFAULT_CROP_SIZE, help='crop size')
    parser.add_argument('--model_path', type=str, default='', metavar='str')
    # data
    parser.add_argument('--val_set', type=str, required=True,help='val_list')
    # hyper-parameters
    parser.add_argument('--num_samples', type=int, default=5, metavar='N', help='sampling batch number in per sample')
    parser.add_argument('--pos_target_topk', type=int, default=5, metavar='N', help='topk grids assigned as positives')
    parser.add_argument('--pos_ignore_ratio', type=int, default=3)
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.7, help='fixed probability threshold for validation')
    
    # detection-hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    # network
    parser.add_argument('--norm_type', type=str, default='batchnorm', metavar='N', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', metavar='N', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', metavar='N', help='act type of network')
    parser.add_argument('--se', action='store_true', default=False, help='use se block')
    # other
    args = parser.parse_args()
    return args

def prepare_validation(args):
    # build model
    detection_loss = DetectionLoss(crop_size = args.crop_size,
                                   pos_target_topk = args.pos_target_topk, 
                                   spacing = IMAGE_SPACING, 
                                   pos_ignore_ratio = args.pos_ignore_ratio)
    model = Resnet18(n_channels = 1, 
                     n_blocks = [2, 3, 3, 3], 
                     n_filters = [64, 96, 128, 160], 
                     stem_filters = 32,
                     norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type, 
                     se = args.se, 
                     first_stride = (1, 2, 2), 
                     detection_loss = detection_loss,
                     device = device)
    detection_postprocess = DetectionPostprocess(topk=args.det_topk, 
                                                 threshold=args.det_threshold, 
                                                 nms_threshold=args.det_nms_threshold,
                                                 nms_topk=args.det_nms_topk,
                                                 crop_size=args.crop_size)
    
    logger.info('Load model from "{}"'.format(args.model_path))
    state_dict = torch.load(args.model_path)
    if 'state_dict' not in state_dict and 'model_state_dict' not in state_dict:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    
    return model, detection_postprocess

def val_data_prepare(args):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * OVERLAY_RATIO) for i in range(len(crop_size))]
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=-1)
    test_dataset = DetDatasetCSVRTest(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,)
    logger.info("Number of test samples: {}".format(len(val_loader.dataset)))
    logger.info("Number of test batches: {}".format(len(val_loader)))
    return val_loader

if __name__ == '__main__':
    args = get_args()
    
    setup_logging(level='info')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, detection_postprocess = prepare_validation(args)
    init_seed(args.seed)
    
    val_loader = val_data_prepare(args)
    
    exp_folder = os.path.join(SAVE_ROOT, 'val_temp')
    metrics = val(args = args,
                model = model,
                detection_postprocess=detection_postprocess,
                val_loader = val_loader, 
                device = device,
                image_spacing = IMAGE_SPACING,
                series_list_path=args.val_set,
                exp_folder=exp_folder)