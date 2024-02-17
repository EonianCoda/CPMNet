# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
from typing import List, Tuple, Any, Dict

from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess, DetectionLoss
### data ###
from dataload.my_dataset_crop import DetDataset
from dataload.collate import infer_collate_fn
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform as transform

from logic.val import val
from logic.utils import load_states

from utils.logs import setup_logging
from utils.utils import init_seed, write_yaml

SAVE_ROOT = './save'
DEFAULT_CROP_SIZE = [64, 128, 128]
OVERLAY_RATIO = 0.25
IMAGE_SPACING = [1.0, 0.8, 0.8]
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 2)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=DEFAULT_CROP_SIZE, help='crop size')
    parser.add_argument('--model_path', type=str, default='')
    # data
    parser.add_argument('--val_set', type=str, required=True,help='val_list')
    # hyper-parameters
    parser.add_argument('--num_samples', type=int, default=5, help='sampling batch number in per sample')
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.7, help='fixed probability threshold for validation')
    
    # detection-hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    # network
    parser.add_argument('--norm_type', type=str, default='batchnorm', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', help='act type of network')
    parser.add_argument('--no_se', action='store_true', default=False, help='not use se')
    # other
    args = parser.parse_args()
    return args

def prepare_validation(args, device):
    # build model
    model = Resnet18(norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type, 
                     first_stride = (1, 2, 2), 
                     se = not args.no_se,
                     device = device)
    detection_postprocess = DetectionPostprocess(topk=args.det_topk, 
                                                 threshold=args.det_threshold, 
                                                 nms_threshold=args.det_nms_threshold,
                                                 nms_topk=args.det_nms_topk,
                                                 crop_size=args.crop_size)
    
    logger.info('Load model from "{}"'.format(args.model_path))
    model.to(device)
    load_states(args.model_path, device, model)
    
    return model, detection_postprocess

def val_data_prepare(args):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * OVERLAY_RATIO) for i in range(len(crop_size))]
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=-1)
    test_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.batch_size, collate_fn=infer_collate_fn, pin_memory=True)
    logger.info("There are {} samples in the val set".format(len(val_loader.dataset)))
    return val_loader

if __name__ == '__main__':
    args = get_args()
    exp_folder = os.path.dirname(args.model_path)
    exp_folder = os.path.join(exp_folder, 'val_temp')
    setup_logging(log_file=os.path.join(exp_folder, 'val.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, detection_postprocess = prepare_validation(args, device)
    init_seed(args.seed)
    
    val_loader = val_data_prepare(args)
    
    
    write_yaml(os.path.join(exp_folder, 'val_config.yaml'), args)
    logger.info('Save validation results to "{}"'.format(exp_folder))
    metrics = val(args = args,
                model = model,
                detection_postprocess=detection_postprocess,
                val_loader = val_loader, 
                device = device,
                image_spacing = IMAGE_SPACING,
                series_list_path=args.val_set,
                exp_folder=exp_folder)
    
    with open(os.path.join(exp_folder, 'val_metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write('{}: {}\n'.format(k, v))