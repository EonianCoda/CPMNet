# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging

from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess
### data ###
from dataload.dataset import DetDataset
from dataload.utils import get_image_padding_value
from dataload.collate import infer_collate_fn
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform as transform

from logic.val import val
from logic.utils import load_states

from utils.logs import setup_logging
from utils.utils import init_seed, write_yaml

SAVE_ROOT = './save'
OVERLAY_RATIO = 0.25
IMAGE_SPACING = [1.0, 0.8, 0.8]
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 2)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[64, 128, 128], help='crop size')
    parser.add_argument('--model_path', type=str, default='')
    # data
    parser.add_argument('--val_set', type=str, default='./data/all_client_test.txt', help='val_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of ground truth, if some nodule's depth < min_d, it will be ignored")
    parser.add_argument('--data_norm_method', type=str, default='scale', help='normalize method, mean_std or scale or none')
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
    parser.add_argument('--first_stride', nargs='+', type=int, default=[1, 2, 2], help='stride of the first layer')
    parser.add_argument('--n_blocks', nargs='+', type=int, default=[2, 3, 3, 3], help='number of blocks in each layer')
    parser.add_argument('--n_filters', nargs='+', type=int, default=[64, 96, 128, 160], help='number of filters in each layer')
    parser.add_argument('--stem_filters', type=int, default=32, help='number of filters in stem layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--no_se', action='store_true', default=False, help='not use se')
    parser.add_argument('--aspp', action='store_true', default=False, help='use aspp')
    parser.add_argument('--dw_type', default='conv', help='downsample type, conv or maxpool')
    parser.add_argument('--up_type', default='deconv', help='upsample type, deconv or interpolate')
    # other
    args = parser.parse_args()
    return args

def prepare_validation(args, device):
    # build model
    model = Resnet18(norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type, 
                     first_stride = args.first_stride,
                     se = not args.no_se,
                     aspp = args.aspp,
                     n_blocks=args.n_blocks,
                     n_filters=args.n_filters,
                     stem_filters=args.stem_filters,
                     dropout=args.dropout,
                     dw_type = args.dw_type,
                     up_type = args.up_type,
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
    
    pad_value = get_image_padding_value(args.data_norm_method)
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value)
    test_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
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
    logger.info('Val set: "{}"'.format(args.val_set))
    metrics = val(args = args,
                model = model,
                detection_postprocess=detection_postprocess,
                val_loader = val_loader, 
                device = device,
                image_spacing = IMAGE_SPACING,
                series_list_path=args.val_set,
                exp_folder=exp_folder,
                min_d=args.min_d)
    
    with open(os.path.join(exp_folder, 'val_metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write('{}: {}\n'.format(k, v))