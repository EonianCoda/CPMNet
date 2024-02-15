# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict

from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess, DetectionLoss
### data ###
from dataload.my_dataset_crop import DetDatasetCSVR, DetDatasetCSVRTest, collate_fn_dict
from dataload.crop import InstanceCrop
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import torch.nn as nn
import transform as transform
import torchvision
from torch.utils.tensorboard import SummaryWriter
### optimzer
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
### postprocessing
from logic.train import train, write_metrics, save_states
from logic.val import val
from inference.evaluation import DEFAULT_FP_RATIOS
### utils
from utils.logs import setup_logging
from utils.utils import init_seed, get_local_time_in_taiwan, write_yaml, load_yaml

SAVE_ROOT = './save'
DEFAULT_CROP_SIZE = [64, 128, 128]
OVERLAY_RATIO = 0.25
IMAGE_SPACING = [1.0, 0.8, 0.8]
logger = logging.getLogger(__name__)
best_epoch = 0
best_metric = 0.0

def get_args():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--seed', type=int, default=0, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='use pin memory')
    parser.add_argument('--num_workers', type=int, default=1, metavar='S', help='num_workers (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=DEFAULT_CROP_SIZE, help='crop size')
    # resume
    parser.add_argument('--resume_folder', type=str, default='', metavar='str', help='resume folder')
    parser.add_argument('--pretrained_model_path', type=str, default='', metavar='str')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='train_list')
    parser.add_argument('--val_set', type=str, required=True,help='val_list')
    # hyper-parameters
    parser.add_argument('--lambda_cls', type=float, default=4.0, help='weights of seg')
    parser.add_argument('--lambda_offset', type=float, default=1.0,help='weights of offset')
    parser.add_argument('--lambda_shape', type=float, default=0.1, help='weights of reg')
    parser.add_argument('--lambda_iou', type=float, default=1.0, help='weights of iou loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--pos_target_topk', type=int, default=5, metavar='N', help='topk grids assigned as positives')
    parser.add_argument('--pos_ignore_ratio', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=6, metavar='N', help='sampling batch number in per sample')
    
    # val-hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.7, help='fixed probability threshold for validation')
    # network
    parser.add_argument('--norm_type', type=str, default='batchnorm', metavar='N', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', metavar='N', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', metavar='N', help='act type of network')
    parser.add_argument('--se', action='store_true', default=False, help='use se block')
    # other
    parser.add_argument('--metric', type=str, default='f1_score', metavar='str', help='metric for validation')
    parser.add_argument('--start_val_epoch', type=int, default=150, help='start to validate from this epoch')
    parser.add_argument('--exp_name', type=str, default='', metavar='str', help='experiment name')
    parser.add_argument('--save_model_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--best_model_metric', type=str, default='sens', metavar='str', help='best model metric')
    parser.add_argument('--best_model_fp_ratio', type=float, default=2.0, metavar='N', help='best model fp ratio')
    args = parser.parse_args()
    return args

def prepare_training(args):
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
    
    start_epoch = 0
    if args.resume_folder != '':
        logger.info('Resume experiment "{}"'.format(os.path.dirname(args.resume_folder)))
        
        model_folder = os.path.join(args.resume_folder, 'model')
        
        # Resume best metric
        if os.path.exists(os.path.join(args.resume_folder, 'best_epoch.txt')):
            global best_epoch
            global best_metric
            with open(os.path.join(args.resume_folder, 'best_epoch.txt'), 'r') as f:
                best_epoch = int(f.readline().split(':')[-1])
                best_metric = float(f.readline().split(':')[-1])
            logger.info('Best epoch: {}, Best metric: {:.4f}'.format(best_epoch, best_metric))
        # Get the latest model
        model_names = os.listdir(model_folder)
        model_epochs = [int(name.split('.')[0].split('_')[-1]) for name in model_names]
        start_epoch = model_epochs[np.argmax(model_epochs)]
        
        model_path = os.path.join(model_folder, f'epoch_{start_epoch}.pth')
        
        logger.info('Load model from "{}"'.format(model_path))
        model.to(device)
        # build optimizer
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=2, after_scheduler=scheduler_reduce)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])       
        scheduler_warm.load_state_dict(state_dict['scheduler'])
        
    elif args.pretrained_model_path != '':
        logger.info('Load model from "{}"'.format(args.pretrained_model_path))
        state_dict = torch.load(args.pretrained_model_path)
        if 'state_dict' not in state_dict:
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict['state_dict'])
        model.to(device)
        # build optimizer
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=2, after_scheduler=scheduler_reduce)
    else:
        model.to(device)
        # build optimizer
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_warm = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=2, after_scheduler=scheduler_reduce)
    
    return start_epoch, model, optimizer, scheduler_warm, detection_postprocess

def get_train_loader(args, blank_side=0):
    crop_size = args.crop_size
    transform_list_train = [transform.RandomFlip(flip_depth=True, flip_height=True, flip_width=True, p=0.5),
                            transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=False, trans_zy=False),
                            transform.Pad(output_size=crop_size),
                            transform.RandomCrop(output_size=crop_size, pos_ratio=0.9),
                            transform.CoordToAnnot(blank_side=blank_side)]
    
    train_transform = torchvision.transforms.Compose(transform_list_train)

    crop_fn_train = InstanceCrop(crop_size=crop_size, tp_ratio=0.75, rand_trans=[10, 20, 20], 
                                 rand_rot=[20, 0, 0], rand_space=[0.9, 1.2],sample_num=args.num_samples,
                                 blank_side=blank_side, instance_crop=True)

    train_dataset = DetDatasetCSVR(series_list_path = args.train_set,
                                   crop_fn = crop_fn_train,
                                   image_spacing=IMAGE_SPACING,
                                   transform_post = train_transform)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              collate_fn=collate_fn_dict,
                              num_workers=args.num_workers, 
                              pin_memory=args.pin_memory, 
                              drop_last=True,
                              persistent_workers=True)
    logger.info("Number of training samples: {}".format(len(train_loader.dataset)))
    logger.info("Number of training batches: {}".format(len(train_loader)))
    return train_loader

def get_val_loader(args):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * OVERLAY_RATIO) for i in range(len(crop_size))]
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=-1)
    test_dataset = DetDatasetCSVRTest(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,)
    logger.info("Number of test samples: {}".format(len(test_loader.dataset)))
    logger.info("Number of test batches: {}".format(len(test_loader)))
    return test_loader

def convert_to_standard_csv(csv_path, save_dir, state, spacing):
    '''
    convert [seriesuid	coordX	coordY	coordZ	w	h	d] to 
    'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'
    spacing:[z, y, x]
    '''
    column_order = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'w', 'h', 'd']
    gt_list = []
    csv_file = pd.read_csv(csv_path)
    seriesuid = csv_file['seriesuid']
    coordX, coordY, coordZ = csv_file['coordX'], csv_file['coordY'], csv_file['coordZ']
    w, h, d = csv_file['w'], csv_file['h'], csv_file['d']
    clean_seriesuid = []
    for j in range(seriesuid.shape[0]):
        if seriesuid[j] not in clean_seriesuid: 
            clean_seriesuid.append(seriesuid[j])
        gt_list.append([seriesuid[j], coordX[j], coordY[j], coordZ[j], w[j]/spacing[2], h[j]/spacing[1], d[j]/spacing[0]])
    df = pd.DataFrame(gt_list, columns=column_order)
    df.to_csv(os.path.join(save_dir, 'annotation_{}.csv'.format(state)), index=False)
    df = pd.DataFrame(clean_seriesuid)
    df.to_csv(os.path.join(save_dir, 'seriesuid_{}.csv'.format(state)), index=False, header=None)

if __name__ == '__main__':
    args = get_args()
    
    if args.resume_folder != '': # resume training
        exp_folder = args.resume_folder
        setting_yaml_path = os.path.join(exp_folder, 'setting.yaml')
        setting = load_yaml(setting_yaml_path)
        for key, value in setting.items():
            if key != 'resume_folder':
                setattr(args, key, value)
    else:     
        cur_time = get_local_time_in_taiwan()
        timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                                cur_time.month, 
                                                cur_time.day, 
                                                cur_time.hour, 
                                                cur_time.minute)
        exp_folder = os.path.join(SAVE_ROOT, timestamp + '_' + args.exp_name)
        
    setup_logging(level='info', log_file=os.path.join(exp_folder, 'log.txt'))
    logger.info("The number of GPUs: {}".format(torch.cuda.device_count()))
    writer = SummaryWriter(log_dir = os.path.join(exp_folder, 'tensorboard'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    write_yaml(os.path.join(exp_folder, 'setting.yaml'), vars(args))
    logger.info('The learning rate: {}'.format(args.lr))
    logger.info('The batch size: {}'.format(args.batch_size))
    logger.info('The Crop Size: [{}, {}, {}]'.format(args.crop_size[0], args.crop_size[1], args.crop_size[2]))
    logger.info('positive_target_topk: {}, lambda_cls: {}, lambda_shape: {}, lambda_offset: {}, lambda_iou: {},, num_samples: {}'.format(args.pos_target_topk, args.lambda_cls, args.lambda_shape, args.lambda_offset, args.lambda_iou, args.num_samples))
    logger.info('norm type: {}, head norm: {}, act_type: {}, using se block: {}'.format(args.norm_type, args.head_norm, args.act_type, args.se))
    start_epoch, model, optimizer, scheduler_warm, detection_postprocess = prepare_training(args)

    if args.best_model_metric not in ['f1','sens','prec']:
        raise ValueError('The best model metric should be in ["f1","sens","prec"], but got {}'.format(args.best_model_metric))
    if args.best_model_fp_ratio not in DEFAULT_FP_RATIOS:
        raise ValueError('The best model fp ratio should be in {}, but got {}'.format(DEFAULT_FP_RATIOS, args.best_model_fp_ratio))
    
    init_seed(args.seed)
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)
    
    model_save_folder = os.path.join(exp_folder, 'model')
    os.makedirs(model_save_folder, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train(args = args,
                            model = model,
                            optimizer = optimizer,
                            scheduler_warm = scheduler_warm,
                            train_loader = train_loader, 
                            device = device)
        write_metrics(train_metrics, epoch, 'Train', writer)
        for key, value in train_metrics.items():
            logger.info('====> Epoch: {} loss: {:.4f}'.format(epoch, value))
        # Remove the checkpoint of epoch % save_model_interval != 0
        for i in range(epoch):
            ckpt_path = os.path.join(model_save_folder, 'epoch_{}.pth'.format(i))
            if (i % args.save_model_interval != 0 or i == 0) and os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        save_states(model, optimizer, scheduler_warm, os.path.join(model_save_folder, f'epoch_{epoch}.pth'))
        
        if epoch >= args.start_val_epoch: 
            all_metrics, inter_btp_mean, inter_points = val(args = args,
                                                            model = model,
                                                            val_loader = val_loader,
                                                            detection_postprocess = detection_postprocess,
                                                            device = device,
                                                            epoch = epoch,
                                                            image_spacing = IMAGE_SPACING,
                                                            series_list_path = args.val_set)
            idx = DEFAULT_FP_RATIOS.index(args.best_model_fp_ratio)
            if inter_points[args.best_model_metric][idx] > best_metric:
                best_metric = inter_points[args.best_model_metric][idx]
                best_epoch = epoch
                save_states(model, optimizer, scheduler_warm, os.path.join(model_save_folder, 'best.pth'))
                with open(os.path.join(exp_folder, 'best_epoch.txt'), 'w') as f:
                    f.write('best_epoch: {}\n'.format(best_epoch))
                    f.write('best_metric: {:.4f}\n'.format(best_metric))
            
            for i, fp_ratio in enumerate(DEFAULT_FP_RATIOS):
                line = '====> fp_ratio:{:.4f}'.format(fp_ratio)
                for key, value in inter_points.items():
                    line += ' {}: {:.4f}'.format(key, value[i])
                    write_metrics(value[i], epoch, 'Val/fp_ratio_{:.3f}_{}'.format(fp_ratio, key), writer)
                logger.info(line)
            
            line = '====> '
            for key, value in inter_points.items():
                line += ' {mean }: {:.4f}'.format(key, np.mean(value))
            logger.info(line)