# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
import numpy as np
from typing import Tuple
from networks.ResNet_3D_CPM import Resnet18, DetectionPostprocess, DetectionLoss
### data ###
from dataload.my_dataset_crop_mutli_window import TrainDataset, DetDataset
from dataload.collate import train_collate_fn, infer_collate_fn
from dataload.crop_multi_window import InstanceCrop
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform as transform
import torchvision
from torch.utils.tensorboard import SummaryWriter
### logic ###
from logic.train import train
from logic.val import val
from logic.utils import write_metrics, save_states, load_states
### optimzer ###
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
### postprocessing ###
from utils.logs import setup_logging
from utils.utils import init_seed, get_local_time_str_in_taiwan, write_yaml, load_yaml
from logic.early_stopping_save import EarlyStoppingSave

SAVE_ROOT = './save'
OVERLAY_RATIO = 0.25
IMAGE_SPACING = [1.0, 0.8, 0.8]
logger = logging.getLogger(__name__)

early_stopping = None

def get_args():
    parser = argparse.ArgumentParser()
    # Rraining settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size for training (default: 3)')
    parser.add_argument('--val_batch_size', type=int, default=2, help='input batch size for validation (default: 1)')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train (default: 250)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[64, 128, 128], help='crop size')
    # Resume
    parser.add_argument('--resume_folder', type=str, default='', help='resume folder')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    # Data
    parser.add_argument('--train_set', type=str, required=True, help='train_list')
    parser.add_argument('--val_set', type=str, required=True,help='val_list')
    parser.add_argument('--test_set', type=str, required=True,help='test_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of ground truth, if some nodule's depth < min_d, it will be ignored")
    parser.add_argument('--data_norm_method', type=str, default='scale', help='normalize method, mean_std or scale or none')
    # Learning rate
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_gamma', type=float, default=0.01, help='warmup gamma')
    parser.add_argument('--decay_cycle', type=int, default=1, help='decay cycle, 1 means no cycle')
    parser.add_argument('--decay_gamma', type=float, default=0.01, help='decay gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    # Loss hyper-parameters
    parser.add_argument('--lambda_cls', type=float, default=4.0, help='weights of seg')
    parser.add_argument('--lambda_offset', type=float, default=1.0,help='weights of offset')
    parser.add_argument('--lambda_shape', type=float, default=0.1, help='weights of reg')
    parser.add_argument('--lambda_iou', type=float, default=1.0, help='weights of iou loss')
    # Train hyper-parameters
    parser.add_argument('--pos_target_topk', type=int, default=5, help='topk grids assigned as positives')
    parser.add_argument('--pos_ignore_ratio', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples for each instance')
    parser.add_argument('--cls_num_hard', type=int, default=100, help='hard negative mining')
    parser.add_argument('--cls_fn_weight', type=float, default=4.0, help='weights of cls_fn')
    parser.add_argument('--cls_fn_threshold', type=float, default=0.8, help='threshold of cls_fn')
    # Val hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.7, help='fixed probability threshold for validation')
    # Network
    parser.add_argument('--norm_type', type=str, default='batchnorm', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', help='act type of network')
    parser.add_argument('--first_stride', nargs='+', type=int, default=[1, 2, 2], help='stride of the first layer')
    parser.add_argument('--n_blocks', nargs='+', type=int, default=[2, 3, 3, 3], help='number of blocks in each layer')
    parser.add_argument('--n_filters', nargs='+', type=int, default=[64, 96, 128, 160], help='number of filters in each layer')
    parser.add_argument('--stem_filters', type=int, default=32, help='number of filters in stem layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--no_se', action='store_true', default=False, help='not use se block')
    parser.add_argument('--aspp', action='store_true', default=False, help='use aspp')
    parser.add_argument('--dw_type', default='conv', help='downsample type, conv or maxpool')
    parser.add_argument('--up_type', default='deconv', help='upsample type, deconv or interpolate')
    # other
    parser.add_argument('--best_metrics', nargs='+', type=str, default=['froc_2_recall', 'f1_score'], help='metric for validation')
    parser.add_argument('--start_val_epoch', type=int, default=150, help='start to validate from this epoch')
    parser.add_argument('--exp_name', type=str, default='', metavar='str', help='experiment name')
    parser.add_argument('--save_model_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args

def prepare_training(args, device) -> Tuple[int, Resnet18, AdamW, GradualWarmupScheduler, DetectionPostprocess]:
    # build model
    detection_loss = DetectionLoss(crop_size = args.crop_size,
                                   pos_target_topk = args.pos_target_topk, 
                                   pos_ignore_ratio = args.pos_ignore_ratio,
                                   spacing = IMAGE_SPACING,
                                   cls_num_hard=args.cls_num_hard,
                                    cls_fn_weight=args.cls_fn_weight,
                                    cls_fn_threshold=args.cls_fn_threshold)
                                   
    model = Resnet18(norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type,
                     n_channels = 2,
                     se = not args.no_se, 
                     aspp = args.aspp,
                     first_stride=args.first_stride,
                     n_blocks=args.n_blocks,
                     n_filters=args.n_filters,
                     stem_filters=args.stem_filters,
                     dropout=args.dropout,
                     dw_type = args.dw_type,
                     up_type = args.up_type,
                     detection_loss = detection_loss,
                     device = device)
    detection_postprocess = DetectionPostprocess(topk = args.det_topk, 
                                                 threshold = args.det_threshold, 
                                                 nms_threshold = args.det_nms_threshold,
                                                 nms_topk = args.det_nms_topk,
                                                 crop_size = args.crop_size)
    start_epoch = 0
    model.to(device)
    # build optimizer and scheduler
    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    T_max = args.epochs // args.decay_cycle
    scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr * args.decay_gamma)
    scheduler_warm = GradualWarmupScheduler(optimizer, gamma=args.warmup_gamma, warmup_epochs=args.warmup_epochs, after_scheduler=scheduler_reduce)
    logger.info('Warmup learning rate from {:.1e} to {:.1e} for {} epochs and then reduce learning rate from {:.1e} to {:.1e} by cosine annealing with {} cycles'.format(args.lr * args.warmup_gamma, args.lr, args.warmup_epochs, args.lr, args.lr * args.decay_gamma, args.decay_cycle))

    if args.resume_folder != '':
        logger.info('Resume experiment "{}"'.format(os.path.dirname(args.resume_folder)))
        model_folder = os.path.join(args.resume_folder, 'model')
        
        # Get the latest model
        model_names = os.listdir(model_folder)
        model_epochs = [int(name.split('.')[0].split('_')[-1]) for name in model_names]
        start_epoch = model_epochs[np.argmax(model_epochs)]
        ckpt_path = os.path.join(model_folder, f'epoch_{start_epoch}.pth')
        logger.info('Load checkpoint from "{}"'.format(ckpt_path))

        load_states(ckpt_path, device, model, optimizer, scheduler_warm)
    
        # Resume best metric
        global early_stopping
        early_stopping = EarlyStoppingSave.load(save_dir=os.path.join(args.resume_folder, 'best'), target_metrics=args.best_metrics, model=model)
    
    elif args.pretrained_model_path != '':
        logger.info('Load model from "{}"'.format(args.pretrained_model_path))
        load_states(args.pretrained_model_path, device, model)
        
    return start_epoch, model, optimizer, scheduler_warm, detection_postprocess

def get_train_dataloder(args, blank_side=0) -> DataLoader:
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

    train_dataset = TrainDataset(series_list_path = args.train_set, crop_fn = crop_fn_train, image_spacing=IMAGE_SPACING, 
                                 transform_post = train_transform, min_d=args.min_d, norm_method=args.data_norm_method)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              collate_fn=train_collate_fn,
                              num_workers=min(args.batch_size, 4),
                              pin_memory=True,
                              drop_last=True, 
                              persistent_workers=True)
    logger.info("There are {} training samples and {} batches in '{}'".format(len(train_loader.dataset), len(train_loader), args.train_set))
    return train_loader

def get_val_test_dataloder(args) -> Tuple[DataLoader, DataLoader]:
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * OVERLAY_RATIO) for i in range(len(crop_size))]
    num_workers = min(args.val_batch_size, 4)
    
    if args.data_norm_method == 'none':
        pad_value = 0
    else:
        pad_value = -1
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value)
    val_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)

    test_dataset = DetDataset(series_list_path = args.test_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)
    
    logger.info("There are {} validation samples and {} batches in '{}'".format(len(val_loader.dataset), len(val_loader), args.val_set))
    logger.info("There are {} test samples and {} batches in '{}'".format(len(test_loader.dataset), len(test_loader), args.test_set))
    return val_loader, test_loader

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
        timestamp = get_local_time_str_in_taiwan()
        exp_folder = os.path.join(SAVE_ROOT, f'{timestamp}_{args.exp_name}')
    setup_logging(level='info', log_file=os.path.join(exp_folder, 'log.txt'))
    init_seed(args.seed)
    write_yaml(os.path.join(exp_folder, 'setting.yaml'), vars(args))
    
    # Prepare training
    model_save_dir = os.path.join(exp_folder, 'model')
    writer = SummaryWriter(log_dir = os.path.join(exp_folder, 'tensorboard'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch, model, optimizer, scheduler_warm, detection_postprocess = prepare_training(args, device)
    train_loader = get_train_dataloder(args)
    val_loader, test_loader = get_val_test_dataloder(args)
    
    if early_stopping is None:
        early_stopping = EarlyStoppingSave(target_metrics=args.best_metrics, save_dir=os.path.join(exp_folder, 'best'), model=model)
        
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train(args = args,
                            model = model,
                            optimizer = optimizer,
                            dataloader = train_loader, 
                            device = device)
        scheduler_warm.step()
        write_metrics(train_metrics, epoch, 'train', writer)
        for key, value in train_metrics.items():
            logger.info('==> Epoch: {} {}: {:.4f}'.format(epoch, key, value))
        # add learning rate to tensorboard
        logger.info('==> Epoch: {} lr: {:.6f}'.format(epoch, scheduler_warm.get_lr()[0]))
        write_metrics({'lr': scheduler_warm.get_lr()[0]}, epoch, 'train', writer)
        
        # Remove the checkpoint of epoch % save_model_interval != 0
        for i in range(epoch):
            ckpt_path = os.path.join(model_save_dir, 'epoch_{}.pth'.format(i))
            if ((i % args.save_model_interval != 0 or i == 0 or i < args.start_val_epoch) and os.path.exists(ckpt_path)):
                os.remove(ckpt_path)
        save_states(os.path.join(model_save_dir, f'epoch_{epoch}.pth'), model, optimizer, scheduler_warm)
        
        if epoch >= args.start_val_epoch: 
            val_metrics = val(args = args,
                            model = model,
                            detection_postprocess=detection_postprocess,
                            val_loader = val_loader, 
                            device = device,
                            image_spacing = IMAGE_SPACING,
                            series_list_path=args.val_set,
                            exp_folder=exp_folder,
                            epoch = epoch,
                            min_d=args.min_d)
            
            early_stopping.step(val_metrics, epoch)
            write_metrics(val_metrics, epoch, 'val', writer)
    
    # Test
    logger.info('Test the best model')
    test_save_dir = os.path.join(exp_folder, 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    for (target_metric, model_path), best_epoch in zip(early_stopping.get_best_model_paths().items(), early_stopping.best_epoch):
        logger.info('Load best model from "{}"'.format(model_path))
        load_states(model_path, device, model)
        test_metrics = val(args = args,
                        model = model,
                        detection_postprocess=detection_postprocess,
                        val_loader = test_loader,
                        device = device,
                        image_spacing = IMAGE_SPACING,
                        series_list_path=args.test_set,
                        exp_folder=exp_folder,
                        epoch = 'test_best_{}'.format(target_metric),
                        min_d=args.min_d)
        write_metrics(test_metrics, epoch, 'test/best_{}'.format(target_metric), writer)
        with open(os.path.join(test_save_dir, 'test_best_{}.txt'.format(target_metric)), 'w') as f:
            f.write('Best epoch: {}\n'.format(best_epoch))
            for key, value in test_metrics.items():
                f.write('{}: {:.4f}\n'.format(key, value))
    writer.close()