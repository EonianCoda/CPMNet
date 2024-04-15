# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import os
import logging
import pickle
import numpy as np
from typing import Tuple
from networks.ResNet_3D_CPM_semi_focal_soft import Resnet18, DetectionPostprocess, DetectionLoss, Unsupervised_DetectionLoss
### data ###
from dataload.dataset_semi_focal import TrainDataset, DetDataset, UnLabeledDataset
from dataload.utils import get_image_padding_value
from dataload.collate import train_collate_fn, infer_collate_fn, unlabeled_focal_train_collate_fn
from dataload.split_combine import SplitComb
from torch.utils.data import DataLoader
import transform as transform
import torchvision
from torch.utils.tensorboard import SummaryWriter
### logic ###
from logic.semi_focal_soft_train import train
from logic.val import val
from logic.pseudo_label import gen_pseu_labels
from logic.utils import write_metrics, save_states, load_states, get_memory_format
from logic.early_stopping_save import EarlyStoppingSave
### optimzer ###
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
from optimizer.ema import EMA
### postprocessing ###
from utils.logs import setup_logging
from utils.utils import init_seed, get_local_time_str_in_taiwan, write_yaml, load_yaml
from config import SAVE_ROOT, DEFAULT_OVERLAP_RATIO, IMAGE_SPACING, NODULE_TYPE_DIAMETERS

logger = logging.getLogger(__name__)

early_stopping = None

def get_args():
    parser = argparse.ArgumentParser()
    # Training settings
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--val_mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=6, help='input batch size for training (default: 6)')
    parser.add_argument('--unlabeled_batch_size', type=int, default=6, help='input batch size for training (default: 6)')
    parser.add_argument('--val_batch_size', type=int, default=2, help='input batch size for validation (default: 2)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 250)')
    parser.add_argument('--crop_size', nargs='+', type=int, default=[96, 96, 96], help='crop size')
    parser.add_argument('--overlap_ratio', type=float, default=DEFAULT_OVERLAP_RATIO, help='overlap ratio')
    # Resume
    parser.add_argument('--resume_folder', type=str, default='', help='resume folder')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    # Data
    parser.add_argument('--train_set', type=str, help='train_list')
    parser.add_argument('--val_set', type=str, help='val_list')
    parser.add_argument('--test_set', type=str, help='test_list')
    parser.add_argument('--unlabeled_train_set', type=str, help='unlabeled_train_list')
    parser.add_argument('--min_d', type=int, default=0, help="min depth of nodule, if some nodule's depth < min_d, it will be` ignored")
    parser.add_argument('--min_size', type=int, default=5, help="min size of nodule, if some nodule's size < min_size, it will be ignored")
    parser.add_argument('--data_norm_method', type=str, default='none', help='normalize method, mean_std or scale or none')
    parser.add_argument('--memory_format', type=str, default='channels_first') # for speed up
    
    parser.add_argument('--crop_partial', action='store_true', default=False, help='crop partial nodule')
    parser.add_argument('--crop_tp_iou', type=float, default=0.5, help='iou threshold for crop tp use if crop_partial is True')
    
    parser.add_argument('--use_bg', action='store_true', default=False, help='use background(healthy lung) in training')
    # Data Augmentation
    parser.add_argument('--tp_ratio', type=float, default=0.75, help='positive ratio in instance crop')
    parser.add_argument('--use_crop', action='store_true', default=False, help='use crop augmentation')
    parser.add_argument('--not_use_itk_rotate', action='store_true', default=False, help='not use itk rotate')
    parser.add_argument('--rand_rot', nargs='+', type=int, default=[30, 0, 0], help='random rotate')
    parser.add_argument('--use_rand_spacing', action='store_true', default=False, help='use random spacing')
    parser.add_argument('--rand_spacing', nargs='+', type=float, default=[0.9, 1.1], help='random spacing range, [min, max]')
    # Learning rate
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_gamma', type=float, default=0.01, help='warmup gamma')
    parser.add_argument('--decay_cycle', type=int, default=1, help='decay cycle, 1 means no cycle')
    parser.add_argument('--decay_gamma', type=float, default=0.05, help='decay gamma')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--apply_ema', action='store_true', default=False, help='apply ema')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay')
    # Loss hyper-parameters
    parser.add_argument('--lambda_cls', type=float, default=4.0, help='weights of seg')
    parser.add_argument('--lambda_offset', type=float, default=1.0,help='weights of offset')
    parser.add_argument('--lambda_shape', type=float, default=0.1, help='weights of reg')
    parser.add_argument('--lambda_iou', type=float, default=1.0, help='weights of iou loss')
    
    parser.add_argument('--lambda_pseu', type=float, default=1.0, help='weights of pseudo label')
    parser.add_argument('--lambda_pseu_cls', type=float, default=4.0, help='weights of seg')
    parser.add_argument('--lambda_pseu_offset', type=float, default=1.0,help='weights of offset')
    parser.add_argument('--lambda_pseu_shape', type=float, default=0.1, help='weights of reg')
    parser.add_argument('--lambda_pseu_iou', type=float, default=1.0, help='weights of iou loss')
    # Train hyper-parameters
    parser.add_argument('--pos_target_topk', type=int, default=7, help='topk grids assigned as positives')
    parser.add_argument('--pos_ignore_ratio', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples for each instance')
    parser.add_argument('--unlabeled_num_samples', type=int, default=5, help='number of samples for each instance')
    parser.add_argument('--iters_to_accumulate', type=int, default=1, help='number of batches to wait before updating the weights')
    parser.add_argument('--cls_num_neg', type=int, default=10000, help='number of negatives (-1 means all)')
    parser.add_argument('--cls_num_hard', type=int, default=100, help='hard negative mining')
    parser.add_argument('--cls_fn_weight', type=float, default=4.0, help='weights of cls_fn')
    parser.add_argument('--cls_fn_threshold', type=float, default=0.8, help='threshold of cls_fn')
    # Semi hyper-parameters
    parser.add_argument('--pos_target_topk_pseu', type=int, default=7, help='topk grids assigned as positives')
    parser.add_argument('--pseudo_label_threshold', type=float, default=0.6, help='threshold of pseudo label')
    parser.add_argument('--pseudo_background_threshold', type=float, default=0.85, help='threshold of pseudo background') #TODO Currently not used
    parser.add_argument('--semi_ema_alpha', type=int, default=0.999, help='alpha of ema')
    parser.add_argument('--num_aug', type=int, default=4, help='number of augmentation times of each unlabeled sample')
    parser.add_argument('--sharpen_temperature', type=float, default=0.5, help='temperature of sharpen')
    # Val hyper-parameters
    parser.add_argument('--det_topk', type=int, default=60, help='topk detections')
    parser.add_argument('--det_threshold', type=float, default=0.15, help='detection threshold')
    parser.add_argument('--det_nms_threshold', type=float, default=0.05, help='detection nms threshold')
    parser.add_argument('--det_nms_topk', type=int, default=20, help='detection nms topk')
    parser.add_argument('--val_iou_threshold', type=float, default=0.1, help='iou threshold for validation')
    parser.add_argument('--val_fixed_prob_threshold', type=float, default=0.65, help='fixed probability threshold for validation')
    # Network
    parser.add_argument('--norm_type', type=str, default='batchnorm', help='norm type of backbone')
    parser.add_argument('--head_norm', type=str, default='batchnorm', help='norm type of head')
    parser.add_argument('--act_type', type=str, default='ReLU', help='act type of network')
    parser.add_argument('--first_stride', nargs='+', type=int, default=[2, 2, 2], help='stride of the first layer')
    parser.add_argument('--n_blocks', nargs='+', type=int, default=[2, 3, 3, 3], help='number of blocks in each layer')
    parser.add_argument('--n_filters', nargs='+', type=int, default=[64, 96, 128, 160], help='number of filters in each layer')
    parser.add_argument('--stem_filters', type=int, default=32, help='number of filters in stem layer')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--no_se', action='store_true', default=False, help='not use se block')
    parser.add_argument('--aspp', action='store_true', default=False, help='use aspp')
    parser.add_argument('--dw_type', default='conv', help='downsample type, conv or maxpool')
    parser.add_argument('--up_type', default='deconv', help='upsample type, deconv or interpolate')
    # other
    parser.add_argument('--nodule_size_mode', type=str, default='seg_size', help='nodule size mode, seg_size or dhw')
    parser.add_argument('--max_workers', type=int, default=4, help='max number of workers, num_workers = min(batch_size, max_workers)')
    parser.add_argument('--best_metrics', nargs='+', type=str, default=['froc_2_recall', 'froc_mean_recall'], help='metric for validation')
    parser.add_argument('--start_val_epoch', type=int, default=150, help='start to validate from this epoch')
    parser.add_argument('--val_interval', type=int, default=1, help='validate interval')
    parser.add_argument('--exp_name', type=str, default='', metavar='str', help='experiment name')
    parser.add_argument('--save_model_interval', type=int, default=10, help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args

def add_weight_decay(net, weight_decay):
    """no weight decay on bias and normalization layer
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        # skip bias and bn layer
        if ".norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay}]

def build_model(args) -> Resnet18:
    model = Resnet18(norm_type = args.norm_type,
                     head_norm = args.head_norm, 
                     act_type = args.act_type, 
                     se = not args.no_se, 
                     aspp = args.aspp,
                     first_stride=args.first_stride,
                     n_blocks=args.n_blocks,
                     n_filters=args.n_filters,
                     stem_filters=args.stem_filters,
                     dropout=args.dropout,
                     dw_type = args.dw_type,
                     up_type = args.up_type)
    return model

def prepare_training(args, device, num_training_steps):
    # build model
    model = build_model(args)
    detection_loss = DetectionLoss(crop_size = args.crop_size,
                                   pos_target_topk = args.pos_target_topk, 
                                   pos_ignore_ratio = args.pos_ignore_ratio,
                                   cls_num_neg=args.cls_num_neg,
                                   cls_num_hard = args.cls_num_hard,
                                   cls_fn_weight = args.cls_fn_weight,
                                   cls_fn_threshold = args.cls_fn_threshold)

    unsupervised_detection_loss = Unsupervised_DetectionLoss(crop_size = args.crop_size,
                                                            pos_target_topk = args.pos_target_topk_pseu, 
                                                            pos_ignore_ratio = args.pos_ignore_ratio,
                                                            cls_num_neg=args.cls_num_neg,
                                                            cls_num_hard = args.cls_num_hard,
                                                            cls_fn_weight = args.cls_fn_weight,
                                                            cls_fn_threshold = args.cls_fn_threshold,
                                                            pos_threshold = args.pseudo_label_threshold)

    detection_postprocess = DetectionPostprocess(topk = args.det_topk, 
                                                 threshold = args.det_threshold, 
                                                 nms_threshold = args.det_nms_threshold,
                                                 nms_topk = args.det_nms_topk,
                                                 crop_size = args.crop_size)
    
    start_epoch = 0
    memory_format = get_memory_format(getattr(args, 'memory_format', 'channels_first'))
    model = model.to(device=device, memory_format=memory_format)

    # build optimizer and scheduler
    params = add_weight_decay(model, args.weight_decay)
    optimizer = AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    
    T_max = args.epochs // args.decay_cycle
    scheduler_reduce = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=args.lr * args.decay_gamma)
    scheduler_warm = GradualWarmupScheduler(optimizer, gamma=args.warmup_gamma, warmup_epochs=args.warmup_epochs, after_scheduler=scheduler_reduce)
    logger.info('Warmup learning rate from {:.1e} to {:.1e} for {} epochs and then reduce learning rate from {:.1e} to {:.1e} by cosine annealing with {} cycles'.format(args.lr * args.warmup_gamma, args.lr, args.warmup_epochs, args.lr, args.lr * args.decay_gamma, args.decay_cycle))

    # Build EMA
    if args.apply_ema:
        ema_warmup_steps = int(args.start_val_epoch * num_training_steps * 1.2 + 1)
        logger.info('Apply EMA with decay: {:.4f}, warmup steps: {}'.format(args.ema_decay, ema_warmup_steps))
        ema = EMA(model, momentum = args.ema_decay, warmup_steps = ema_warmup_steps)
        ema.register()
    else:
        ema = None

    if args.resume_folder != '':
        logger.info('Resume experiment "{}"'.format(os.path.dirname(args.resume_folder)))
        model_folder = os.path.join(args.resume_folder, 'model')
        
        # Get the latest model
        model_names = os.listdir(model_folder)
        model_epochs = [int(name.split('.')[0].split('_')[-1]) for name in model_names]
        start_epoch = model_epochs[np.argmax(model_epochs)]
        ckpt_path = os.path.join(model_folder, f'epoch_{start_epoch}.pth')
        logger.info('Load checkpoint from "{}"'.format(ckpt_path))

        load_states(ckpt_path, device, model, optimizer, scheduler_warm, ema)
        # Resume best metric
        global early_stopping
        early_stopping = EarlyStoppingSave.load(save_dir=os.path.join(args.resume_folder, 'best'), target_metrics=args.best_metrics, model=model)
    
    elif args.pretrained_model_path != '':
        logger.info('Load model from "{}"'.format(args.pretrained_model_path))
        load_states(args.pretrained_model_path, device, model)
        
    return start_epoch, model, detection_loss, unsupervised_detection_loss, optimizer, scheduler_warm, ema, detection_postprocess

def build_train_augmentation(args, crop_size: Tuple[int, int, int], pad_value: int, blank_side: int):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
        
    transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
    transform_list_train.append(transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=rot_zx, trans_zy=rot_zy))
        
    if args.use_crop:
        transform_list_train.append(transform.RandomCrop(p=0.3, crop_ratio=0.95, ctr_margin=10, pad_value=pad_value))
        
    transform_list_train.append(transform.CoordToAnnot())
                            
    logger.info('Augmentation: random flip: True, random transpose: {}, random crop: {}'.format([True, rot_zy, rot_zx], args.use_crop))
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

def build_strong_augmentation(args, crop_size: Tuple[int, int, int], pad_value: int, blank_side: int):
    rot_zy = (crop_size[0] == crop_size[1] == crop_size[2])
    rot_zx = (crop_size[0] == crop_size[1] == crop_size[2])
    
    transform_list_train = [transform.RandomFlip(p=0.5, flip_depth=True, flip_height=True, flip_width=True)]
                            # transform.RandomBlur(p=0.5, sigma_range=(0.4, 0.6)),
                            # transform.RandomNoise(p=0.5, gamma_range=(4e-4, 8e-4))]
    transform_list_train.append(transform.RandomTranspose(p=0.5, trans_xy=True, trans_zx=rot_zx, trans_zy=rot_zy))
        
    if args.use_crop:
        transform_list_train.append(transform.RandomCrop(p=0.3, crop_ratio=0.95, ctr_margin=10, pad_value=pad_value))
        
    transform_list_train.append(transform.CoordToAnnot())
                            
    train_transform = torchvision.transforms.Compose(transform_list_train)
    return train_transform

def get_train_dataloder(args, blank_side=0) -> DataLoader:
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    rand_trans = [int(s * 2/3) for s in overlap_size]
    pad_value = get_image_padding_value(args.data_norm_method)
    
    logger.info('Crop size: {}, overlap size: {}, rand_trans: {}, pad value: {}, tp_ratio: {:.3f}'.format(crop_size, overlap_size, rand_trans, pad_value, args.tp_ratio))
    
    # Build crop function
    from dataload.crop import InstanceCrop
    from dataload.crop_semi import InstanceCrop as InstanceCrop_semi
    crop_fn_train_l = InstanceCrop(crop_size=crop_size, overlap_ratio=args.overlap_ratio, tp_ratio=args.tp_ratio, rand_trans=rand_trans, rand_rot=args.rand_rot,
                                sample_num=args.num_samples, blank_side=blank_side, instance_crop=True)
    crop_fn_train_u = InstanceCrop_semi(crop_size=crop_size, overlap_ratio=args.overlap_ratio, rand_trans=rand_trans, rand_rot=args.rand_rot,
                                sample_num=args.unlabeled_num_samples, blank_side=blank_side)
    mmap_mode = None
    logger.info('Use itk rotate {}'.format(args.rand_rot))

    # Build labeled dataloader
    train_transform = build_train_augmentation(args, crop_size, pad_value, blank_side)
    train_dataset_l = TrainDataset(series_list_path = args.train_set, crop_fn = crop_fn_train_l, image_spacing=IMAGE_SPACING, transform_post = train_transform, 
                                 min_d=args.min_d, min_size = args.min_size, use_bg = args.use_bg, norm_method=args.data_norm_method, mmap_mode=mmap_mode)
    train_loader_l = DataLoader(train_dataset_l, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=min(args.batch_size, args.max_workers), 
                                pin_memory=True, drop_last=True, persistent_workers=True)
    
    # Build unlabeled dataloader
    strong_aug = build_strong_augmentation(args, crop_size, pad_value, blank_side)
    train_dataset_u = UnLabeledDataset(series_list_path = args.unlabeled_train_set, crop_fn = crop_fn_train_u, image_spacing=IMAGE_SPACING, strong_aug=strong_aug, 
                                       min_d=args.min_d, min_size = args.min_size, norm_method=args.data_norm_method, mmap_mode=mmap_mode, num_aug=args.num_aug)
    train_loader_u = DataLoader(train_dataset_u, batch_size=args.unlabeled_batch_size, shuffle=True, collate_fn=unlabeled_focal_train_collate_fn, 
                                num_workers=min(args.unlabeled_batch_size, args.max_workers), pin_memory=True, drop_last=True)
    
    # Build unlabeled detection dataloader for generating pseudo labels
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=False)
    det_dataset_u = DetDataset(series_list_path = args.unlabeled_train_set, 
                               SplitComb=split_comber, 
                               image_spacing=IMAGE_SPACING, 
                               norm_method=args.data_norm_method)
    det_loader_u = DataLoader(det_dataset_u, batch_size=args.val_batch_size, shuffle=False, collate_fn=infer_collate_fn, 
                              num_workers=min(args.unlabeled_batch_size, args.max_workers), pin_memory=True, drop_last=False)
    
    logger.info("There are {} training labeled samples and {} batches in '{}'".format(len(train_loader_l.dataset), len(train_loader_l), args.train_set))
    logger.info("There are {} training unlabeled samples and {} batches in '{}'".format(len(train_loader_u.dataset), len(train_loader_u), args.unlabeled_train_set))
    
    return train_loader_l, train_loader_u, det_loader_u

def get_val_test_dataloder(args) -> Tuple[DataLoader, DataLoader]:
    crop_size = args.crop_size
    overlap_size = (np.array(crop_size) * args.overlap_ratio).astype(np.int32).tolist()
    pad_value = get_image_padding_value(args.data_norm_method)
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=pad_value, do_padding=False)
    
    # Build val dataloader
    val_dataset = DetDataset(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=min(args.val_batch_size, args.max_workers), pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)

    # Build test dataloader
    test_dataset = DetDataset(series_list_path = args.test_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING, norm_method=args.data_norm_method)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=min(args.val_batch_size, args.max_workers), 
                             pin_memory=True, drop_last=False, collate_fn=infer_collate_fn)
    
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
    train_loader_l, train_loader_u, det_loader_u = get_train_dataloder(args)
    start_epoch, model, detection_loss, unsupervised_detection_loss, optimizer, scheduler_warm, ema, detection_postprocess = prepare_training(args, device, len(train_loader_u))
    val_loader, test_loader = get_val_test_dataloder(args)
    
    if early_stopping is None:
        early_stopping = EarlyStoppingSave(target_metrics=args.best_metrics, save_dir=os.path.join(exp_folder, 'best'), model=model)

    # pseu_labels = gen_pseu_labels(model = model_s,
    #                                 dataloader = det_loader_u,
    #                                 device = device,
    #                                 detection_postprocess = detection_postprocess,
    #                                 prob_threshold = 0.5,
    #                                 mixed_precision = args.val_mixed_precision,
    #                                 memory_format = args.memory_format)
        
    # pseu_label_save_path = os.path.join(exp_folder, 'pseu_labels.pkl')
    # with open(pseu_label_save_path, 'wb') as f:
    #     pickle.dump(pseu_labels, f)
    # if not os.path.exists('./pseu_labels.pkl'):
    #     pseu_labels = gen_pseu_labels(model = model_s,
    #                                 dataloader = det_loader_u,
    #                                 device = device,
    #                                 detection_postprocess = detection_postprocess,
    #                                 prob_threshold = 0.4,
    #                                 mixed_precision = args.val_mixed_precision,
    #                                 memory_format = args.memory_format)
        
    #     pseu_label_save_path = os.path.join(exp_folder, 'pseu_labels.pkl')
    #     with open(pseu_label_save_path, 'wb') as f:
    #         pickle.dump(pseu_labels, f)
    # else:
    #     logger.info('Load pseudo labels from "./pseu_labels.pkl"')
    #     with open('./pseu_labels.pkl', 'rb') as f:
    #         pseu_labels = pickle.load(f)
            
    # original_num_unlabeled = len(train_loader_u.dataset)
    # train_loader_u.dataset.set_pseu_labels(pseu_labels)
    # new_num_unlabeled = len(train_loader_u.dataset)
    # logger.info('After setting pseudo labels, the number of unlabeled samples is changed from {} to {}'.format(original_num_unlabeled, new_num_unlabeled))
    
    for epoch in range(start_epoch, args.epochs + 1):
        # # Update teacher model
        # if epoch % 10 == 0:
        #     for param_t, param_s in zip(model_t.parameters(), model_s.parameters()):
        #         param_t.data = param_s.data.clone().detach()
            
        train_metrics = train(args = args,
                                model = model,
                                detection_loss = detection_loss,
                                unsupervised_detection_loss =  unsupervised_detection_loss,
                                optimizer = optimizer,
                                dataloader_u = train_loader_u,
                                dataloader_l = train_loader_l,
                                detection_postprocess = detection_postprocess,
                                num_iters = len(train_loader_u),
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
            if ((i % args.save_model_interval != 0 or i == 0) and os.path.exists(ckpt_path)):
                os.remove(ckpt_path)
            #TODO
            # if ((i % args.save_model_interval != 0 or i == 0 or i < args.start_val_epoch) and os.path.exists(ckpt_path)):
            #     os.remove(ckpt_path)
        save_states(os.path.join(model_save_dir, f'epoch_{epoch}.pth'), model, optimizer, scheduler_warm, ema)
        
        if (epoch >= args.start_val_epoch and epoch % args.val_interval == 0) or epoch == args.epochs:
            # Use Shadow model to validate and save model
            if ema is not None:
                ema.apply_shadow()
            
            val_metrics = val(args = args,
                            model = model,
                            detection_postprocess=detection_postprocess,
                            val_loader = val_loader, 
                            device = device,
                            image_spacing = IMAGE_SPACING,
                            series_list_path=args.val_set,
                            exp_folder=exp_folder,
                            epoch = epoch,
                            nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                            min_d=args.min_d,
                            min_size=args.min_size,
                            nodule_size_mode=args.nodule_size_mode)
            
            early_stopping.step(val_metrics, epoch)
            write_metrics(val_metrics, epoch, 'val', writer)

            # Restore model
            if ema is not None:
                ema.restore()
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
                            nodule_type_diameters=NODULE_TYPE_DIAMETERS,
                            exp_folder=exp_folder,
                            epoch = 'test_best_{}'.format(target_metric),
                            min_d=args.min_d,
                            min_size=args.min_size,
                            nodule_size_mode=args.nodule_size_mode)
        write_metrics(test_metrics, epoch, 'test/best_{}'.format(target_metric), writer)
        with open(os.path.join(test_save_dir, 'test_best_{}.txt'.format(target_metric)), 'w') as f:
            f.write('Best epoch: {}\n'.format(best_epoch))
            f.write('-' * 30 + '\n')
            max_length = max([len(key) for key in test_metrics.keys()])
            for key, value in test_metrics.items():
                f.write('{}: {:.4f}\n'.format(key.ljust(max_length), value))
    writer.close()