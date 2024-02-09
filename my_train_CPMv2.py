# %% -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import math
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
###optimzer###
from optimizer.optim import AdamW
from optimizer.scheduler import GradualWarmupScheduler
###postprocessing###
from utils.box_utils import nms_3D
from evaluationScript.detectionCADEvalutionIOU import nodule_evaluation

from utils.logs import setup_logging
from utils.average_meter import AverageMeter
from utils.utils import init_seed, get_local_time_in_taiwan, get_progress_bar, write_yaml, load_yaml
from utils.generate_annot_csv_from_series_list import generate_annot_csv

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
    args = parser.parse_args()
    return args

def write_metrics(metrics: dict, 
                epoch: int,
                prefix: str,
                writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

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

def training_data_prepare(args, blank_side=0):
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

def test_val_data_prepare(args):
    crop_size = args.crop_size
    overlap_size = [int(crop_size[i] * OVERLAY_RATIO) for i in range(len(crop_size))]
    
    split_comber = SplitComb(crop_size=crop_size, overlap_size=overlap_size, pad_value=-1)
    test_dataset = DetDatasetCSVRTest(series_list_path = args.val_set, SplitComb=split_comber, image_spacing=IMAGE_SPACING)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False,)
    logger.info("Number of test samples: {}".format(len(test_loader.dataset)))
    logger.info("Number of test batches: {}".format(len(test_loader)))
    return test_loader

def train_one_step(args, model, sample, device):
    image = sample['image'].to(device, non_blocking=True) # z, y, x
    labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
    
    # Compute loss
    cls_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
    cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
    loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
    return loss, (cls_loss, shape_loss, offset_loss, iou_loss)

def train(args,
          model,
          optimizer,
          train_loader,
          scheduler_warm,
          device,
          writer: SummaryWriter,
          epoch: int,
          exp_folder: str):
    model.train()
    scheduler_warm.step()
    
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
    progress_bar = get_progress_bar('Train', len(train_loader))
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, (cls_loss, shape_loss, offset_loss, iou_loss) = train_one_step(args, model, sample, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, (cls_loss, shape_loss, offset_loss, iou_loss) = train_one_step(args, model, sample, device)
            loss.backward()
            optimizer.step()
        
        # Update history
        avg_cls_loss.update(cls_loss.item() * args.lambda_cls)
        avg_shape_loss.update(shape_loss.item() * args.lambda_shape)
        avg_offset_loss.update(offset_loss.item() * args.lambda_offset)
        avg_iou_loss.update(iou_loss.item() * args.lambda_iou)
        avg_loss.update(loss.item())
        
        if progress_bar is not None:
            progress_bar.set_postfix(loss = avg_loss.avg,
                                    cls_Loss = avg_cls_loss.avg,
                                    shape_loss = avg_shape_loss.avg,
                                    offset_loss = avg_offset_loss.avg,
                                    giou_loss = avg_iou_loss.avg)
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()

    logger.info('====> Epoch: {} train_cls_loss: {:.4f}'.format(epoch, avg_cls_loss.avg))
    logger.info('====> Epoch: {} train_shape_loss: {:.4f}'.format(epoch, avg_shape_loss.avg))
    logger.info('====> Epoch: {} train_offset_loss: {:.4f}'.format(epoch, avg_offset_loss.avg))
    logger.info('====> Epoch: {} train_iou_loss: {:.4f}'.format(epoch, avg_iou_loss.avg))
    
    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    
    write_metrics(metrics, epoch, 'train', writer)
    
    # Remove the checkpoint of epoch % save_model_interval != 0
    model_save_folder = os.path.join(exp_folder, 'model')
    os.makedirs(model_save_folder, exist_ok=True)
    for i in range(epoch):
        ckpt_path = os.path.join(model_save_folder, 'epoch_{}.pth'.format(i))
        if (i % args.save_model_interval != 0 or i == 0) and os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    
    # Save checkpoint    
    ckpt_path = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler_warm.state_dict()}
    torch.save(ckpt_path, os.path.join(model_save_folder, 'epoch_{}.pth'.format(epoch)))    

def val(model: nn.Module,
        test_loader: DataLoader,
        epoch: int,
        exp_folder: str,
        save_dir: str,
        annot_path: str, 
        seriesuids_path: str,
        writer: SummaryWriter = None,
        nms_keep_top_k: int = 40):
    def convert_to_standard_output(output: np.ndarray, spacing: torch.Tensor, name: str) -> List[List[Any]]:
        '''
        convert [id, prob, ctr_z, ctr_y, ctr_x, d, h, w] to
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
        '''
        preds = []
        spacing = np.array([spacing[0].numpy(), spacing[1].numpy(), spacing[2].numpy()]).reshape(-1, 3)
        for j in range(output.shape[0]):
            preds.append([name, output[j, 4], output[j, 3], output[j, 2], output[j, 1], output[j, 7], output[j, 6], output[j, 5]])
        return preds
    
    model.eval()
    split_comber = test_loader.dataset.splitcomb
    batch_size = args.batch_size * args.num_samples
    all_preds = []
    for sample in test_loader:
        data = sample['split_images'][0].to(device, non_blocking=True)
        nzhw = sample['nzhw']
        name = sample['file_name'][0]
        spacing = sample['spacing'][0]
        outputlist = []
        
        for i in range(int(math.ceil(data.size(0) / batch_size))):
            end = (i + 1) * batch_size
            if end > data.size(0):
                end = data.size(0)
            input = data[i * batch_size:end]
            with torch.no_grad():
                output = model(input)
                output = detection_postprocess(output, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputlist.append(output.data.cpu().numpy())
            
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        output = torch.from_numpy(output).view(-1, 8)
        
        # Remove the padding
        object_ids = output[:, 0] != -1.0
        output = output[object_ids]
        
        # NMS
        if len(output) > 0:
            keep = nms_3D(output[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
            output = output[keep]
        output = output.numpy()
        
        preds = convert_to_standard_output(output, spacing, name) # convert to ['seriesuid', 'coordX', 'coordY', 'coordZ', 'radius', 'probability']
        all_preds.extend(preds)
        
    # Save the results to csv
    header = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability', 'w', 'h', 'd']
    df = pd.DataFrame(all_preds, columns=header)
    pred_results_path = os.path.join(save_dir, 'predict_epoch_{}.csv'.format(epoch))
    df.to_csv(pred_results_path, index=False)
    
    outputDir = os.path.join(save_dir, pred_results_path.split('/')[-1].split('.')[0])
    os.makedirs(outputDir, exist_ok=True)
    FP_ratios = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    out_01, fiexed_out = nodule_evaluation(annot_path = annot_path,
                                            seriesuids_path = seriesuids_path, 
                                            pred_results_path = pred_results_path,
                                            output_dir = outputDir,
                                            iou_threshold = args.val_iou_threshold,
                                            fixed_prob_threshold=args.val_fixed_prob_threshold)
    frocs = out_01[-1]
    logger.info('====> Epoch: {}'.format(epoch))
    for i in range(len(frocs)):
        logger.info('====> fps:{:.4f} iou 0.1 frocs:{:.4f}'.format(FP_ratios[i], frocs[i]))
    logger.info('====> mean frocs:{:.4f}'.format(np.mean(np.array(frocs))))
    
    fixed_tp, fixed_fp, fixed_fn, fixed_recall, fixed_precision, fixed_f1_score = fiexed_out
    metrics = {'tp': fixed_tp,
                'fp': fixed_fp,
                'fn': fixed_fn,
                'recall': fixed_recall,
                'precision': fixed_precision,
                'f1_score': fixed_f1_score}
    global best_metric
    global best_epoch
    if metrics[args.metric] >= best_metric:
        best_metric = metrics[args.metric]
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(exp_folder, 'best_model.pth'))
        logger.info('====> Best model saved at epoch: {}'.format(epoch))
        logger.info('====> Best metric {}: {:.4f}'.format(args.metric, best_metric))
        with open(os.path.join(exp_folder, 'best_epoch.txt'), 'w') as f:
            f.write('Epoch: {}\n'.format(best_epoch))
            f.write('Metric: {}\n'.format(args.metric))
            f.write('Value: {}\n'.format(best_metric))
            f.write('-'*20)
            for key, value in metrics.items():
                f.write('{}: {:.4f}\n'.format(key, value))
    if writer is not None:
        write_metrics(metrics, epoch, 'val', writer)

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

    init_seed(args.seed)
    
    train_loader = training_data_prepare(args)
    val_loader = test_val_data_prepare(args)
    
    annot_dir = os.path.join(exp_folder, 'annotation')
    state = 'validate'
    origin_annot_path = os.path.join(annot_dir, 'origin_annotation_{}.csv'.format(state))
    annot_path = os.path.join(annot_dir, 'annotation_{}.csv'.format(state))
    generate_annot_csv(args.val_set, origin_annot_path, spacing=IMAGE_SPACING)
    convert_to_standard_csv(csv_path = origin_annot_path, 
                            save_dir = annot_dir, 
                            state = state, 
                            spacing = IMAGE_SPACING)
    for epoch in range(start_epoch, args.epochs + 1):
        train(args = args,
              model = model,
              optimizer = optimizer,
              scheduler_warm = scheduler_warm,
              train_loader = train_loader, 
              device = device,
              writer = writer,
              epoch = epoch, 
              exp_folder = exp_folder)
        if epoch >= args.start_val_epoch: 
            val(epoch = epoch,
                test_loader = val_loader, 
                save_dir = annot_dir,
                exp_folder=exp_folder,
                writer = writer,
                annot_path = os.path.join(annot_dir, 'annotation_validate.csv'), 
                seriesuids_path = os.path.join(annot_dir, 'seriesuid_validate.csv'), 
                model = model)