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
    if 'state_dict' not in state_dict:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict['state_dict'])
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

def val(model: nn.Module,
        test_loader: DataLoader,
        epoch: int,
        exp_folder: str,
        save_dir: str,
        annot_path: str, 
        seriesuids_path: str,
        writer: SummaryWriter,
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
    
    setup_logging(level='info')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, detection_postprocess = prepare_validation(args)
    init_seed(args.seed)
    
    val_loader = val_data_prepare(args)
    
    exp_folder = os.path.join(SAVE_ROOT, 'val_temp')
    os.makedirs(exp_folder, exist_ok=True)
    annot_dir = os.path.join(exp_folder, 'annotation')
    state = 'validate'
    origin_annot_path = os.path.join(annot_dir, 'origin_annotation_{}.csv'.format(state))
    annot_path = os.path.join(annot_dir, 'annotation_{}.csv'.format(state))
    generate_annot_csv(args.val_set, origin_annot_path, spacing=IMAGE_SPACING)
    convert_to_standard_csv(csv_path = origin_annot_path, 
                            save_dir = annot_dir, 
                            state = state, 
                            spacing = IMAGE_SPACING)
    val(epoch = 0,
        test_loader = val_loader, 
        save_dir = annot_dir,
        exp_folder=exp_folder,
        writer = None,
        annot_path = os.path.join(annot_dir, 'annotation_validate.csv'), 
        seriesuids_path = os.path.join(annot_dir, 'seriesuid_validate.csv'), 
        model = model)