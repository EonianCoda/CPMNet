from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch

def train_collate_fn(batches) -> Dict[str, torch.Tensor]:
    batch = []
    for b in batches:
        batch.extend(b)
        
    imgs = []
    annots = []
    for b in batch:
        imgs.append(b['image'])
        annots.append(b['annot'])
    imgs = np.stack(imgs)
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 10), dtype='float32') * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 10), dtype='float32') * -1

    return {'image': torch.from_numpy(imgs), 'annot': torch.from_numpy(annot_padded)}

def infer_collate_fn(batches) -> Dict[str, torch.Tensor]:
    num_splits = []
    imgs = []
    nzhws = []
    spacings = []
    series_names = []
    series_folders = []

    for b in batches:
        imgs.append(b['split_images'])
        num_splits.append(b['split_images'].shape[0])
        nzhws.append(b['nzhw'])
        spacings.append(b['spacing'])
        series_names.append(b['series_name'])
        series_folders.append(b['series_folder'])
        
    imgs = np.concatenate(imgs, axis=0)
    nzhws = np.stack(nzhws)
    num_splits = np.array(num_splits)
    
    return {'split_images': torch.from_numpy(imgs),
            'nzhws': torch.from_numpy(nzhws), 
            'num_splits': num_splits, 
            'spacings': spacings, 
            'series_names': series_names,
            'series_folders': series_folders}

def infer_refined_collate_fn(batches) -> Dict[str, torch.Tensor]:
    num_splits = []
    crop_images = []
    nodule_centers = []
    nodule_shapes = []
    crop_bb_mins = []
    series_names = []
    series_paths = []

    for b in batches:
        crop_images.append(b['crop_images'])
        num_splits.append(b['crop_images'].shape[0])
        nodule_centers.append(b['nodule_centers'])
        nodule_shapes.append(b['nodule_shapes'])
        crop_bb_mins.append(b['crop_bb_mins'])
        series_names.append(b['series_name'])
        series_paths.append(b['series_path'])
        
    crop_images = np.concatenate(crop_images, axis=0)
    
    return {'crop_images': torch.from_numpy(crop_images),
            'num_splits': np.array(num_splits),
            'nodule_centers': nodule_centers,
            'nodule_shapes': nodule_shapes,
            'crop_bb_mins': crop_bb_mins,
            'series_names': series_names,
            'series_paths': series_paths}

def semi_unlabeled_train_collate_fn_dict(batches):
    batch = []
    for b in batches:
        if b is not None:
            batch.extend(b)
    
    if len(batch) == 0:
        return {'image': torch.tensor([]), 'annot': torch.tensor([]), 'ctr_transform': []}
    
    imgs = []
    annots = []
    for b in batch:
        imgs.append(b['image'])
        annots.append(b['annot'])
    
    imgs = np.stack(imgs)
    max_num_annots = max(annot.shape[0] for annot in annots)

    ctr_transforms = [s['ctr_transform'] for s in batch]
    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 10), dtype='float32') * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 10), dtype='float32') * -1
    
    return {'image': torch.from_numpy(imgs), 'annot': torch.from_numpy(annot_padded), 'ctr_transform': ctr_transforms}

def semi_unlabeled_infer_collate_fn(batches) -> Dict[str, torch.Tensor]:
    num_splits = []
    imgs = []
    nzhws = []
    spacings = []
    series_names = []
    series_folders = []

    for b in batches:
        imgs.append(b['split_images'])
        num_splits.append(b['split_images'].shape[0])
        nzhws.append(b['nzhw'])
        spacings.append(b['spacing'])
        series_names.append(b['series_name'])
        series_folders.append(b['series_folder'])
        
    imgs = np.concatenate(imgs, axis=0)
    nzhws = np.stack(nzhws)
    num_splits = np.array(num_splits)
    
    return {'split_images': torch.from_numpy(imgs),
            'nzhws': torch.from_numpy(nzhws), 
            'num_splits': num_splits, 
            'spacings': spacings, 
            'series_names': series_names,
            'series_folders': series_folders}

def semi_labeled_collate_fn_dict(batches) -> Dict[str, torch.Tensor]:
    batch = []
    for b in batches:
        batch.extend(b)
    
    imgs = []
    annots = []
    for b in batch:
        imgs.append(b['image'])
        annots.append(b['annot'])
        
    imgs = np.stack(imgs)
    max_num_annots = max(annot.shape[0] for annot in annots)

    ctr_transforms = [s['ctr_transform'] for s in batch]
    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 10), dtype='float32') * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 10), dtype='float32') * -1

    return {'image': torch.from_numpy(imgs), 'annot': torch.from_numpy(annot_padded), 'ctr_transform': ctr_transforms}
