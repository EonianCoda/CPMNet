from scipy.spatial.distance import pdist, squareform
import numpy as np
from dataload.utils import load_label, load_series_list, gen_label_path, ALL_RAD
import cc3d
import os
from multiprocessing import Pool
import pickle
import json
import shutil

ME_FOLDER = r'D:\workspace\medical_dataset\ME_dataset'
LDCT_TEST_FOLDER = r'D:\workspace\medical_dataset\LDCT_test_dataset'

def load_gt_mask_maps(mask_maps_path: str):
    gt_mask_maps = np.load(mask_maps_path)
    # npz
    if mask_maps_path.endswith('.npz'):
        gt_mask_maps = gt_mask_maps['image'] 

    # binarize
    bg_mask = (gt_mask_maps <= 125)
    gt_mask_maps[bg_mask] = 0
    gt_mask_maps[~bg_mask] = 1
    return gt_mask_maps.astype(np.uint8, copy=False)

def get_cc3d(mask_path: str):
    binary_mask_maps = load_gt_mask_maps(mask_path)
    labels = cc3d.connected_components(binary_mask_maps, out_dtype=np.uint32)
    stats = cc3d.statistics(labels)
    
    valid_component_indices = [] # The ID of component whose number of element is larger than nodule_3d_minimum_size.
    valid_nodule_sizes = []
    bboxes = []
    
    # Get the two diagonal vertex's coordinate(xMin, yMin, zMin) and (xMax, yMax, zMax) of a cube 
    for component_id, (counts, box) in enumerate(zip(stats['voxel_counts'], stats['bounding_boxes'])):
        # If componentID == 0, this compoent is background.
        # If this component is too small, then ignore it.
        if component_id == 0:
            continue

        valid_component_indices.append(component_id)
        valid_nodule_sizes.append(counts)
        # y is top-to-down
        # x is left-to-right
        y_range, x_range, z_range = box 
        
        y_min, y_max = y_range.start, y_range.stop
        x_min, x_max = x_range.start, x_range.stop
        z_min, z_max = z_range.start, z_range.stop
        coord = [[y_min, x_min, z_min], [y_max, x_max, z_max]]
        bboxes.append(coord)
        
    return labels, valid_component_indices, valid_nodule_sizes, bboxes

def get_label(mask_path: str):
    series_name = os.path.dirname(mask_path).split('\\')[-2]
    
    cc3d_labels, valid_component_indices, valid_nodule_sizes, bboxes = get_cc3d(mask_path)
    diameters = []
    for component_i in valid_component_indices:
        nodule_mask = (cc3d_labels == component_i)
        nonzero = np.count_nonzero(nodule_mask, axis=(0, 1))
        max_xy_z = np.argmax(nonzero)
        nodule_mask = nodule_mask[:, :, max_xy_z]
        
        non_zero_coords = np.transpose(np.nonzero(nodule_mask))
        pairwise_distances = squareform(pdist(non_zero_coords))
        max_distance_index = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
        farthest_point_1 = non_zero_coords[max_distance_index[0]]
        farthest_point_2 = non_zero_coords[max_distance_index[1]]
        diameter = np.linalg.norm(farthest_point_1 - farthest_point_2) * 0.8
        
        diameters.append(diameter)
    
    label = {}
    label['series_name'] = series_name
    label['nodule_size'] = valid_nodule_sizes
    label['bboxes'] = bboxes
    label['diameters'] = diameters
    
    bboxes = np.array(bboxes, dtype=np.int32)
    if len(bboxes) == 0:
        label['nodule_start_slice_ids'] = []
    else:
        nodule_start_slice_ids = bboxes[:, 0, 2].tolist()
        label['nodule_start_slice_ids'] = nodule_start_slice_ids
    
    return label

if __name__ == '__main__':
    new_save_folder = './data/new_diamters_labels'
    series_infos = load_series_list('./data/all_old.txt')
    
    mask_paths = []
    label_paths = []
    for info in series_infos:
        mask_paths.append(os.path.join(info[0], 'mask', '{}_crop.npz'.format(info[1])))
        label_paths.append(gen_label_path(info[0], info[1]))
    
    with Pool(os.cpu_count() // 3) as p:
        labels = p.map(get_label, mask_paths)
    
    new_labels = dict()
    for label in labels:
        series_name = label['series_name']
        new_labels[series_name] = label
    
    os.makedirs(new_save_folder, exist_ok=True)
    for mask_path, label_path, (series_folder, series_name) in zip(mask_paths, label_paths, series_infos):
        last_modified_time = os.path.getmtime(mask_path)
        
        with open(label_path, 'r') as f:
            label = json.load(f)     
        label['diameters'] = new_labels[series_name]['diameters']
        
        save_name = os.path.basename(label_path)
        save_path = os.path.join(new_save_folder, save_name)
        with open(save_path, 'w') as f:
            json.dump(label, f)
            
        src_path = save_path
        dst_path = os.path.join(series_folder, 'mask', f'{series_name}_nodule_count_crop_diameters.json')
        shutil.copy(src_path, dst_path)