# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random

def compute_bbox3d_intersection_volume(box1: np.ndarray, box2: np.ndarray):
    """ 
    Args:
        box1 (shape = [N, 2, 3])
        box2 (shape = [M, 2, 3])
    Return:
        the area of the intersection between box1 and box2, shape = [N, M]
    """
    a1, a2 = box1[:,np.newaxis, 0,:], box1[:,np.newaxis, 1,:] # [N, 1, 3]
    b1, b2 = box2[np.newaxis,:, 0,:], box2[np.newaxis,:, 1,:] # [1, N, 3]
    inter_volume = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod(axis=2)

    return inter_volume

class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W]
    """

    def __init__(self, crop_size, rand_trans=None, instance_crop=True, overlap_size=[16, 32, 32], 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou = 0.5):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            instance_crop: additional sampling with instance around center
            overlap_size: the size of overlap  of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.
            sample_cls: the class of the sample
        """
        self.sample_cls = sample_cls
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.overlap_size = np.array(overlap_size, dtype=np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop
        self.tp_iou = tp_iou

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

    def get_crop_centers(self, shape, dim: int):
        crop = self.crop_size[dim]
        overlap = self.overlap_size[dim]
        stride = self.stride_size[dim]
        shape = shape[dim]
        
        crop_centers = np.arange(0, shape - overlap, stride) + crop / 2
        crop_centers = np.clip(crop_centers, a_max=shape - crop / 2, a_min=None)
        
        # Add final center
        crop_centers = np.append(crop_centers, shape - crop / 2)
        crop_centers = np.unique(crop_centers)
        
        return crop_centers
    
    def __call__(self, sample, image_spacing: np.ndarray):
        image = sample['image']
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        
        all_rad_pixel = all_rad / image_spacing
        all_nodule_bb_min = all_loc - all_rad_pixel / 2
        all_nodule_bb_max = all_loc + all_rad_pixel / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]

        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        # Generate crop centers
        crop_centers = []
        for z in z_crop_centers:
            for y in y_crop_centers:
                for x in x_crop_centers:
                    crop_centers.append([z, y, x])
        crop_centers = np.array(crop_centers)
        
        if self.instance_crop and len(instance_loc) > 0:
            if self.rand_trans is not None:
                instance_crop = instance_loc + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(instance_loc), 3))
            else:
                instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)

        if self.rand_trans is not None:
            crop_centers = crop_centers + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(crop_centers), 3))
        
        all_crop_bb_min = crop_centers - crop_size / 2
        all_crop_bb_min = np.clip(all_crop_bb_min, a_min=0, a_max=shape - crop_size)
        all_crop_bb_min = np.unique(all_crop_bb_min, axis=0)
        
        all_crop_bb_max = all_crop_bb_min + crop_size
        all_crop_bboxes = np.stack([all_crop_bb_min, all_crop_bb_max], axis=1) # [M, 2, 3]
        
        
        # Compute IoU to determine the label of the patches
        inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, nodule_bboxes) # [M, N]
        all_ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
        max_ious = np.max(all_ious, axis=1) # [M]
        
        tp_indices = max_ious > self.tp_iou
        neg_indices = ~tp_indices

        # Sample patches
        tp_prob = self.tp_ratio / tp_indices.sum() if tp_indices.sum() > 0 else 0
        probs = np.zeros(shape=len(max_ious))
        probs[tp_indices] = tp_prob
        probs[neg_indices] = (1. - probs.sum()) / neg_indices.sum() if neg_indices.sum() > 0 else 0
        probs = probs / probs.sum() # normalize
        sample_indices = np.random.choice(np.arange(len(all_crop_bboxes)), size=self.sample_num, p=probs, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                               crop_bb_min[1]: crop_bb_max[1], 
                               crop_bb_min[2]: crop_bb_max[2]]
            image_crop = np.expand_dims(image_crop, axis=0)
            
            ious = all_ious[sample_i] # [N]
            in_idx = np.where(ious > self.tp_iou)[0]
            if in_idx.size > 0:
                # Compute new ctr and rad because of the crop
                all_nodule_bb_min_crop = all_nodule_bb_min - crop_bb_min
                all_nodule_bb_max_crop = all_nodule_bb_max - crop_bb_min
                
                nodule_bb_min_crop = all_nodule_bb_min_crop[in_idx]
                nodule_bb_max_crop = all_nodule_bb_max_crop[in_idx]
                
                nodule_bb_min_crop = np.clip(nodule_bb_min_crop, a_min=0, a_max=None)
                nodule_bb_max_crop = np.clip(nodule_bb_max_crop, a_min=None, a_max=crop_size)
                
                ctr = (nodule_bb_min_crop + nodule_bb_max_crop) / 2
                rad = nodule_bb_max_crop - nodule_bb_min_crop
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])

            sample = dict()
            sample['image'] = image_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            samples.append(sample)
        return samples

def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec

def apply_transformation_coord(coord, transform_param_list, rot_center):
    """
    apply rotation transformation to an ND image
    Args:
        image (nd array): the input nd image
        transform_param_list (list): a list of roration angle and axes
        order (int): interpolation order
    """
    for angle, axes in transform_param_list:
        # rot_center = np.random.uniform(low=np.min(coord, axis=0), high=np.max(coord, axis=0), size=3)
        org = coord - rot_center
        new = rotate_vecs_3d(org, angle, axes)
        coord = new + rot_center

    return coord


def rand_rot_coord(coord, angle_range_d, angle_range_h, angle_range_w, rot_center, p):
    transform_param_list = []

    if (angle_range_d[1]-angle_range_d[0] > 0) and (random.random() < p):
        angle_d = np.random.uniform(angle_range_d[0], angle_range_d[1])
        transform_param_list.append([angle_d, (-2, -1)])
    if (angle_range_h[1]-angle_range_h[0] > 0) and (random.random() < p):
        angle_h = np.random.uniform(angle_range_h[0], angle_range_h[1])
        transform_param_list.append([angle_h, (-3, -1)])
    if (angle_range_w[1]-angle_range_w[0] > 0) and (random.random() < p):
        angle_w = np.random.uniform(angle_range_w[0], angle_range_w[1])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord


def convert_to_one_hot(label, class_num):
    label_prob = []
    for i in range(class_num):
        temp_prob = label == i * np.ones_like(label)
        label_prob.append(temp_prob)
    label_prob = np.asarray(label_prob, dtype='float32')
    return label_prob


def reorient(itk_img, mark_matrix, spacing=[1., 1., 1.], interp1=sitk.sitkLinear):
    '''
    itk_img: image to reorient
    mark_matric: physical mark point
    '''
    spacing = spacing[::-1]
    origin, x_mark, y_mark, z_mark = np.array(mark_matrix[0]), np.array(mark_matrix[1]), np.array(
        mark_matrix[2]), np.array(mark_matrix[3])

    # centroid_world = itk_img.TransformContinuousIndexToPhysicalPoint(centroid)
    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetInterpolator(interp1)
    filter_resample.SetOutputSpacing(spacing)

    # set origin
    origin_reorient = mark_matrix[0]
    # set direction
    # !!! note: column wise
    x_base = (x_mark - origin) / np.linalg.norm(x_mark - origin)
    y_base = (y_mark - origin) / np.linalg.norm(y_mark - origin)
    z_base = (z_mark - origin) / np.linalg.norm(z_mark - origin)
    direction_reorient = np.stack([x_base, y_base, z_base]).transpose().reshape(-1).tolist()

    # set size
    x, y, z = np.linalg.norm(x_mark - origin) / spacing[0], np.linalg.norm(y_mark - origin) / spacing[
        1], np.linalg.norm(z_mark - origin) / spacing[2]
    size_reorient = (int(np.ceil(x + 0.5)), int(np.ceil(y + 0.5)), int(np.ceil(z + 0.5)))

    filter_resample.SetOutputOrigin(origin_reorient)
    filter_resample.SetOutputDirection(direction_reorient)
    filter_resample.SetSize(size_reorient)
    # filter_resample.SetSpacing([sp]*3)

    filter_resample.SetOutputPixelType(itk_img.GetPixelID())
    itk_out = filter_resample.Execute(itk_img)

    return itk_out


def resample_simg(simg, interp=sitk.sitkBSpline, spacing=[1., 0.7, 0.7]):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    new_spacing = spacing[::-1]

    sp1 = simg.GetSpacing()
    sz1 = simg.GetSize()
    sz2 = (int(round(sz1[0] * sp1[0] / new_spacing[0])), int(round(sz1[1] * sp1[1] / new_spacing[1])),
           int(round(sz1[2] * sp1[2] / new_spacing[2])))

    new_origin = simg.GetOrigin()
    new_origin = (new_origin[0] - sp1[0] / 2 + new_spacing[0] / 2, new_origin[1] - sp1[1] / 2 + new_spacing[1] / 2,
                  new_origin[2] - sp1[2] / 2 + new_spacing[2] / 2)
    imRefImage = sitk.Image(sz2, simg.GetPixelIDValue())
    imRefImage.SetSpacing(new_spacing)
    imRefImage.SetOrigin(new_origin)
    imRefImage.SetDirection(simg.GetDirection())
    resampled_image = sitk.Resample(simg, imRefImage, identity1, interp)
    return resampled_image