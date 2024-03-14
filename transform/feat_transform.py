import numpy as np
from typing import Tuple
import torch

class AbstractFeatTransform(object):
    def __init__(self, params):
        pass
    def forward(self, feat):
        return feat
    
    def backward(self, feat):
        return feat

class FlipFeatTransform(AbstractFeatTransform):
    def __init__(self, flip_axes: Tuple[int]):
        self.flip_axes = list(flip_axes)
        for i in self.flip_axes:
            if i > 0:
                raise ValueError("flip_axes should be negative index")
        
    def forward(self, feat):
        if isinstance(feat, np.ndarray):
            return np.flip(feat, self.flip_axes)
        elif isinstance(feat, torch.Tensor):
            return torch.flip(feat, self.flip_axes)
    
    def backward(self, feat):
        return self.forward(feat)

class Rot90FeatTransform(AbstractFeatTransform):
    def __init__(self, rot_angle: int, rot_axes: Tuple[int]):
        self.rot_axes = list(rot_axes)
        self.rot_angle = rot_angle
        for i in self.rot_axes:
            if i > 0:
                raise ValueError("rot_axes should be negative index")
        assert rot_angle % 90 == 0, "rot_angle must be multiple of 90"
        
    def forward(self, feat):
        if isinstance(feat, np.ndarray):
            return np.rot90(feat, self.rot_angle // 90, self.rot_axes)
        elif isinstance(feat, torch.Tensor):
            return torch.rot90(feat, self.rot_angle // 90, self.rot_axes)
    
    def backward(self, feat):
        if isinstance(feat, np.ndarray):
            return np.rot90(feat, -self.rot_angle // 90, self.rot_axes)
        elif isinstance(feat, torch.Tensor):
            return torch.rot90(feat, -self.rot_angle // 90, self.rot_axes)

class TransposeFeatTransform(AbstractFeatTransform):
    def __init__(self, transpose_order: Tuple[int]):
        self.transpose_order = list(transpose_order)
        
    def forward(self, feat):
        assert len(feat) == 4, "feat must be 4D(C, D, H, W)"
        if isinstance(feat, np.ndarray):
            return np.transpose(feat, self.transpose_order)
        elif isinstance(feat, torch.Tensor):
            return feat.permute(self.transpose_order)
        
    def backward(self, feat):
        assert len(feat) == 4, "feat must be 4D(C, D, H, W)"
        if isinstance(feat, np.ndarray):
            return np.transpose(feat, self.transpose_order)
        elif isinstance(feat, torch.Tensor):
            return feat.permute(self.transpose_order)
        
class RotateCTR(AbstractFeatTransform):
    def __init__(self, angle: float, axes: Tuple[int, int], image_shape: np.ndarray):
        self.angle = angle
        self.radian = np.deg2rad(angle)
        self.image_center = np.array(image_shape) / 2
        self.cos = np.cos(self.radian)
        self.sin = np.sin(self.radian)
        self.axes = axes
        
        if self.angle % 90 != 0:
            raise ValueError("angle must be multiple of 90")
        
    def forward(self, ctr):
        if len(ctr) == 0:
            return ctr
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos - (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr
    
    def backward(self, ctr):
        if len(ctr) == 0:
            return ctr
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * -self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr
    
    def forward_rad(self, rads):
        if len(rads) == 0:
            return rads
        if self.angle == 90 or self.angle == 270:
            new_coords = rads.copy()
            new_coords[:, self.axes[0]] = rads[:, self.axes[1]]
            new_coords[:, self.axes[1]] = rads[:, self.axes[0]]
            return new_coords
        else:
            return rads
        
    def backward_rad(self, coords):
        return self.forward_rad(coords)
    
    def forward_spacing(self, spacing):
        if self.angle == 90 or self.angle == 270:
            new_spacing = spacing.copy()
            new_spacing[self.axes[0]] = spacing[self.axes[1]]
            new_spacing[self.axes[1]] = spacing[self.axes[0]]
            return new_spacing
        else:
            return spacing
    
    def backward_spacing(self, spacing):
        return self.forward_spacing(spacing)