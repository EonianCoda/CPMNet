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