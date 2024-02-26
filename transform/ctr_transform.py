import numpy as np
from typing import Tuple

class AbstractCTRTransform(object):
    def __init__(self, params):
        pass
    def __call__(self, ctr):
        return ctr

class EmptyTransform(AbstractCTRTransform):
    def __init__(self):
        pass
    def __call__(self, ctr):
        return ctr    

class OffsetMinusCTR(AbstractCTRTransform):
    def __init__(self, offset: np.ndarray):
        if not isinstance(offset, np.ndarray):
            offset = np.array([offset])
        self.offset = offset
        
    def forward(self, ctr):
        """new = offset - old
        """
        return self.offset - ctr
    
    def backward(self, ctr):
        """old = offset - new
        """
        return self.offset - ctr

class OffsetPlusCTR(AbstractCTRTransform):
    def __init__(self, offset: np.ndarray):
        if not isinstance(offset, np.ndarray):
            offset = np.array([offset])
        self.offset = offset
        
    def forward(self, ctr):
        """new = offset + old
        """
        return self.offset + ctr
    
    def backward(self, ctr):
        """old = new - offset
        """
        return ctr - self.offset    

class TransposeCTR(AbstractCTRTransform):
    def __init__(self, transpose: np.ndarray):
        if not isinstance(transpose, np.ndarray):
            transpose = np.array([transpose], dtype=np.int32)
        self.transpose = transpose
        
    def forward(self, ctr):
        return ctr[:, self.transpose]
    
    def backward(self, ctr):
        return ctr[:, self.transpose]

class RotateCTR(AbstractCTRTransform):
    def __init__(self, angle: float, axes: Tuple[int, int], image_shape: np.ndarray):
        self.radian = np.deg2rad(angle)
        self.image_center = np.array(image_shape) / 2
        self.cos = np.cos(self.radian)
        self.sin = np.sin(self.radian)
        self.axes = axes
        
    def forward(self, ctr):
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos - (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr
    
    def backward(self, ctr):
        new_ctr = ctr.copy()
        new_ctr[:, self.axes[0]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * self.cos + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.sin + self.image_center[self.axes[0]]
        new_ctr[:, self.axes[1]] = (ctr[:, self.axes[0]] - self.image_center[self.axes[0]]) * -self.sin + (ctr[:, self.axes[1]] - self.image_center[self.axes[1]]) * self.cos + self.image_center[self.axes[1]]
        return new_ctr