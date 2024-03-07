import numpy as np
from .abstract_transform import AbstractTransform

class CoordToAnnot(AbstractTransform):
    """Convert one-channel label map to one-hot multi-channel probability map"""
    def __call__(self, sample):
        ctr = sample['ctr']
        rad = sample['rad']
        cls = sample['cls']
        
        spacing = sample['spacing']
        n = ctr.shape[0]
        spacing = np.tile(spacing, (n, 1))
        
        annot = np.concatenate([ctr, rad.reshape(-1, 3), spacing.reshape(-1, 3), cls.reshape(-1, 1)], axis=-1).astype('float32') # (n, 10)

        sample['annot'] = annot
        return sample
