import numpy as np
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
    