import math
from typing import List
import numpy as np

IMAGE_SPACING = [1, 0.8, 0.8]
class NoduleTyperRect:
    def __init__(self):
        self.diamters = {'benign': [0,4], 
                        'probably_benign': [4, 6],
                        'probably_suspicious': [6, 8],
                        'suspicious': [8, -1]}
        
        self.spacing = np.array(IMAGE_SPACING)
        self.voxel_volume = np.prod(self.spacing)
        
        self.areas = {}
        for key in self.diamters:
            self.areas[key] = [round(self.compute_sphere_volume(self.diamters[key][0]) / self.voxel_volume),
                               round(self.compute_sphere_volume(self.diamters[key][1]) / self.voxel_volume)]
        
    @staticmethod
    def compute_sphere_volume(diameter: float) -> float:
        if diameter == 0:
            return 0
        elif diameter == -1:
            return 100000000
        else:
            radius = diameter / 2
            return 4/3 * math.pi * radius**3
        
    def get_nodule_type(self, nodule_size: float) -> str:
        for key in self.areas:
            if nodule_size >= self.areas[key][0] and (nodule_size < self.areas[key][1] or self.areas[key][1] == -1):
                return key
        return 'benign'

class NoduleFinding(object):
    '''
    Represents a nodule
    '''
    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
            CADprobability=None, nodule_type=None, w=None, h=None, d=None, seriesInstanceUID=None):

        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        # self.diameter_mm = diameter
        self.w = w
        self.h = h
        self.d = d
        self.candidateID = None
        self.seriesuid = seriesInstanceUID
        self.nodule_type = nodule_type
        
    def auto_nodule_type(self):
        nodule_typer = NoduleTyperRect()
        self.volume = 4/3 * math.pi * ((self.w * self.h * self.d) / 6)  # volume of an ellipsoid
        self.nodule_type = nodule_typer.get_nodule_type(self.volume)