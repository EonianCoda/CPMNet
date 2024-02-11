import math

def get_nodule_type(nodule_size: float) -> str:
    nodule_size_rangs = {'benign': [0, 33.5],
                        'probably_benign': [33.5, 113.1], 
                        'probably_suspicious': [113.1, 268.1],
                        'suspicious': [268.1, -1]}
    for key in nodule_size_rangs:
        if nodule_size >= nodule_size_rangs[key][0] and (nodule_size < nodule_size_rangs[key][1] or nodule_size_rangs[key][1] == -1):
            return key
    return 'benign'

class NoduleFinding(object):
    '''
    Represents a nodule
    '''
    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
            CADprobability=None, noduleType=None, w=None, h=None, d=None, seriesInstanceUID=None):

        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        # self.diameter_mm = diameter
        self.w = w
        self.h = h
        self.d = d
        self.candidateID = None
        self.seriesuid = seriesInstanceUID

        self.area = None
        self.area_method = 0
        
    def update_area(self):
        if self.area_method == 0:
            self.area = self.w * self.h * self.d
        elif self.area_method == 1:
            r = max(self.w, self.h, self.d) / 2
            self.area = 4/3 * math.pi * r**3
        elif self.area_method == 2:
            r = (self.w + self.h + self.d) / 6.0
            self.area = 4/3 * math.pi * r**3
            
        self.nodule_type = get_nodule_type(self.area)