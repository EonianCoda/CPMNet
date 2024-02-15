import math

def get_nodule_type(nodule_size: float) -> str:
    nodule_size_rangs = {'benign': [0, 33.5],
                        'probably_benign': [33.5, 113.1], 
                        'probably_suspicious': [113.1, 268.1],
                        'suspicious': [268.1, -1]}
    for key in nodule_size_rangs:
        if nodule_size >= nodule_size_rangs[key][0] and (nodule_size < nodule_size_rangs[key][1] or nodule_size_rangs[key][1] == -1):
            return key

class NoduleFinding(object):
    '''
    Represents a nodule
    '''
    def __init__(self, coord_x=None, coord_y=None, coord_z=None, 
            pred_prob=None, w=None, h=None, d=None, iou=None):

        # set the variables and convert them to the correct type
        
        self.coord_x = float(coord_x)
        self.coord_y = float(coord_y)
        self.coord_z = float(coord_z)
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.bbox = [self.coord_x - self.w/2, self.coord_x + self.w/2, 
                     self.coord_y - self.h/2, self.coord_y + self.h/2,
                     self.coord_z - self.d/2, self.coord_z + self.d/2]
        
        self.pred_prob = float(pred_prob)
        self.iou = iou
        
        self.area = None
        self.area_method = 0
        self.update_area()
        
    def update_area(self):
        if self.w is None or self.h is None or self.d is None:
            return
        
        if self.area_method == 0:
            self.area = self.w * self.h * self.d
        elif self.area_method == 1:
            r = max(self.w, self.h, self.d) / 2
            self.area = 4/3 * math.pi * r**3
        elif self.area_method == 2:
            r = (self.w + self.h + self.d) / 6.0
            self.area = 4/3 * math.pi * r**3
            
        self.nodule_type = get_nodule_type(self.area)