from typing import List, Tuple, Dict

import numpy as np

from dataload.utils import ALL_LOC, ALL_RAD
from evaluationScript.nodule_finding_original import NoduleFinding

from typing import List, Tuple

def output2nodulefinding(output: np.ndarray) -> List[NoduleFinding]:
    pred_nodules = []
    for n in output:
        prob, z, y, x, d, h, w = n
        nodule_finding = NoduleFinding(coordX=x, coordY=y, coordZ=z, w=w, h=h, d=d, CADprobability=prob)
        nodule_finding.auto_nodule_type()
        pred_nodules.append(nodule_finding)
    return pred_nodules

def label2nodulefinding(label: Dict[str, np.ndarray]) -> List[NoduleFinding]:
    """
    Args:
        label: a dictionary with keys 'all_loc' and 'all_rad'
    """
    nodules = []
    loc = label[ALL_LOC]
    rad = label[ALL_RAD]
    for (z, y, x), (d, h, w), r in zip(loc, rad, rad):
        nodule_finding = NoduleFinding(coordX=x, coordY=y, coordZ=z, w=w, h=h, d=d, CADprobability=1.0)
        nodule_finding.auto_nodule_type()
        nodules.append(nodule_finding)
    return nodules

def nodule2cude(nodules: List[NoduleFinding], shape: Tuple[int, int, int]) -> np.ndarray:
    if not isinstance(nodules, list):
        nodules = [nodules]
    
    bboxes = []
    for nodule in nodules:
        z, y, x, d, h, w = nodule.coordZ, nodule.coordY, nodule.coordX, nodule.d, nodule.h, nodule.w
        z1 = max(round(z - d/2), 0)
        y1 = max(round(y - h/2), 0)
        x1 = max(round(x - w/2), 0)

        z2 = min(round(z + d/2), shape[0])
        y2 = min(round(y + h/2), shape[1])
        x2 = min(round(x + w/2), shape[2])
        bboxes.append((z1, y1, x1, z2, y2, x2))
    bboxes = np.array(bboxes)
    return bboxes