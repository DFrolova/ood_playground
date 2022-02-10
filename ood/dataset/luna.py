import os

import numpy as np

from dpipe.dataset.wrappers import Proxy
from dpipe.io import load
from neurodata.lidc import LIDC, LUNA16 as LUNA16Neurodata
# from ood.paths import LIDC_SUPPLEMENTARY_PATH


class LUNA16(Proxy):
    def __init__(self):
        super().__init__(LIDC()) # TODO change to luna

    def load_image(self, i):
        return np.float32(self.image(i))

    def load_segm(self, i):
        return np.float32(self.cancer(i))

    def load_spacing(self, i):
        diffs, counts = np.unique(np.round(np.diff(self.slice_locations(i)), decimals=5), return_counts=True)
        pixel_spacing = self.pixel_spacing(i)
        return np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
    
    def n_tumors(self, i):
        return len(self.nodules(i))
        
#     def load_lungs(self, i):
#         return load(os.path.join(LIDC_SUPPLEMENTARY_PATH, str(i)))
    
    
def get_n_tumors(dataset, ids):
    n_tumors = []
    for uid in ids:
        n_tumors.append(dataset.n_tumors(uid))
    return n_tumors