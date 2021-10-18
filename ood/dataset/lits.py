import numpy as np

from dpipe.dataset.wrappers import Proxy
from neurodata.lits import LiTS as LiTSNeurodata


class LiTS(Proxy):
    def __init__(self):
        super().__init__(LiTSNeurodata())

    def load_image(self, i):
        return np.float32(self.image(i))

    def load_segm(self, i):
        if 'test' in i:
            raise IndexError('Trying to load mask for the test image')
        else:
            return np.float32(self.segmentation(i))

    def load_spacing(self, i):
        return np.float32(self.spacing(i))
