import numpy as np

from dpipe.dataset.wrappers import Proxy
from neurodata.lits import LiTS as LiTSNeurodata


class LiTS(Proxy):
    def __init__(self):
        super().__init__(LiTSNeurodata())
        self.ids = [uid for uid in self.ids if not uid.startswith('lits-test-')]

    def load_image(self, i):
        return np.float32(self.image(i))

    def load_spacing(self, i):
        return np.float32(self.spacing(i))