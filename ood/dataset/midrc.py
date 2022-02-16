import os

import numpy as np

from dpipe.dataset.wrappers import Proxy
from dpipe.io import load
from neurodata.midrc_ricord import MIDRCRICORD1a


class MIDRC(Proxy):
    def __init__(self):
        super().__init__(MIDRCRICORD1a())

    def load_image(self, i):
        return np.float32(self.image(i))

    def load_spacing(self, i):
        diffs, counts = np.unique(np.round(np.diff(self.slice_locations(i)), decimals=5), return_counts=True)
        pixel_spacing = self.pixel_spacing(i)
        return np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])