import os

import numpy as np

from dpipe.dataset.wrappers import Proxy
from dpipe.io import load
from neurodata.midrc_ricord import MIDRCRICORD1a


class MIDRC(Proxy):
    def __init__(self, covid_degrees_path='/homes/dfrolova/ood-playground/ood/dataset/midrc_covid_degree.json'):
        super().__init__(MIDRCRICORD1a())
#         # load
#         cur_path = str(os.path.abspath('.'))
#         folders = cur_path.split('/')
#         i = len(folders) - 1
#         while folders[i] != 'ood-playground':
#             i -= 1
#         self._covid_degrees = load(os.path.join('/'.join(folders[:i+1]), covid_degrees_path))
        self._covid_degrees = load(covid_degrees_path)
        self.ids = tuple(sorted(self._covid_degrees.keys()))
        
    def get_covid_degree(self, i):
        return self._covid_degrees[i]
        
    def load_image(self, i):
        return np.float32(self.image(i))

    def load_spacing(self, i):
        diffs, counts = np.unique(np.round(np.diff(self.slice_locations(i)), decimals=5), return_counts=True)
        pixel_spacing = self.pixel_spacing(i)
        return np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
    
    def load_segm(self, i):
        inf_opacity = self.inf_opacity(i)
        inf_nodules = self.inf_nodules(i)
        shape = self.load_image(i).shape
        
        covid = np.zeros(shape, dtype=bool)

        if inf_opacity is not None:
            covid |= inf_opacity

        if inf_nodules is not None:
            covid |= inf_nodules

        return np.float32(covid)