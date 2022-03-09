import os

import numpy as np

from dpipe.dataset.wrappers import Proxy
from dpipe.io import load
from neurodata.medseg_covid_public import MedsegCOVIDPublic9


class Medseg9(Proxy):
    def __init__(self):
        super().__init__(MedsegCOVIDPublic9())
        
    def load_image(self, i):
        return np.float32(self.image(i))

    def load_spacing(self, i):
        return np.float32(self.spacing(i))
    
    def load_segm(self, i):
        return np.float32(self.covid(i))