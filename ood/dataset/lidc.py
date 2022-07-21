import numpy as np

from connectome import Transform
from skimage.measure import label


class CanonicalCTOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.flip(image, axis=-1)

    def mask(cancer):
        return np.flip(cancer, axis=-1)


class NumberOfTumors(Transform):
    __inherit__ = True

    def n_tumors(cancer):
        return None if (cancer is None) else label(cancer, return_num=True, connectivity=3)[-1]
