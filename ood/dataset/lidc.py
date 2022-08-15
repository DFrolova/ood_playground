import numpy as np

from connectome import Transform
from skimage.measure import label


class CanonicalCTOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.flip(image, axis=-1)

    def mask(mask):
        return np.flip(mask, axis=-1)


class RenameFields(Transform):
    __inherit__ = True

    def mask(cancer):
        return cancer


class NumberOfTumors(Transform):
    __inherit__ = True

    def n_tumors(cancer):
        return None if (cancer is None) else label(cancer, return_num=True, connectivity=3)[-1]
