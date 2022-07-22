import numpy as np
from connectome import Transform


class CanonicalCTOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.moveaxis(image, 1, 0)[::-1, :, ::-1]

    def mask(mask):
        return np.moveaxis(mask, 1, 0)[::-1, :, ::-1]

    def voxel_spacing(voxel_spacing):
        return np.array(voxel_spacing)[[1, 0, 2]]
