import numpy as np
from connectome import Transform


class CanonicalCTOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.swapaxes(image, 0, 1)

    def voxel_spacing(voxel_spacing):
        return np.array(voxel_spacing)[[1, 0, 2]]
