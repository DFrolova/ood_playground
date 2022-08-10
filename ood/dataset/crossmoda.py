import numpy as np
from connectome import Transform


class RenameFields(Transform):
    __inherit__ = True

    def voxel_spacing(pixel_spacing):
        return pixel_spacing

    def mask(masks):
        return masks == 1


class CanonicalMRIOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.moveaxis(image, 1, 0)[:, :, ::-1]

    def mask(mask):
        return np.moveaxis(mask, 1, 0)[:, :, ::-1]

    def voxel_spacing(voxel_spacing):
        return np.array(voxel_spacing)[[1, 0, 2]]