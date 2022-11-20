import numpy as np
from connectome import Transform


class RenameFields(Transform):
    __inherit__ = True

    def image(image_t1):
        return image_t1

    def mask(schwannoma_t1):
        return schwannoma_t1

    def voxel_spacing(voxel_spacing_t1):
        return voxel_spacing_t1


class CanonicalMRIOrientation(Transform):
    __inherit__ = True
    
    def image(image):
        return np.transpose(image, (1, 0, 2))[..., ::-1]
        
    def voxel_spacing(voxel_spacing):
        return tuple(np.array(voxel_spacing)[[1, 0, 2]].tolist())
