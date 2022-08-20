from typing import Union

import numpy as np
from connectome import Transform

from dpipe.im.shape_ops import zoom, crop_to_box, pad


class Shape(Transform):
    __inherit__ = True

    def shape(image):
        return image.shape


class ScaleHU(Transform):
    __inherit__ = True

    # Standard lung window by default:
    _min_hu: int = -1350
    _max_hu: int = 300

    def image(image, _min_hu, _max_hu):
        assert _max_hu > _min_hu
        image = np.clip(image, _min_hu, _max_hu)
        min_val = np.min(image)
        max_val = np.max(image)
        return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


class ScaleUniversal(Transform):
    __inherit__ = True
    
    # Standard lung window by default:
    _min_hu: int = -1350
    _max_hu: int = 300

    _q_min: int = 1
    _q_max: int = 99

    def image(image, modality, _min_hu, _max_hu, _q_min, _q_max):
        if modality == 'MRI':
            image = np.clip(image, *np.percentile(np.float32(image), [_q_min, _q_max]))
            min_val = np.min(image)
            max_val = np.max(image)
            return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)
        # modality == 'CT'
        image = np.clip(image, _min_hu, _max_hu)
        min_val = np.min(image)
        max_val = np.max(image)
        return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


class ScaleMRI(Transform):
    __inherit__ = True

    _q_min: int = 1
    _q_max: int = 99

    def image(image, _q_min, _q_max):
        image = np.clip(image, *np.percentile(np.float32(image), [_q_min, _q_max]))
        min_val = np.min(image)
        max_val = np.max(image)
        return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


class Zoom(Transform):
    __inherit__ = True
    _new_spacing: Union[tuple, float, int] = (None, None, None)
    _order = 1

    def _scale_factor(voxel_spacing, _new_spacing):
        if not isinstance(_new_spacing, (tuple, list, np.ndarray)):
            _new_spacing = np.broadcast_to(_new_spacing, 3)
        return np.nan_to_num(np.float32(voxel_spacing) / np.float32(_new_spacing), nan=1)

    def image(image, _scale_factor, _order):
        return np.array(zoom(image.astype(np.float32), _scale_factor, order=_order))

    def mask(mask, _scale_factor):
        return np.array(zoom(mask.astype(np.float32), _scale_factor) > 0.5, dtype=mask.dtype)


class ZoomImage(Transform):
    __inherit__ = True
    _new_spacing: Union[tuple, float, int] = (None, None, None)
    _order = 1

    def _scale_factor(voxel_spacing, _new_spacing):
        if not isinstance(_new_spacing, (tuple, list, np.ndarray)):
            _new_spacing = np.broadcast_to(_new_spacing, 3)
        return np.nan_to_num(np.float32(voxel_spacing) / np.float32(_new_spacing), nan=1)

    def image(image, _scale_factor, _order):
        return np.array(zoom(image.astype(np.float32), _scale_factor, order=_order))


class VoxelSpacing(Transform):
    __inherit__ = True

    def voxel_spacing(pixel_spacing, slice_locations):
        slice_spacing = abs(np.round(np.median(np.diff(slice_locations)), decimals=5))
        return np.float32([pixel_spacing[0], pixel_spacing[1], slice_spacing])
