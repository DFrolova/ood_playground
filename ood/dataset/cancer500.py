import numpy as np
from connectome import Transform


class CancerMask(Transform):
    __inherit__ = True
    _center = None
    _diameter = None

    def _image_shape(image):
        return image.shape

    def mask(nodules, voxel_spacing, _image_shape):

        shape = _image_shape

        mask = np.zeros(shape, dtype=bool)
        if nodules is not None:
            for nodule in nodules:
                center_sum = 0
                diameters = []
                for ann_nodule in nodule.values():
                    center_sum += np.array(ann_nodule.center_voxel)
                    diameters.append(ann_nodule.diameter_mm)

                center = np.round(center_sum / len(nodule))
                diameters = np.array(diameters, dtype=float)
                if np.isnan(diameters).all():
                    continue

                diameter = np.nanmean(diameters)

                # create circular mask
                X, Y, Z = np.ogrid[:shape[0], :shape[1], :shape[2]]
                dist_from_center = np.sqrt(voxel_spacing[0] * (X - center[0]) ** 2 +
                                           voxel_spacing[1] * (Y - center[1]) ** 2 +
                                           voxel_spacing[2] * (Z - center[2]) ** 2)

                current_mask = dist_from_center <= (diameter / 2)

                mask |= current_mask

        return mask


class CanonicalCTOrientation(Transform):
    __inherit__ = True

    def image(image):
        return np.transpose(image, (1, 0, 2))[..., ::-1]

    def mask(mask):
        return np.transpose(mask, (1, 0, 2))[..., ::-1]

    def voxel_spacing(voxel_spacing):
        return tuple(np.array(voxel_spacing)[[1, 0, 2]].tolist())
