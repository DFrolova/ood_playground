import numpy as np

from dpipe.dataset.wrappers import Proxy
from dpipe.io import load
from neurodata.cancer_500 import Cancer500 as Cancer_neurodata
from ood.utils import get_lib_root_path


class Cancer500(Proxy):
    def __init__(self, cancer500_ids_path='ood/dataset/cancer500_ids.json'):
        super().__init__(Cancer_neurodata())

        lib_root_path = get_lib_root_path()
        self._cancer500_ids = load(lib_root_path / cancer500_ids_path)
        self.ids = tuple(sorted(self._cancer500_ids))

    def load_image(self, i):
        return np.float32(self.image(i))

    def load_spacing(self, i):
        diffs, counts = np.unique(np.round(np.diff(self.slice_locations(i)), decimals=5), return_counts=True)
        pixel_spacing = self.pixel_spacing(i)
        return np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])

    def _create_circular_mask(self, image_shape, image_spacing, center, diameter):

        X, Y, Z = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
        dist_from_center = np.sqrt(image_spacing[0] * (X - center[0]) ** 2 +
                                   image_spacing[1] * (Y - center[1]) ** 2 +
                                   image_spacing[2] * (Z - center[2]) ** 2)

        mask = dist_from_center <= (diameter / 2)
        return mask

    def load_segm(self, i):
        nodules = self.nodules(i)
        image_shape = self.image(i).shape
        image_spacing = self.load_spacing(i)

        mask = np.zeros(image_shape, dtype=bool)
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

                mask |= self._create_circular_mask(image_shape, image_spacing, center, diameter)

        return np.float32(mask)
