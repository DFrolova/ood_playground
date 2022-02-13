import numpy as np
from skimage import measure
from skimage.segmentation import flood

from dpipe.dataset.wrappers import Proxy
from dpipe.im.shape_ops import zoom, crop_to_box, pad
from dpipe.im.box import mask2bounding_box


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)


class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=3):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)
    
    def load_tumor_centers(self, i):
        segm = self.load_segm(i=i)
        y_bin = segm == 1.
        labels, n_labels = measure.label(y_bin, connectivity=2, return_num=True)
        result_centers = [np.argwhere(labels == label) for label in range(1, n_labels + 1)]
        lengths_cc = np.array([len(cc) for cc in result_centers])
        probas = np.array([1 / len_cc for len_cc in lengths_cc]) / n_labels
        broadcated_probas = np.repeat(probas, lengths_cc)[:, None]
        return np.hstack((np.vstack(result_centers), broadcated_probas))
    
    
class CropToLungs(Change):
    def __init__(self, shadowed, lungs_threshold=-600, lungs_fraction_threshold=0.005):
        super().__init__(shadowed)
        self.lungs_threshold = lungs_threshold
        self.lungs_fraction_threshold = lungs_fraction_threshold
        
    def _bbox(self, i):
        x = self._shadowed.load_image(i)
        # find lungs
        lungs_mask = x < self.lungs_threshold
        # find air
        air_mask = flood(pad(lungs_mask, padding=1, axis=(0, 1), padding_values=True),
                         seed_point=(0, 0, 0))[1:-1, 1:-1]
        lungs_mask = lungs_mask & ~air_mask

        # filter lungs as biggest connected components
        labels, n_labels = measure.label(lungs_mask, connectivity=2, return_num=True)
        result_centers = [np.argwhere(labels == label) for label in range(1, n_labels + 1)]
        fractions_cc = np.array([len(cc) for cc in result_centers]) / np.prod(lungs_mask.shape)

        lungs_mask_final = np.zeros_like(lungs_mask, dtype=bool)
        for i, frac in enumerate(fractions_cc):
            if frac > self.lungs_fraction_threshold:
                lungs_mask_final = lungs_mask_final | labels == (i + 1)
        box = mask2bounding_box(lungs_mask_final)
        return box

    def _change(self, x, i):
        return crop_to_box(x, self._bbox(i))

    def load_orig_image(self, i):
        return self._shadowed.load_image(i)


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


def scale_ct(x: np.ndarray, min_value: float = -1350, max_value: float = 300) -> np.ndarray:
    x = np.clip(x, a_min=min_value, a_max=max_value)
    x -= np.min(x)
    x /= np.max(x)
    return np.float32(x)
