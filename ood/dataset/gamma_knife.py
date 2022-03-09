from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import grey_opening, label as _label

from dpipe.im.shape_ops import zoom
from dpipe.io import load
from dpipe.dataset.wrappers import Proxy
from ood.batch_iter.pipeline import SPATIAL_DIMS
from dpipe.im.box import mask2bounding_box
from dpipe.im import crop_to_box
from dpipe.im.utils import build_slices


class GammaKnife:
    def __init__(self, data_path, metadata_rpath, index_col='ID',
                 t1c='t1c', t1='t1', target='brain', brain_mask='brain_mask', tumor_mask='tumor_mask'):
        self.data_path = Path(data_path)
        self.metadata_rpath = metadata_rpath
        self.index_col = index_col

        self.df = pd.read_csv(self.data_path / metadata_rpath, index_col=index_col)
        self.ids = tuple(self.df.index.tolist())

        self.t1c = t1c
        self.t1 = t1

        self.brain_option = 'brain'
        self.tumor_option = 'tumor'
        if target not in (self.brain_option, self.tumor_option):
            raise ValueError(f'`target` should be either `{self.brain_option}` or `{self.tumor_option}`; '
                             f'however, `{target}` is given.')
        self.target = target

        self.brain_mask = brain_mask
        self.tumor_mask = tumor_mask

    def load_image(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.t1c]))

    def load_t1_image(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.t1]))

    def load_segm(self, i):
        tumor_mask = load(self.data_path / self.df.loc[i][self.tumor_mask]) > 0.5
        brain_mask = load(self.data_path / self.df.loc[i][self.brain_mask]) > 0.5
        return np.float32(tumor_mask | brain_mask)

    def load_tumor_segm(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.tumor_mask]) > 0.5)

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        # TODO: handle different modalities
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)

    def load_t1_image(self, i):
        return self._change(self._shadowed.load_t1_image(i), i)

    def load_tumor_segm(self, i):
        return np.float32(self._change(self._shadowed.load_tumor_segm(i), i) >= .5)


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
    

class RotateImage(Change):
    def _change(self, x, i):
        return np.flip(np.rot90(x, k=3, axes=(-3, -2)), axis=SPATIAL_DIMS[2])
    
    
class CropToScull(Change):
    def _skull_bbox(self, scan, kernel_size=14, q=70, k=10):
        scan = scan - scan.min()

        # Opening to get rid of steel
        scan_opening = grey_opening(scan, size=[kernel_size] * 3)
        labels = _label(scan_opening > np.percentile(scan_opening, q))[0]

        l, c = np.unique(labels, return_counts=True)
        mask = labels == l[np.argmax(c[1:]) + 1]

        start, stop = mask2bounding_box(mask)

        shift = np.array([10, 10, 10])
        start = np.clip(start - shift, 0, mask.shape)
        stop = np.clip(stop + shift, 0, mask.shape)

        mask = np.zeros_like(scan, dtype=bool)
        mask[build_slices(start, stop)] = True
        bbox = mask2bounding_box(mask & (scan > scan.max() / k))

        return bbox
    
    def _change(self, x, i):
        return crop_to_box(x, self._skull_bbox(self._shadowed.load_image(i)))