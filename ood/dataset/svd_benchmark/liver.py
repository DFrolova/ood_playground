import os.path
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from connectome import Source, meta
from connectome.interface.nodes import Silent
from dicom_csv import (expand_volumetric, drop_duplicated_instances,
                       drop_duplicated_slices, order_series, stack_images,
                       get_slice_locations, get_pixel_spacing, get_tag, join_tree)


class LiverDataset(Source):
    _root: str = None

    @meta
    def ids(_joined):
        modality_dict = {True: 'CT', False: 'MRI'}
        unique_rows = _joined[['ProtocolName', 'SeriesInstanceUID', 'PathToFolder']].drop_duplicates()
        unique_rows['modality'] = unique_rows['PathToFolder'].str.contains('CT').map(modality_dict)
        unique_rows['ids'] = unique_rows['modality'] + '__' + unique_rows['ProtocolName'] + '__' + unique_rows[
            'SeriesInstanceUID']
        return tuple(sorted(unique_rows['ids']))

    @lru_cache(None)
    def _joined(_root: Silent):
        if os.path.exists(Path(_root) / "joined.csv"):
            return pd.read_csv(Path(_root) / "joined.csv")
        joined = join_tree(Path(_root))
        joined = joined[joined['NoError']]
        joined = joined[[x.endswith('.dcm') for x in joined.FileName]]
        joined.to_csv(Path(_root) / "joined.csv")
        return joined

    def _series(i, _root: Silent, _joined):
        modality, protocol, uid = i.split('__')
        sub = _joined[_joined.SeriesInstanceUID == uid]
        sub = sub[sub.ProtocolName == protocol]
        series_files = sub['PathToFolder'] + os.path.sep + sub['FileName']
        series_files = [Path(_root) / x for x in series_files]
        series = list(map(pydicom.dcmread, series_files))
        series = expand_volumetric(series)
        series = drop_duplicated_instances(series)

        _original_num_slices = len(series)
        series = drop_duplicated_slices(series)
        if len(series) < _original_num_slices:
            warnings.warn(f'Dropped duplicated slices for series {series[0]["StudyInstanceUID"]}.')

        series = order_series(series)
        return series

    def image(i, _series):
        modality, _, _ = i.split('__')
        image = stack_images(_series, -1)
        return image.astype(np.float32)

    def voxel_spacing(_series):
        pixel_spacing = get_pixel_spacing(_series).tolist()
        slice_locations = get_slice_locations(_series)
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
        return spacing

    def protocol_name(i):
        return i.split('__')[1]

    def modality(i):
        return i.split('__')[0]
