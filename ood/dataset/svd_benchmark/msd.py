from pathlib import Path
from glob import glob

import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent

from amid.cc359 import open_nii_gz_file


class MSD(Source):
    _root: str = None

    @meta
    def ids(_root: Silent):
        result = set()
        for filename in glob(f'{_root}/*/images*/*'):
            result.add('__'.join(Path(filename).parts[-3:]).strip('.nii.gz'))
        return tuple(sorted(result))

    def task(i):
        return i.split('__')[0]

    def fold(i):
        return i.split('__')[1]

    def image(i, _root: Silent):
        task, fold, fname = i.split('__')
        with open_nii_gz_file(Path(_root) / task / fold / f'{fname}.nii.gz') as nii_image:
            image = nii_image.get_fdata().astype(np.float32)
            
        if len(image.shape) > 3:
            image = image[..., 0]#.reshape(image.shape[0], image.shape[1], -1)
        return image.astype(np.float32)

    def voxel_spacing(i, _root: Silent):
        task, fold, fname = i.split('__')
        with open_nii_gz_file(Path(_root) / task / fold / f'{fname}.nii.gz') as nii_image:
            return tuple(nii_image.header['pixdim'][1:4])
        
    def modality(i):
        task, fold, fname = i.split('__')
        if task in ['Task07_Pancreas', 'Task09_Spleen']:
            return 'CT'
        return 'MRI'
