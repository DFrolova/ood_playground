




# import warnings
# import zipfile
# from functools import lru_cache
# from pathlib import Path
# from zipfile import ZipFile

# import numpy as np
# from connectome import Source, meta
# from connectome.interface.nodes import Silent
# from dpipe.io import load
# from amid.lidc import LIDC


# class LIDC_augm(Source):

#     _root: str = None

#     @meta
#     def ids(_root: Silent):
#         result = set()
#         for folder in Path(_root).glob('*/*/*'):
#             if 'mask' not in str(folder):
#                 result.add('__'.join(folder.parts[-3:]).split('.npy')[0])

#         return tuple(sorted(result))

#     @lru_cache
#     def _lidc_dataset():
#         return LIDC()

#     def noise_param(i, _root: Silent):
#         noise, augm_fn, uid = i.split('__')
#         return noise

#     def augm_transform(i, _root: Silent):
#         noise, augm_fn, uid = i.split('__')
#         return augm_fn

#     def image(i, _root: Silent):
#         noise, augm_fn, uid = i.split('__')
#         return np.float16(load(Path(_root) / noise / augm_fn / f'{uid}.npy'))

#     def cancer(i, _root: Silent, _lidc_dataset):
#         noise, augm_fn, uid = i.split('__')
#         if augm_fn == 'elastic_transform':
#             return load(Path(_root) / noise / augm_fn / f'mask_{uid}.npy')
#         return _lidc_dataset.cancer(uid)

#     def voxel_spacing(i, _lidc_dataset):
#         noise, augm_fn, uid = i.split('__')
#         return _lidc_dataset.voxel_spacing(uid)

