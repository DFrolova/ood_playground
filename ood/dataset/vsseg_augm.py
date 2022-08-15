# import numpy as np
# from connectome import Transform
#
# from ood.dataset.augm_transforms import elastic_transform, blur_transform, slice_drop_transform, contrast_transform, \
#     corruption_transform, pixel_shuffling_transform
#
#
# class ApplyAugm(Transform):
#     __inherit__ = True
#
#     _transform_fn = None
#     _param = .5
#
#     def _transformed(image, cancer, _modified_id, _transform_fn, _param):
#         random_state = hash(_modified_id) % (2 ** 32)
#         print(_modified_id, hash(_modified_id), random_state)
#         if _transform_fn == elastic_transform:
#             result = _transform_fn(img=image, mask=cancer.astype(np.float32), param=_param, random_state=random_state)
#         else:
#             result = _transform_fn(img=image, param=_param, random_state=random_state)
#         return result
#
#     def image(_transformed, _transform_fn):
#         print(_transform_fn)
#         if _transform_fn == elastic_transform:
#             image_new, mask_new = _transformed
#         else:
#             image_new = _transformed
#         return image_new
#
#     def cancer(cancer, _transformed, _transform_fn):
#         if _transform_fn == elastic_transform:
#             image_new, cancer_new = _transformed
#         else:
#             cancer_new = cancer
#         return cancer_new
#
#     def _modified_id(id, _transform_fn, _param):
#         return '__'.join([str(_param), _transform_fn.__name__, id])
#
#     def modified_id(_modified_id):
#         return _modified_id
#
#
#
# # import warnings
# # import zipfile
# # from functools import lru_cache
# # from pathlib import Path
# # from zipfile import ZipFile
# #
# # import numpy as np
# # from connectome import Source, meta
# # from connectome.interface.nodes import Silent
# # from dpipe.io import load
# # from amid.vs_seg import VSSEG
# #
# #
# # class VSSEG_augm(Source):
# #
# #     _root: str = None
# #
# #     @meta
# #     def ids(_root: Silent):
# #         result = set()
# #         for folder in Path(_root).glob('*/*/*'):
# #             if 'mask' not in str(folder):
# #                 result.add('__'.join(folder.parts[-3:]).split('.npy')[0])
# #
# #         return tuple(sorted(result))
# #
# #     @lru_cache
# #     def _vsseg_dataset():
# #         return VSSEG()
# #
# #     def noise_param(i, _root: Silent):
# #         noise, augm_fn, uid = i.split('__')
# #         return noise
# #
# #     def augm_transform(i, _root: Silent):
# #         noise, augm_fn, uid = i.split('__')
# #         return augm_fn
# #
# #     def image_t1(i, _root: Silent):
# #         noise, augm_fn, uid = i.split('__')
# #         return np.float16(load(Path(_root) / noise / augm_fn / f'{uid}.npy'))
# #
# #     def schwannoma_t1(i, _root: Silent, _vsseg_dataset):
# #         noise, augm_fn, uid = i.split('__')
# #         if augm_fn == 'elastic_transform':
# #             return load(Path(_root) / noise / augm_fn / f'mask_{uid}.npy')
# #         return _vsseg_dataset.schwannoma_t1(uid)
# #
# #     def voxel_spacing_t1(i, _vsseg_dataset):
# #         noise, augm_fn, uid = i.split('__')
# #         return _vsseg_dataset.voxel_spacing_t1(uid)
# #