import os
from collections import defaultdict
from typing import Union
from functools import lru_cache
import hashlib

import numpy as np
from tqdm import tqdm
from connectome import Transform

from dpipe.io import save_json, load_pred, save, load, load_json
from ood.utils import get_lib_root_path


def elastic_transform(img: np.ndarray, mask: np.ndarray = None, param: float = 0.5, random_state: int = 5):
    from albumentations.augmentations.geometric.transforms import ElasticTransform
    import cv2
    from skimage.measure import label

    fill_value = int(img.min())
    border_mode = cv2.BORDER_CONSTANT

    t = ElasticTransform(alpha=12000 * param, sigma=np.mean(img.shape[:2]) / 11., alpha_affine=0,
                         always_apply=True, border_mode=border_mode, value=fill_value)
    t.set_deterministic(True)

    img_new = t.apply(img, random_state=random_state)
    if mask is not None:
        mask_new = t.apply_to_mask(mask, random_state=random_state)

    lbl = label(np.pad(img_new, [[1, 1], [1, 1], [0, 0]]) == 0, connectivity=3)
    lbl = lbl[1:-1, 1:-1, :]

    # always lbl == 1 because algorithm starts at the corner of the image.
    img_new[lbl == 1] = fill_value

    if mask is None:
        return img_new
    return img_new, mask_new


def blur_transform(img: np.ndarray, param: float = 0.5, random_state: int = 5):
    from skimage.filters import gaussian
    return gaussian(np.float32(img), sigma=5 * param, preserve_range=True)


def slice_drop_transform(img: np.ndarray, param: float = 0.5,
                         random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    drop_indexes = np.nonzero(random_state.binomial(1, param, size=img.shape[-1]))[0]

    fill_value = img.min()

    img_new = img.copy()
    for i in drop_indexes:
        img_new[..., i] = np.zeros_like(img_new[..., i]) + fill_value

    return img_new


def sample_box(img: np.ndarray, param: float = 0.5, random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    img_shape = np.int16(img.shape)
    box_shape = np.int16(np.round(np.float32(img.shape) * param))
    center_min = box_shape // 2
    center_max = img_shape - box_shape // 2

    center = np.int16([random_state.randint(cmin, cmax) for cmin, cmax in zip(center_min, center_max)])
    return [center - box_shape // 2, center + box_shape // 2]


def min_max_scale(img: np.ndarray):
    img = np.float32(img)
    img -= img.min()
    img /= img.max()
    return img


def min_max_descale(img: np.ndarray, minv, maxv):
    img = np.float32(img)
    img *= (maxv - minv)
    img += minv
    return img


def contrast_transform(img: np.ndarray, param: float = 0.5,
                       random_state: int = 5):
    from skimage.exposure import adjust_gamma

    random_state_np = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    while minv == maxv:
        print('resampling bbox because crop.min() == crop.max()', flush=True)
        random_state += 1
        random_state_np = np.random.RandomState(random_state)

        box = sample_box(img, param, random_state)
        crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
        minv, maxv = crop.min(), crop.max()

    gamma = 4 if (random_state_np.random_sample() >= 0.5) else 0.25
    crop_corrected = adjust_gamma(min_max_scale(crop), gamma=gamma)
    crop_corrected = min_max_descale(crop_corrected, minv, maxv)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_corrected

    return img_new


def corruption_transform(img: np.ndarray, param: float = 0.5,
                         random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    minv, maxv = crop.min(), crop.max()

    crop_corrupted = min_max_descale(random_state.rand(*crop.shape), minv, maxv)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_corrupted

    return img_new


def pixel_shuffling_transform(img: np.ndarray, param: float = 0.5,
                              random_state: Union[int, np.random.RandomState] = 5):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    box = sample_box(img, param, random_state)
    crop = np.copy(img[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]])
    crop_shape = np.copy(np.asarray(crop.shape))

    crop_shuffled = crop.ravel()
    random_state.shuffle(crop_shuffled)
    crop_shuffled = np.reshape(crop_shuffled, crop_shape)

    img_new = np.copy(img)
    img_new[box[0][0]:box[1][0], box[0][1]:box[1][1], box[0][2]:box[1][2]] = crop_shuffled

    return img_new


class ApplyAllAugms(Transform):
    __inherit__ = True

    _param_file = None
    _transform_file = None
    
    @lru_cache(None)
    def _param_dict(_param_file):
        return load(os.path.join(get_lib_root_path(), 'configs/assets/dataset/', _param_file))
    
    @lru_cache(None)
    def _transform_dict(_transform_file):
        return load(os.path.join(get_lib_root_path(), 'configs/assets/dataset/', _transform_file))
    
    _str_to_tranform_fn = {
        'elastic_transform': elastic_transform, 
        'blur_transform': blur_transform, 
        'slice_drop_transform': slice_drop_transform, 
        'contrast_transform': contrast_transform,
        'corruption_transform': corruption_transform,
        'pixel_shuffling_transform': pixel_shuffling_transform,
    }

    def _transformed(image, mask, _modified_id, _transform_fn, _param):
        random_state = int(hashlib.sha1(str.encode(_modified_id)).hexdigest(), 16) % (2 ** 32)
        if _transform_fn == elastic_transform:
            result = _transform_fn(img=image, mask=mask.astype(np.float32), param=_param, random_state=random_state)
        else:
            result = _transform_fn(img=image, param=_param, random_state=random_state)
        return result

    def image(_transformed, _transform_fn):
        if _transform_fn == elastic_transform:
            image_new, _ = _transformed
        else:
            image_new = _transformed
        return image_new

    def mask(mask, _transformed, _transform_fn):
        if _transform_fn == elastic_transform:
            _, mask_new = _transformed
        else:
            mask_new = mask
        return mask_new

    def _modified_id(id, _transform_fn, _param):
        return '__'.join([str(_param), _transform_fn.__name__, id])

    def modified_id(_modified_id):
        return _modified_id
    
    def _param(id, _param_dict):
        return _param_dict[id]
        
    def _transform_fn(id, _transform_dict, _str_to_tranform_fn):
        return _str_to_tranform_fn[_transform_dict[id]]


class ApplyAugm(Transform):
    __inherit__ = True

    _transform_fn = None
    _param = .5

    def _transformed(image, mask, _modified_id, _transform_fn, _param):
        random_state = int(hashlib.sha1(str.encode(_modified_id)).hexdigest(), 16) % (2 ** 32)
        if _transform_fn == elastic_transform:
            result = _transform_fn(img=image, mask=mask.astype(np.float32), param=_param, random_state=random_state)
        else:
            result = _transform_fn(img=image, param=_param, random_state=random_state)
        return result

    def image(_transformed, _transform_fn):
        if _transform_fn == elastic_transform:
            image_new, _ = _transformed
        else:
            image_new = _transformed
        return image_new

    def mask(mask, _transformed, _transform_fn):
        if _transform_fn == elastic_transform:
            _, mask_new = _transformed
        else:
            mask_new = mask
        return mask_new

    def _modified_id(id, _transform_fn, _param):
        return '__'.join([str(_param), _transform_fn.__name__, id])

    def modified_id(_modified_id):
        return _modified_id


def evaluate_individual_metrics_no_pred_with_augm_transforms(load_x_fns, load_y_fns, full_uid_fns, predict,
                                                             metrics: dict, test_ids, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for uid_num, identifier in enumerate(tqdm(test_ids)):

        for load_x, load_y, full_uid_fn in zip(load_x_fns, load_y_fns, full_uid_fns):

            image = load_x(identifier)
            target = load_y(identifier)

            full_identifier = full_uid_fn(identifier)
            prediction = predict(image)

            for metric_name, metric in metrics.items():
                try:
                    results[metric_name][full_identifier] = metric(target, prediction, identifier)
                except TypeError:
                    results[metric_name][full_identifier] = metric(target, prediction)

        # save everytime just for convenience
        for metric_name, result in results.items():
            save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

            
# def evaluate_individual_metrics_no_pred_with_augm_transforms(load_y, load_x, predict, metrics: dict, test_ids,
#                                                              results_path, param_dict, transform_fns, exist_ok=False):
#     assert len(metrics) > 0, 'No metric provided'
#     os.makedirs(results_path, exist_ok=exist_ok)
#
#     results = defaultdict(dict)
#     for uid_num, identifier in enumerate(tqdm(test_ids)):
#
#         image = load_x(identifier)
#         target = load_y(identifier)
#
#         for transform_id, transform_fn in enumerate(transform_fns):
#             for cur_param in param_dict[transform_fn]:
#                 full_identifier = '__'.join([str(cur_param), transform_fn.__name__, identifier])
#                 # apply transform
#                 if transform_fn == elastic_transform:
#                     img_ood, mask_ood = transform_fn(image, mask=target, param=cur_param,
#                                                      random_state=transform_id * len(test_ids) + uid_num)
#                 else:
#                     img_ood = transform_fn(image, param=cur_param, random_state=transform_id * len(test_ids) + uid_num)
#                     mask_ood = target
#
#                 prediction = predict(img_ood)
#
#                 for metric_name, metric in metrics.items():
#                     try:
#                         results[metric_name][full_identifier] = metric(mask_ood, prediction, identifier)
#                     except TypeError:
#                         results[metric_name][full_identifier] = metric(mask_ood, prediction)
#
#     for metric_name, result in results.items():
#         save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
