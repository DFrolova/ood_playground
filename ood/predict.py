import numpy as np
from functools import wraps
from typing import Union, Callable, Type

from dpipe.itertools import lmap
from dpipe.im import pad_to_divisible, crop_to_shape, pad_to_shape
from dpipe.im.grid import PatchCombiner, Average, combine, divide
from dpipe.im.slices import iterate_slices
from dpipe.im.shape_utils import prepend_dims
from dpipe.batch_iter import unpack_args
from dpipe.im.axes import AxesLike, AxesParams, axis_from_dim, broadcast_to_axis, resolve_deprecation
from dpipe.itertools import extract, pmap


def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper


def divisible_shape_ood(divisor: AxesLike, axis: AxesLike = None, padding_values: Union[AxesParams, Callable] = 0,
                        ratio: AxesParams = 0.5):
    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            local_axis = axis_from_dim(axis, x.ndim)
            local_divisor, local_ratio = broadcast_to_axis(local_axis, divisor, ratio)
            shape = np.array(x.shape)[list(local_axis)]

            x = pad_to_divisible(x, local_divisor, local_axis, padding_values, local_ratio)
            result_segm, result_feat = predict(x, *args, **kwargs)

            return (crop_to_shape(result_segm, shape, local_axis, local_ratio),
                    crop_to_shape(result_feat, shape, local_axis, local_ratio))

        return wrapper

    return decorator


def patches_grid_ood(patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
                     padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5,
                     combiner: Type[PatchCombiner] = Average):
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            input_axis = resolve_deprecation(axis, x.ndim, patch_size, stride)
            local_size, local_stride = broadcast_to_axis(input_axis, patch_size, stride)

            if valid:
                shape = extract(x.shape, input_axis)
                padded_shape = np.maximum(shape, local_size)
                new_shape = padded_shape + (local_stride - padded_shape + local_size) % local_stride
                x = pad_to_shape(x, new_shape, input_axis, padding_values, ratio)

            patches = pmap(predict, divide(x, local_size, local_stride, input_axis), *args, **kwargs)

            patches_segm, patches_feat = [], []
            for p in patches:
                patches_segm.append(p[0])
                patches_feat.append(p[1])

            pred_segm = combine(patches_segm, extract(x.shape, input_axis), local_stride, axis, combiner=combiner)
            pred_feat = combine(patches_feat, extract(x.shape, input_axis), local_stride, axis, combiner=combiner)

            if valid:
                pred_segm = crop_to_shape(pred_segm, shape, axis, ratio)
                pred_feat = crop_to_shape(pred_feat, shape, axis, ratio)
            return pred_segm, prepend_dims(pred_feat, ndim=1)

        return wrapper

    return decorator


def divisible_shape_ood_single_feature(divisor: AxesLike, axis: AxesLike = None,
                                       padding_values: Union[AxesParams, Callable] = 0,
                                       ratio: AxesParams = 0.5):
    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            local_axis = axis_from_dim(axis, x.ndim)
            local_divisor, local_ratio = broadcast_to_axis(local_axis, divisor, ratio)
            shape = np.array(x.shape)[list(local_axis)]

            x = pad_to_divisible(x, local_divisor, local_axis, padding_values, local_ratio)
            result_feat = predict(x, *args, **kwargs)

            return crop_to_shape(result_feat, shape, local_axis, local_ratio)

        return wrapper

    return decorator


def patches_grid_ood_single_feature(patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
                                    padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5,
                                    combiner: Type[PatchCombiner] = Average):
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            input_axis = resolve_deprecation(axis, x.ndim, patch_size, stride)
            local_size, local_stride = broadcast_to_axis(input_axis, patch_size, stride)

            if valid:
                shape = extract(x.shape, input_axis)
                padded_shape = np.maximum(shape, local_size)
                new_shape = padded_shape + (local_stride - padded_shape + local_size) % local_stride
                x = pad_to_shape(x, new_shape, input_axis, padding_values, ratio)

            patches = pmap(predict, divide(x, local_size, local_stride, input_axis), *args, **kwargs)

            pred_feat = combine(patches, extract(x.shape, input_axis), local_stride, axis, combiner=combiner)

            if valid:
                pred_feat = crop_to_shape(pred_feat, shape, axis, ratio)
            return prepend_dims(pred_feat, ndim=1)

        return wrapper

    return decorator
