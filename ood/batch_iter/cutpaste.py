from typing import Sequence, Callable

import numpy as np
from dpipe.batch_iter.sources import sample
from dpipe.im.shape_ops import crop_to_box
from dpipe.im.box import get_centered_box, box2slices
from skimage.filters import threshold_otsu
from skimage.measure import label

from perlin_numpy import generate_perlin_noise_3d  # https://github.com/pvigier/perlin-numpy
from .pipeline import sample_center_uniformly, SPATIAL_DIMS
from .augmentations import flips


def load_pair_by_random_ids(load_x: Callable, ids: Sequence, random_state: np.random.RandomState,
                            weights: Sequence[float] = None):

    for id0, id1 in zip(sample(ids, weights, random_state), sample(ids, weights, random_state)):
        yield load_x(id0), load_x(id1)


def generate_anomaly_mask(inputs, random_state: np.random.RandomState, max_anomaly_size: int = 128,
                          spatial_dims=SPATIAL_DIMS):
    x0, x1 = inputs
    x1 = flips((x1, ))[0]

    a = max_anomaly_size
    min_image_shape = np.min((x0.shape, x1.shape), axis=0)
    while np.any(min_image_shape <= a):
        a //= 2

    anomaly_box_shape = random_state.choice([a // 2, a], size=3, p=[0.33, 0.67])

    c0 = sample_center_uniformly(x0.shape, anomaly_box_shape, spatial_dims=spatial_dims, random_state=random_state)
    c1 = sample_center_uniformly(x1.shape, anomaly_box_shape, spatial_dims=spatial_dims, random_state=random_state)

    box0 = get_centered_box(c0, anomaly_box_shape)
    box1 = get_centered_box(c1, anomaly_box_shape)

    patch0 = crop_to_box(x0, box=box0, axis=spatial_dims)
    patch1 = crop_to_box(x1, box=box1, axis=spatial_dims)

    noise = generate_perlin_noise_3d(shape=anomaly_box_shape, res=random_state.choice([1, 2, 4, 8], size=3))
    mask = noise > np.percentile(noise, random_state.uniform(10, 90))

    beta = random_state.uniform(0.1, 1.0)
    anomaly = patch0 * (~mask) + beta * patch1 * mask + (1 - beta) * patch0 * mask

    # filter black-on-black mask elements:
    local_organ_mask_0 = crop_to_box(get_organ_mask(x0), box=box0, axis=spatial_dims)
    local_organ_mask_1 = crop_to_box(get_organ_mask(x1), box=box1, axis=spatial_dims)
    mask &= local_organ_mask_0 | local_organ_mask_1

    # creating full image and mask:
    x = x0.copy()
    y = np.zeros_like(x)

    slices0 = box2slices(box0)
    x[slices0] = anomaly
    y[slices0] = np.array(mask, dtype=y.dtype)

    return x, y


def get_organ_mask(img) -> np.ndarray:
    outer_mask = img < threshold_otsu(img)
    outer_mask[:1, :, :] = 1
    outer_mask[-1:, :, :] = 1
    outer_mask[:, :1, :] = 1
    outer_mask[:, -1:, :] = 1
    outer_mask = label(outer_mask, connectivity=3) == 1
    return ~outer_mask
