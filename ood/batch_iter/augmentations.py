import numpy as np

from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.im import zoom_to_shape
from scipy.ndimage import rotate

from .pipeline import sample_center_uniformly, SPATIAL_DIMS


def apply_augm_pipeline(inputs, pipeline, p, random_state: np.random.RandomState):
    outputs = inputs
    if random_state.rand() < p:
        augm = pipeline(image=inputs[0], mask=inputs[1])
        outputs = (augm['image'], augm['mask'])
    return outputs


def augm_pipeline_3d_tumor_sampling(inputs, x_patch_size, p, random_state: np.random.RandomState):
    outputs = inputs
    if random_state.rand() < p:
        outputs = shift_center(outputs, max_shift=x_patch_size//2, random_state=random_state)
    return outputs


def shift_center(inputs, max_shift, random_state: np.random.RandomState):
    x, y, center = inputs
    spatial_shape = np.array(x.shape)[list(SPATIAL_DIMS)]
    low = np.maximum(max_shift, center - max_shift)
    high = np.minimum(spatial_shape - max_shift, center + max_shift + 1)
    new_center = random_state.randint(low=low, high=high, size=len(SPATIAL_DIMS))
    return x, y, new_center


def augm_pipeline_3d(inputs, shape, crop_shape, angle_limit, p, random_state: np.random.RandomState):
    outputs = inputs
    if random_state.rand() < p:
        outputs = random_sized_crop_3d(outputs, shape, crop_shape, random_state)
        outputs = rotate_3d(outputs, angle_limit, random_state)
    return outputs


def random_sized_crop_3d(inputs, shape, crop_shape, random_state):
    outputs = []
    crop_shape = np.array([random_state.randint(int(crop_shape[0]), shape[0]), ] * len(SPATIAL_DIMS))
    center = sample_center_uniformly(shape, crop_shape, spatial_dims=SPATIAL_DIMS)
    for output in inputs:
        spatial_box = get_centered_box(center, crop_shape)
        output = crop_to_box(output, box=spatial_box, padding_values=np.min, axis=SPATIAL_DIMS)
        output = zoom_to_shape(output, shape=shape, axis=SPATIAL_DIMS, order=3)
        outputs.append(output)
    return tuple(outputs)


def rotate_3d(inputs, limit, random_state):
    angle = random_state.uniform(low=-limit, high=limit)
    outputs = []
    for output in inputs:
        output = rotate(output, angle, mode='nearest', axes=SPATIAL_DIMS[:2], reshape=False)
        outputs.append(output)
    return tuple(outputs)