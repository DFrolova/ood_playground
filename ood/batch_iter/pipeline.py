import numpy as np

from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box


SPATIAL_DIMS = (-3, -2, -1)


def center_choice_random(inputs, y_patch_size, random_state: np.random.RandomState):
    x, y = inputs
    center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS,
                                     random_state=random_state)
    return x, y, center


def sample_center_uniformly(shape, patch_size, spatial_dims, random_state):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size, random_state=random_state)
    else:
        return spatial_shape // 2


def extract_patch(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
    return x_patch, y_patch


def extract_patch_of_slices(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    x, y, center = inputs

    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = x[..., x_spatial_box[0][-1] : x_spatial_box[1][-1]]
    y_patch = y[..., y_spatial_box[0][-1] : y_spatial_box[1][-1]]
    return x_patch, y_patch


def center_choice(inputs, y_patch_size, random_state: np.random.RandomState, nonzero_fraction=0.5):
    x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    if len(centers) > 0 and random_state.uniform() < nonzero_fraction:
        sampled_id = random_state.choice(range(len(centers)), p=centers[:, -1])
        center = centers[sampled_id, :-1].astype(int)
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size, 
                                         spatial_dims=SPATIAL_DIMS, random_state=random_state)

    return x, y, center


def get_random_patch_of_slices(inputs, z_patch_size, random_state: np.random.RandomState):
    x, y = inputs
    z_min = random_state.randint(low=0, high=max(1, x.shape[-1] - z_patch_size))
    return x[..., z_min:z_min + z_patch_size], y[..., z_min:z_min + z_patch_size]


# ### 2D pipeline: ###


def get_random_slice(*arrays, interval: int = 1, random_state: np.random.RandomState = None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    slc = random_state.randint(arrays[0].shape[-1] // interval) * interval
    return tuple(array[..., slc] for array in arrays)


def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size, random_state: np.random.RandomState):
    center = sample_center_uniformly(shape=segm_slc.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS[-2:],
                                     random_state=random_state)
    x, y = extract_patch((image_slc, segm_slc, center),
                         x_patch_size=x_patch_size, y_patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS[-2:])
    return x, y
