from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.io import save_json, load
from .pipeline import sample_center_uniformly, SPATIAL_DIMS


def get_spatial_boxes_cc359(test_ids, n_repeats, x_patch_size, pred_patch_stride, load_x, random_state, high_eps=50,
                            results_path='spatial_boxes.json'):
    low_eps = x_patch_size[2]
    pred_patch_stride = pred_patch_stride[2]

    results = defaultdict(dict)
    for identifier in tqdm(test_ids):

        image = load_x(identifier)
        high_lim = image.shape[SPATIAL_DIMS[2]] - high_eps

        for i in range(n_repeats):
            crop_shape = np.array([image.shape[SPATIAL_DIMS[0]], image.shape[SPATIAL_DIMS[1]],
                                   random_state.randint(low_eps // pred_patch_stride,
                                                        high_lim // pred_patch_stride + 1) * pred_patch_stride])
            center = sample_center_uniformly(image.shape, crop_shape, spatial_dims=SPATIAL_DIMS,
                                             random_state=random_state)
            spatial_box = get_centered_box(center, crop_shape)

            results[identifier + '_' + str(i)] = spatial_box

    save_json(results, results_path, indent=0)


def get_spatial_boxes(test_ids, n_repeats, x_patch_size, pred_patch_stride, load_x, random_state, high_eps=50,
                      results_path='spatial_boxes.json'):
    low_eps = x_patch_size
    pred_patch_stride = pred_patch_stride

    results = defaultdict(dict)
    for identifier in tqdm(test_ids):

        image = load_x(identifier)
        high_lim = image.shape[SPATIAL_DIMS[2]] - high_eps

        for i in range(n_repeats):
            crop_shape = np.array([image.shape[SPATIAL_DIMS[0]], image.shape[SPATIAL_DIMS[1]],
                                   random_state.randint(low_eps // pred_patch_stride,
                                                        high_lim // pred_patch_stride + 1) * pred_patch_stride])
            center = sample_center_uniformly(image.shape, crop_shape, spatial_dims=SPATIAL_DIMS,
                                             random_state=random_state)
            spatial_box = get_centered_box(center, crop_shape)

            results[identifier + '_' + str(i)] = spatial_box

    save_json(results, results_path, indent=0)


def load_cropped_image(uid, load_x, spatial_boxes_path):
    spatial_boxes = load(spatial_boxes_path)
    spatial_box = spatial_boxes[uid]
    image = load_x(uid.split('_')[0])
    image_cropped = crop_to_box(image, box=spatial_box, padding_values=np.min, axis=SPATIAL_DIMS)
    return image_cropped


def get_padded_prediction(prediction, target_full, uid, spatial_box):
    prediction_full = np.zeros_like(target_full)
    prediction_full[..., spatial_box[0][-1]: spatial_box[1][-1]] = prediction
    return prediction_full
