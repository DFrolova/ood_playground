import numpy as np

from ood.batch_iter.pipeline import SPATIAL_DIMS


def get_mean_std(feature):
    feature = feature.astype(np.float32)
    means = np.mean(feature, axis=SPATIAL_DIMS)
    stds = np.std(feature, axis=SPATIAL_DIMS)
    return np.concatenate((means, stds))

# from dpipe.im.shape_ops import zoom


# def get_mean_std(feature_map):

#     scale_factor = 50 / feature_map.shape[1]
#     feature_map = zoom(feature_map, [scale_factor, scale_factor, scale_factor, scale_factor], order=3)

#     feature_map = np.swapaxes(feature_map, 0, 1)
#     # feature_map = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)

#     means = np.mean(feature_map, axis=SPATIAL_DIMS)
#     stds = np.std(feature_map, axis=SPATIAL_DIMS)
#     return np.concatenate((means, stds))
