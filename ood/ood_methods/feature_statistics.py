import os
from collections import defaultdict

import numpy as np

from dpipe.io import load, save_json, save
from ood.batch_iter.pipeline import SPATIAL_DIMS


def get_mean_std(feature):
    feature = feature.astype(np.float32)
    means = np.mean(feature, axis=SPATIAL_DIMS)
    stds = np.std(feature, axis=SPATIAL_DIMS)
    return np.concatenate((means, stds))