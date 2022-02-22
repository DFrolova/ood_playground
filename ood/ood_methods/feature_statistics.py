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
    
    
def get_ood_scores_from_mean_std(test_ids, train_predictions_path, spectrum_folder, results_path, exist_ok=False):
    results = defaultdict(dict)
    
    train_ids = load(os.path.join(train_predictions_path, 'train_ids.json'))
    train_matrix = np.stack([load(os.path.join(train_predictions_path, spectrum_folder, f'{uid}.npy')) 
                             for uid in train_ids])
    
    for uid in test_ids:
        spectrum = load(os.path.join(spectrum_folder, f'{uid}.npy'))
        distances = np.linalg.norm(train_matrix - spectrum, axis=1)
        results['min_distance'][uid] = min(distances)
        results['5_percentile'][uid] = np.percentile(distances, 5)
        results['mean_distance'][uid] = np.mean(distances)
        results['distance_from_center'][uid] = np.linalg.norm((train_matrix.mean(axis=0) - spectrum))
    
    os.makedirs(results_path, exist_ok=exist_ok)
    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)