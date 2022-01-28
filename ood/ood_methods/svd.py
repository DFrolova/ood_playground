import os
from collections import defaultdict

import numpy as np

from dpipe.io import load, save_json


def get_spectrum(feature_map):
    feature_map = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)
    S = np.linalg.svd(feature_map, full_matrices=False, compute_uv=False)
    normalized_S = np.log(S)
    normalized_S /= np.linalg.norm(normalized_S)
    return normalized_S


def get_ood_scores_from_spectrum(predictions_path, results_path):
    results = defaultdict(dict)
    
    train_ids = load(os.path.join(predictions_path, 'train_ids.json'))
    train_matrix = np.stack([load(os.path.join(predictions_path, f'test_predictions/{uid}.npy')) 
                             for uid in train_ids])
    
    test_ids = load(os.path.join(predictions_path, 'test_ids.json'))
    for uid in test_ids:
        spectrum = load(os.path.join(predictions_path, f'test_predictions/{uid}.npy'))
        distances = np.linalg.norm(train_matrix - spectrum, axis=1)
        results['min_distance'][uid] = min(distances)
        results['mean_distance'][uid] = np.mean(distances)
        results['distance_from_center'][uid] = np.linalg.norm((train_matrix.mean(axis=0) - spectrum))
        
    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)