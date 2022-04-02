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


def get_ood_scores_from_spectrum(test_ids, train_predictions_path, spectrum_folder, results_path, exist_ok=False):
    results = defaultdict(dict)
    
    train_ids = load(os.path.join(train_predictions_path, 'train_ids.json'))
    train_matrix = np.stack([load(os.path.join(train_predictions_path, spectrum_folder, f'{uid}.npy')) 
                             for uid in train_ids])
    
    # scaled matrix
    max_vals = train_matrix.max(axis=0)
    min_vals = train_matrix.min(axis=0)
    train_matrix_scaled = (train_matrix - min_vals) / (max_vals - min_vals)
    
    # normalized matrix
    mean_vals = train_matrix.mean(axis=0)
    std_vals = train_matrix.std(axis=0)
    train_matrix_norm = (train_matrix - mean_vals) / std_vals
    
    for uid in test_ids:
        spectrum = load(os.path.join(spectrum_folder, f'{uid}.npy'))
        distances = np.linalg.norm(train_matrix - spectrum, axis=1)
        results['min_distance'][uid] = min(distances)
        results['5_percentile'][uid] = np.percentile(distances, 5)
        results['mean_distance'][uid] = np.mean(distances)
        results['distance_from_center'][uid] = np.linalg.norm((train_matrix.mean(axis=0) - spectrum))
        
        # scaled
        spectrum_scaled = (spectrum - mean_vals) / std_vals
        distances_scaled = np.linalg.norm(train_matrix_scaled - spectrum_scaled, axis=1)
        results['min_distance_scale'][uid] = min(distances_scaled)
        results['5_percentile_scale'][uid] = np.percentile(distances_scaled, 5)
        results['mean_distance_scale'][uid] = np.mean(distances_scaled)
        results['distance_from_center_scale'][uid] = np.linalg.norm((train_matrix_scaled.mean(axis=0) - spectrum_scaled))
        
        # normalized
        spectrum_norm = (spectrum - mean_vals) / std_vals
        distances_norm = np.linalg.norm(train_matrix_norm - spectrum_norm, axis=1)
        results['min_distance_norm'][uid] = min(distances_norm)
        results['5_percentile_norm'][uid] = np.percentile(distances_norm, 5)
        results['mean_distance_norm'][uid] = np.mean(distances_norm)
        results['distance_from_center_norm'][uid] = np.linalg.norm((train_matrix_norm.mean(axis=0) - spectrum_norm))
    
    os.makedirs(results_path, exist_ok=exist_ok)
    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)