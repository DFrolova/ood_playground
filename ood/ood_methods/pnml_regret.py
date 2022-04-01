import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dpipe.io import load, save_json


def calculate_regret(x_image, predict_with_features_fn, projection_matrix, eps=1e-9):
    probs, test_features = predict_with_features_fn(x_image)
    test_features = test_features.reshape(test_features.shape[0], -1).astype(np.float32).T
    probs = probs.flatten().astype(np.float32)
            
    # Normalize
    norm = np.linalg.norm(test_features, axis=-1, keepdims=True)
    test_features /= norm
    
    # Calc projection
    x_proj = ((test_features @ projection_matrix) * test_features).sum(axis=1)
    xt_g = x_proj / (1 + x_proj)

    # Equation 20 
    regrets = probs / (eps + probs + (1 - probs) * (probs ** xt_g)) + \
              (1 - probs) / (eps + (1 - probs) + probs * ((1 - probs) ** xt_g))
    regrets = np.log(regrets) / np.log(2) # n_classes (as in github)

    return regrets.sum()


def get_pnml_regrets(load_x, test_ids, projection_matrix_path, predict_with_features_fn, 
                     results_path, exist_ok=False):
    results = defaultdict(dict)
    
    projection_matrix = load(os.path.join(projection_matrix_path, 'experiment_0/projection_matrix.npy'))
    
    for uid in tqdm(test_ids):
        x = load_x(uid)
        results['regret'][uid] = calculate_regret(x, predict_with_features_fn=predict_with_features_fn,
                                                 projection_matrix=projection_matrix)
    
    os.makedirs(results_path, exist_ok=exist_ok)
    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)