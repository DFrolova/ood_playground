import os
from tqdm import tqdm
from collections import defaultdict

import numpy as np

from dpipe.io import load, save_json, save, load_json
from ood.ood_methods.svd import get_singular_vectors_and_values
from ood.ood_methods.feature_statistics import get_mean_std


def get_ood_scores_from_embedding(test_ids, train_predictions_path, spectrum_folder,
                                  results_path, postfix='init', exist_ok=False):
    results = defaultdict(dict)

    train_ids = load(os.path.join(train_predictions_path, 'train_ids.json'))
    train_matrix = np.stack([load(os.path.join(train_predictions_path, spectrum_folder, f'{uid}.npy'))
                             for uid in train_ids])

    if postfix == 'scale':
        # scaled matrix
        max_vals = train_matrix.max(axis=0)
        min_vals = train_matrix.min(axis=0)
        train_matrix = (train_matrix - min_vals) / (max_vals - min_vals)
    elif postfix == 'norm':
        # normalized matrix
        mean_vals = train_matrix.mean(axis=0)
        std_vals = train_matrix.std(axis=0)
        train_matrix = (train_matrix - mean_vals) / std_vals

    mean_mahalanobis = train_matrix.mean(axis=0)

    cov_mahal = np.zeros((train_matrix.shape[1], train_matrix.shape[1]))
    for train_sample in train_matrix:
        cov_mahal += np.outer(train_sample - mean_mahalanobis, train_sample - mean_mahalanobis)

    cov_mahal /= len(train_matrix)
    inv_covariance_mahalanobis = np.linalg.inv(cov_mahal)

    for uid in test_ids:
        spectrum = load(os.path.join(spectrum_folder, f'{uid}.npy'))

        if postfix == 'scale':
            # scaled matrix
            spectrum = (spectrum - min_vals) / (max_vals - min_vals)
        elif postfix == 'norm':
            # normalized matrix
            spectrum = (spectrum - mean_vals) / std_vals

        distances = np.linalg.norm(train_matrix - spectrum, axis=1)
        results[f'min_distance_{postfix}'][uid] = min(distances)
        results[f'5_percentile_{postfix}'][uid] = np.percentile(distances, 5)
        results[f'mahalanobis_{postfix}'][uid] = (spectrum - mean_mahalanobis) @ inv_covariance_mahalanobis @ \
                                                 (spectrum - mean_mahalanobis)

    os.makedirs(results_path, exist_ok=exist_ok)
    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, spectrum_folder + '__' + metric_name + '.json'), indent=0)


def get_all_scores_from_features(predict_fn, load_x, test_ids, train_predictions_path, results_path,
                                 exist_ok=False):

    if not os.path.exists('spectrum'):
        for i in range(1, 6):
            os.makedirs(f'sing_vector_{i}', exist_ok=exist_ok)

        os.makedirs('spectrum', exist_ok=exist_ok)
        os.makedirs('normalized_spectrum', exist_ok=exist_ok)
        os.makedirs('mean', exist_ok=exist_ok)
        os.makedirs('mean_std', exist_ok=exist_ok)

        for uid in tqdm(test_ids):
            feature_map = predict_fn(load_x(uid))
            u, s, s_norm = get_singular_vectors_and_values(feature_map)
            mean_std_embed = get_mean_std(feature_map)

            for i in range(1, 6):
                save(u[:i].flatten(), f'sing_vector_{i}/{uid}.npy')

            save(s, f'spectrum/{uid}.npy')
            save(s_norm, f'normalized_spectrum/{uid}.npy')
            save(mean_std_embed[:int(len(mean_std_embed) // 2)], f'mean/{uid}.npy')
            save(mean_std_embed, f'mean_std/{uid}.npy')

    for embedding_folder in tqdm(['spectrum', 'normalized_spectrum', 'mean', 'mean_std'] + [f'sing_vector_{i}' for i in
                                                                                            range(1, 6)]):
        for postfix in ['init', 'scale', 'norm']:
            get_ood_scores_from_embedding(test_ids=test_ids, train_predictions_path=train_predictions_path,
                                          spectrum_folder=embedding_folder, results_path=results_path,
                                          exist_ok=True, postfix=postfix)
