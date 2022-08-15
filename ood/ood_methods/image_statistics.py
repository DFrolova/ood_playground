import os
from tqdm import tqdm

import numpy as np

from dpipe.io import load, save_json, save, load_json
from dpipe.im.shape_ops import zoom
from ood.ood_methods.feature_based_methods import get_ood_scores_from_embedding


def get_image_spectrum(feature_map):
    feature_map = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)
    U, S, _ = np.linalg.svd(feature_map, full_matrices=False, compute_uv=True)
    normalized_S = np.log(S + 1e-9)
    normalized_S /= np.linalg.norm(normalized_S)
    return S, normalized_S, U[0]


def get_image_mean_std(feature):
    feature = feature.astype(np.float32)
    means = np.mean(feature, axis=(1, 2))
    stds = np.std(feature, axis=(1, 2))
    return means, np.concatenate((means, stds))


def get_all_scores_from_image(load_x, test_ids_embeddings, test_ids, train_predictions_path, results_path,
                              exist_ok=False):
    hist_bins = [100, 150, 200]
    scale_factors = [50, 100, 150, 200]

    if not os.path.exists(f'spectrum_{scale_factors[0]}'):
        for n_bins in hist_bins:
            os.makedirs(f'histograms_{n_bins}', exist_ok=exist_ok)

        for sc_fact in scale_factors:
            os.makedirs(f'spectrum_{sc_fact}', exist_ok=exist_ok)
            os.makedirs(f'normalized_spectrum_{sc_fact}', exist_ok=exist_ok)
            os.makedirs(f'sing_vector_{sc_fact}', exist_ok=exist_ok)
            os.makedirs(f'mean_{sc_fact}', exist_ok=exist_ok)
            os.makedirs(f'mean_std_{sc_fact}', exist_ok=exist_ok)

        for uid in tqdm(test_ids_embeddings):
            image = load_x(uid).astype(np.float32)

            for sc_fact in scale_factors:
                scale_factor = sc_fact / image.shape[0]
                new_image = zoom(image, [scale_factor, scale_factor, scale_factor], order=3)

                s, norm_s, u = get_image_spectrum(new_image)
                mean, mean_std = get_image_mean_std(new_image)

                save(s, f'spectrum_{sc_fact}/{uid}.npy')
                save(norm_s, f'normalized_spectrum_{sc_fact}/{uid}.npy')
                save(u, f'sing_vector_{sc_fact}/{uid}.npy')
                save(mean, f'mean_{sc_fact}/{uid}.npy')
                save(mean_std, f'mean_std_{sc_fact}/{uid}.npy')

            # compute histogram
            for n_bins in hist_bins:
                # scale to 0-1
                image -= np.min(image)
                image /= np.max(image)

                histogram, bin_edges = np.histogram(image, bins=n_bins, range=(0, 1), density=True)
                save(histogram, f'histograms_{n_bins}/{uid}.npy')

    all_methods = [f'histograms_{n_bins}' for n_bins in hist_bins]
    more_methods = [[f'spectrum_{sc_fact}', f'normalized_spectrum_{sc_fact}', f'mean_{sc_fact}', f'mean_std_{sc_fact}',
                     f'sing_vector_{sc_fact}'] for sc_fact in scale_factors]
    all_methods += [x for method_list in more_methods for x in method_list]

    for embedding_folder in tqdm(all_methods):
        for postfix in ['init', 'scale', 'norm']:
            get_ood_scores_from_embedding(test_ids=test_ids, train_predictions_path=train_predictions_path,
                                          spectrum_folder=embedding_folder, results_path=results_path,
                                          exist_ok=True, postfix=postfix)


def get_all_scores_from_image_augm(load_x_fns, full_uid_fns, test_ids, train_predictions_path, results_path,
                                   exist_ok=False):
    hist_bins = [100, 150, 200]

    test_ids_full = []
    if not os.path.exists(f'histograms_{hist_bins[0]}'):
        for n_bins in hist_bins:
            os.makedirs(f'histograms_{n_bins}', exist_ok=exist_ok)

        for uid_num, uid in enumerate(tqdm(test_ids)):
            for load_x, full_uid_fn in zip(load_x_fns, full_uid_fns):
                image = load_x(uid).astype(np.float32)
                full_uid = full_uid_fn(uid)
                test_ids_full.append(full_uid_fn(uid))

                # compute histogram
                for n_bins in hist_bins:
                    # scale to 0-1
                    image -= np.min(image)
                    image /= np.max(image)

                    histogram, bin_edges = np.histogram(image, bins=n_bins, range=(0, 1), density=True)
                    save(histogram, f'histograms_{n_bins}/{full_uid}.npy')

    all_methods = [f'histograms_{n_bins}' for n_bins in hist_bins]

    for embedding_folder in tqdm(all_methods):
        for postfix in ['init', 'scale', 'norm']:
            get_ood_scores_from_embedding(test_ids=test_ids_full, train_predictions_path=train_predictions_path,
                                          spectrum_folder=embedding_folder, results_path=results_path,
                                          exist_ok=True, postfix=postfix)

# def get_all_scores_from_image_augm(load_x, test_ids, train_predictions_path, results_path,
#                                    param_dict, transform_fns, exist_ok=False, ):
#     hist_bins = [100, 150, 200]
#
#     test_ids_full = []
#     if not os.path.exists(f'histograms_{hist_bins[0]}'):
#         for n_bins in hist_bins:
#             os.makedirs(f'histograms_{n_bins}', exist_ok=exist_ok)
#
#         for uid_num, uid in enumerate(tqdm(test_ids)):
#             image = load_x(uid).astype(np.float32)
#
#             for transform_id, transform_fn in enumerate(transform_fns):
#                 for cur_param in param_dict[transform_fn]:
#                     full_uid = '__'.join([str(cur_param), transform_fn.__name__, uid])
#                     test_ids_full.append(full_uid)
#                     # apply transform
#                     img_ood = transform_fn(image, param=cur_param, random_state=transform_id * len(test_ids) + uid_num)
#
#                     # compute histogram
#                     for n_bins in hist_bins:
#                         # scale to 0-1
#                         image -= np.min(image)
#                         image /= np.max(image)
#
#                         histogram, bin_edges = np.histogram(img_ood, bins=n_bins, range=(0, 1), density=True)
#                         save(histogram, f'histograms_{n_bins}/{full_uid}.npy')
#
#     all_methods = [f'histograms_{n_bins}' for n_bins in hist_bins]
#
#     for embedding_folder in tqdm(all_methods):
#         for postfix in ['init', 'scale', 'norm']:
#             get_ood_scores_from_embedding(test_ids=test_ids_full, train_predictions_path=train_predictions_path,
#                                           spectrum_folder=embedding_folder, results_path=results_path,
#                                           exist_ok=True, postfix=postfix)
