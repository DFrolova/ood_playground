import os
from collections import defaultdict
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dpipe.io import save_json
from dpipe.im.metrics import iou


def evaluate_individual_metrics_probably_with_ids_no_pred_mc_dropout(load_y, load_x, predict, predict_logit, 
                                                                     predict_with_dropout, test_ids, 
                                                                     results_path, agg_functions: dict, 
                                                                     segm_functions: dict, exist_ok=False):
    assert len(agg_functions) > 0, 'No aggregate functions provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        input_img = load_x(_id)
        deterministic_prediction = predict(input_img)
        predictions_list = predict_with_dropout(input_img)
        
        for agg_func_name, agg_func in segm_functions.items():
            if agg_func_name == 'froc_records':
                deterministic_logit = predict_logit(load_x(_id))
                results[agg_func_name][_id] = agg_func(target, deterministic_prediction, deterministic_logit)
            else:
                try:
                    results[agg_func_name][_id] = agg_func(target, deterministic_prediction, _id)
                except TypeError:
                    results[agg_func_name][_id] = agg_func(target, deterministic_prediction)

        for agg_func_name, agg_func in agg_functions.items():
            results[agg_func_name][_id] = agg_func(predictions_list)

    for agg_func_name, result in results.items():
        save_json(result, os.path.join(results_path, agg_func_name + '.json'), indent=0)


def detection_accuracy(y_true, y_pred):
    all_thresholds = [0.] + list(np.sort(np.unique(y_pred))) + [1.]
    detection_accuracies = []
    for threshold in all_thresholds:
        in_distr_acc = np.mean(y_pred[~y_true] <= threshold)
        ood_acc = np.mean(y_pred[y_true] > threshold)
        detection_accuracies.append(0.5 * (in_distr_acc + ood_acc))
    return max(detection_accuracies)


def tnr_at_95_tpr(y_true, y_pred):
    threshold = np.percentile(y_pred[~y_true], 95) 
    return (y_pred[y_true] > threshold).sum() / y_true.sum()


def calc_ood_scores(labels, is_ood_true):
    det_acc = detection_accuracy(is_ood_true, labels)
    roc_auc = roc_auc_score(is_ood_true, labels)
    tnr = tnr_at_95_tpr(is_ood_true, labels)
    print(f'Detection accuracy: {det_acc:.4f}')
    print(f'AUROC: {roc_auc:.4f}')
    print(f'TNR @ 95% TPR: {tnr:.4f}')
    return det_acc, roc_auc, tnr


def get_top_n_labels_var(prediction_list, n):
    ensemble_preds = []
    for preds in prediction_list:
        sorted_preds = np.sort(preds)
        ensemble_preds.append(sorted_preds.flatten()[-n:])
    
    ensemble_preds = np.array(ensemble_preds)
    var_preds = ensemble_preds.var(axis=0)
    label = var_preds.mean()
    return label


def get_all_labels_var(prediction_list):
    ensemble_preds = []
    for preds in prediction_list:
        ensemble_preds.append(preds.flatten())
    
    ensemble_preds = np.array(ensemble_preds)
    var_preds = ensemble_preds.var(axis=0)
    label = var_preds.mean()
    return label


def get_all_labels_std(prediction_list):
    ensemble_preds = []
    for preds in prediction_list:
        ensemble_preds.append(preds.flatten())
    
    ensemble_preds = np.array(ensemble_preds)
    var_preds = ensemble_preds.std(axis=0)
    label = var_preds.mean()
    return label


def get_mean_iou(prediction_list):
    ensemble_preds = []
    for preds in prediction_list:
        ensemble_preds.append(preds.flatten())
    
    ensemble_preds = np.array(ensemble_preds)
    mean_preds = ensemble_preds.mean(axis=0)
    ious = [1 - iou(mean_preds > 0.5, pred > 0.5) for pred in ensemble_preds]
    label = np.mean(ious)
    return label


def get_mean_volume(prediction_list):
    ensemble_preds = []
    for preds in prediction_list:
        ensemble_preds.append(preds.flatten())
    
    ensemble_preds = np.array(ensemble_preds)
    volumes = np.array([(pred > 0.5).sum() for pred in ensemble_preds])
    mean_volume = volumes.mean()
    std_volume = volumes.std()
    label = std_volume / mean_volume
    return label


def get_maxprob_score(y_true, prediction):
    # y_true is not used here, added just to have similar interface to other metrics
    uncertainty_result = 1 - np.maximum(prediction, 1 - prediction)
    return uncertainty_result.mean()


def get_entropy(y_true, prediction, eps=1e-9):        
    # y_true is not used here, added just to have similar interface to other metrics
    warnings.filterwarnings('ignore')
    uncertainty_result = - (prediction * np.log2(prediction + eps) + (1 - prediction) * np.log2(1 - prediction + eps))
    uncertainty_result[prediction == 0] = 0
    uncertainty_result[prediction == 1] = 0
    warnings.filterwarnings('default')
    return uncertainty_result.mean()