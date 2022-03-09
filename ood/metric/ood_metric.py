import os
import warnings

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dpipe.io import save_json
from dpipe.im.metrics import iou, dice_score


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


def calc_ood_scores(labels, is_ood_true, print_results=True):
    det_acc = detection_accuracy(is_ood_true, labels)
    roc_auc = roc_auc_score(is_ood_true, labels)
    tnr = tnr_at_95_tpr(is_ood_true, labels)
    if print_results:
        print(f'Detection accuracy: {det_acc:.4f}')
        print(f'AUROC: {roc_auc:.4f}')
        print(f'TNR @ 95% TPR: {tnr:.4f}')
    return det_acc, roc_auc, tnr


def get_maxprob_metric(y_true, prediction):
    # y_true is not used here, added just to have similar interface to other metrics
    uncertainty_result = 1 - np.maximum(prediction, 1 - prediction)
    return uncertainty_result.mean()


def get_entropy_metric(y_true, prediction, eps=1e-9):        
    # y_true is not used here, added just to have similar interface to other metrics
    return get_entropy(prediction, eps=eps).mean()

def get_entropy(prediction, eps=1e-9):        
    # y_true is not used here, added just to have similar interface to other metrics
    warnings.filterwarnings('ignore')
    uncertainty_result = - (prediction * np.log2(prediction + eps) + (1 - prediction) * \
                            np.log2(1 - prediction + eps))
    uncertainty_result[prediction == 0] = 0
    uncertainty_result[prediction == 1] = 0
    warnings.filterwarnings('default')
    return uncertainty_result

def get_mutual_info(ensemble_preds, mean_preds, eps=1e-9):
    entropies = get_entropy(ensemble_preds, eps=eps)
    return get_entropy(mean_preds, eps=eps) - np.mean(entropies, axis=0)

def get_inconsistency_metrics(ensemble_preds, top_n_voxels=500000, entr_eps=1e-9):
    labels = {}
    
    mean_preds = ensemble_preds.mean(axis=0)
    std_preds = ensemble_preds.std(axis=0).flatten()
    # take top uncertain pixels
    sorted_ids = np.argsort(mean_preds.flatten())
    std_preds_top = std_preds[sorted_ids][- 4 * top_n_voxels:]
    # get segmentation metrics
    labels['std'] = std_preds.mean()
    labels['var'] = np.square(std_preds).mean()
    ious = []
    for i in range(len(ensemble_preds)):
        ious.append(1 - iou(mean_preds > 0.5, ensemble_preds[i] > 0.5))
    labels['mean_iou'] = np.mean(ious)
    dices = []
    for i in range(len(ensemble_preds)):
        dices.append(1 - dice_score(mean_preds > 0.5, ensemble_preds[i] > 0.5))
    labels['mean_dice'] = np.mean(dices)
    volumes = np.array([(pred > 0.5).sum() for pred in ensemble_preds])
    mean_volume = volumes.mean()
    std_volume = volumes.std()
    labels['mean_volume'] = std_volume / mean_volume
    labels[f'top_n_{top_n_voxels}_std'] = std_preds_top[-top_n_voxels:].mean()
    labels[f'top_n_{2 * top_n_voxels}_std'] = std_preds_top[-2 * top_n_voxels:].mean()
    labels[f'top_n_{4 * top_n_voxels}_std'] = std_preds_top.mean()
    labels[f'top_n_{top_n_voxels}_var'] = np.square(std_preds_top[-top_n_voxels:]).mean()
    labels[f'top_n_{2 * top_n_voxels}_var'] = np.square(std_preds_top[-2 * top_n_voxels:]).mean()
    labels[f'top_n_{4 * top_n_voxels}_var'] = np.square(std_preds_top).mean()
    labels['mut_info'] = get_mutual_info(ensemble_preds, mean_preds, eps=entr_eps).mean()
    return labels