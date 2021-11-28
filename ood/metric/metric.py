import os
from collections import defaultdict
from typing import Sequence, Callable

import numpy as np
from tqdm import tqdm
from skimage.measure import label

from dpipe.commands import load_from_folder
from dpipe.io import save_json, load_pred
from dpipe.im.metrics import dice_score
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.itertools import zip_equal
from ood.utils import volume2diameter, get_pred


def aggregate_metric_probably_with_ids(xs, ys, ids, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    try:
        return aggregate_fn([metric(x, y, i) for x, y, i in zip_equal(xs, ys, ids)])
    except TypeError:
        return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def evaluate_with_ids(y_true: Sequence, y_pred: Sequence, ids: Sequence[str], metrics: dict) -> dict:
    return {name: metric(y_true, y_pred, ids) for name, metric in metrics.items()}


def compute_metrics_probably_with_ids(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str],
                                      metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)


def evaluate_individual_metrics_with_froc(load_y, metrics: dict,
                                         predictions_path, logits_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y(identifier)

        for metric_name, metric in metrics.items():
            if metric_name == 'froc_records':
                logit = load_pred(identifier, logits_path)
                results[metric_name][identifier] = metric(target, prediction, logit)
            else:
                try:
                    results[metric_name][identifier] = metric(target, prediction, identifier)
                except TypeError:
                    results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        
        
def evaluate_individual_metrics_with_froc_no_logits(load_y, load_x, metrics: dict, predict_logit,
                                         predictions_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y(identifier)

        for metric_name, metric in metrics.items():
            if metric_name == 'froc_records':
                logit = predict_logit(load_x(identifier))
                results[metric_name][identifier] = metric(target, prediction, logit)
            else:
                try:
                    results[metric_name][identifier] = metric(target, prediction, identifier)
                except TypeError:
                    results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        
        
def evaluate_individual_metrics_with_froc_no_pred(load_y, load_x, predict, predict_logit, metrics: dict, test_ids, 
                                                  results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier in tqdm(test_ids):
        target = load_y(identifier)
        prediction = predict(load_x(identifier))

        for metric_name, metric in metrics.items():
            if metric_name == 'froc_records':
                logit = predict_logit(load_x(identifier))
                results[metric_name][identifier] = metric(target, prediction, logit)
            else:
                try:
                    results[metric_name][identifier] = metric(target, prediction, identifier)
                except TypeError:
                    results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        
        
def evaluate_individual_metrics_with_froc_no_pred_with_crops(load_y, load_x, predict, predict_logit, metrics: dict, test_ids, 
                                                  results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier in tqdm(test_ids):
        target = load_y(identifier)
        prediction, center, crop_shape = predict(load_x(identifier))
        spatial_box = get_centered_box(center, crop_shape)
        target = crop_to_box(target, box=spatial_box, padding_values=np.min, axis=SPATIAL_DIMS)

        for metric_name, metric in metrics.items():
            if metric_name == 'froc_records':
                logit = predict_logit(load_x(identifier))
                results[metric_name][identifier] = metric(target, prediction, logit)
            else:
                try:
                    results[metric_name][identifier] = metric(target, prediction, identifier)
                except TypeError:
                    results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        

def evaluate_individual_metrics_probably_with_ids(load_y_true, metrics: dict, predictions_path, results_path,
                                                  exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][identifier] = metric(target, prediction, identifier)
            except TypeError:
                results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

        
def evaluate_individual_metrics_probably_with_ids_no_pred(load_y, load_x, predict, metrics: dict, test_ids,
                                                          results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        prediction = predict(load_x(_id))

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)

        
def get_intersection_stat_dice_id(cc_mask, one_cc, pred=None, logit=None):
    """Returns max local dice and corresponding stat to this hit component.
    If ``pred`` is ``None``, ``cc_mask`` treated as ground truth and stat sets to be 1."""
    hit_components = np.unique(cc_mask[one_cc])
    hit_components = hit_components[hit_components != 0]

    hit_stats = dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [[], [], [], []]))
    hit_dice, hit_id = [], []

    for n in hit_components:
        cc_mask_hit_one = cc_mask == n
        hit_dice.append(dice_score(cc_mask_hit_one, one_cc))
        hit_id.append(n)

        hit_stats['hit_max'].append(1. if pred is None else np.max(pred[cc_mask_hit_one]))
        hit_stats['hit_median'].append(1. if pred is None else np.median(pred[cc_mask_hit_one]))
        hit_stats['hit_q95'].append(1. if pred is None else np.percentile(pred[cc_mask_hit_one].astype(int), q=95))
        hit_stats['hit_logit'].append(np.inf if logit is None else np.max(logit[cc_mask_hit_one]))

    if len(hit_dice) == 0:
        return dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [0., 0., 0., -np.inf])), 0., None
    else:
        max_idx = np.argmax(hit_dice)
        hit_id = np.array(hit_id)[max_idx]
        hit_stats['hit_max'] = np.array(hit_stats['hit_max'])[max_idx]
        hit_stats['hit_median'] = np.array(hit_stats['hit_median'])[max_idx]
        hit_stats['hit_q95'] = np.array(hit_stats['hit_q95'])[max_idx]
        hit_stats['hit_logit'] = np.array(hit_stats['hit_logit'])[max_idx]
        return hit_stats, np.max(hit_dice), hit_id
    

def froc_records(segm, pred, logit):
    segm_split, segm_n_splits = label(get_pred(segm), return_num=True)
    pred_split, pred_n_splits = label(get_pred(pred), return_num=True)

    records = []

    for n in range(1, segm_n_splits + 1):
        record = {}
        segm_cc = segm_split == n

        record['obj'] = f'tum_{n}'
        record['is_tum'] = True
        record['diameter'] = volume2diameter(np.sum(segm_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=pred_split, one_cc=segm_cc,
                                                            pred=pred, logit=logit)
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'pred_{hit_id}'
        record['self_stat'] = 1.
        record['self_logit'] = np.inf

        records.append(record)

    for n in range(1, pred_n_splits + 1):
        record = {}
        pred_cc = pred_split == n

        record['obj'] = f'pred_{n}'
        record['is_tum'] = False
        record['diameter'] = volume2diameter(np.sum(pred_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=segm_split, one_cc=pred_cc)
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'tum_{hit_id}'
        record['self_stat'] = np.max(pred[pred_cc])
        record['self_logit'] = np.max(logit[pred_cc])

        records.append(record)

    return records