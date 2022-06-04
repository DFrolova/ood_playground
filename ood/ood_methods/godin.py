import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dpipe.io import save_json, load
from dpipe.im.shape_ops import crop_to_box
from ood.batch_iter.pipeline import SPATIAL_DIMS


def evaluate_individual_metrics_godin(load_y, load_x, predict, metrics: dict, test_ids,
                                      results_path, predict_logit=None, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        image = load_x(_id)
        prediction, scores = predict(image)
        
        results['godin'][_id] = scores.mean()

        for metric_name, metric in metrics.items():
            if metric_name == 'froc_records':
                logit = predict_logit(image)
                results[metric_name][_id] = metric(target, prediction, logit)
            else:
                try:
                    results[metric_name][_id] = metric(target, prediction, _id)
                except TypeError:
                    results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
        
        
def evaluate_individual_metrics_godin_with_crops(load_x, load_y, predict, predict_logit, test_ids, n_repeats,
                                                 metrics: dict, spatial_boxes_path, results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)
    
    spatial_boxes = load(spatial_boxes_path)

    results = defaultdict(dict)
    
    for _id in tqdm(test_ids):
        image = load_x(_id)
        target = load_y(_id)
        
        for i in range(n_repeats):
            spatial_box = spatial_boxes[_id + '_' + str(i)]
            image_cropped = crop_to_box(image, box=spatial_box, padding_values=np.min, axis=SPATIAL_DIMS)
            target_cropped = crop_to_box(target, box=spatial_box, padding_values=np.min, axis=SPATIAL_DIMS)
            
            prediction, scores = predict(image_cropped)
            results['godin'][_id + '_' + str(i)] = scores.mean()
            
            for metric_name, metric in metrics.items():
                if metric_name == 'froc_records':
                    logit = predict_logit(image_cropped)
                    results[metric_name][_id + '_' + str(i)] = metric(target_cropped, prediction, logit)
                else:
                    try:
                        results[metric_name][_id + '_' + str(i)] = metric(target_cropped, prediction, _id)
                    except TypeError:
                        results[metric_name][_id + '_' + str(i)] = metric(target_cropped, prediction)
                        
            for metric_name, metric in metrics.items():
                if metric_name == 'froc_records':
                    logit = predict_logit(image_cropped)
                    results[f'{metric_name}_padded'][_id + '_' + str(i)] = metric(target_cropped, prediction, logit)
                elif metric_name == 'dice_score' or metric_name == 'sdice_score':
                    try:
                        results[f'{metric_name}_padded'][_id + '_' + str(i)] = metric(target_cropped, prediction, _id)
                    except TypeError:
                        results[f'{metric_name}_padded'][_id + '_' + str(i)] = metric(target_cropped, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)