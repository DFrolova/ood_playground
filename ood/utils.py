import os
import random
from pathlib import Path

import numpy as np
import torch

import surface_distance.metrics as sdsc
from dpipe.io import PathLike


def get_pred(x, threshold=0.5):
    return x > threshold


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def get_lib_root_path():
    return Path('/'.join(os.path.abspath(__file__).split('/')[:-2]))


def sdice(a, b, spacing, tolerance):
    surface_distances = sdsc.compute_surface_distances(a, b, spacing)
    return sdsc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)
    
    
def skip_calculating_metrics(**args):
    print('>>> Passing the step of calculating test metrics', flush=True)
    os.makedirs('test_metrics')
    

def volume2diameter(volume):
    return (6 * volume / np.pi) ** (1 / 3)