from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import Module

from dpipe.im.utils import identity
from dpipe.torch import sequence_to_var, to_np


def inference_step_ood(*inputs: np.ndarray, architecture: Module, activation: Callable = identity,
                       amp: bool = False) -> (np.ndarray, np.ndarray):
    architecture.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(amp or torch.is_autocast_enabled()):
            pred_segm, pred_feat = architecture(*sequence_to_var(*inputs, device=architecture), return_features=True)
            return to_np(activation(pred_segm)), to_np(pred_feat)
        
        
def inference_step_ood_lidc_last(*inputs: np.ndarray, architecture: Module, activation: Callable = identity,
                                 amp: bool = False) -> (np.ndarray):
    architecture.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(amp or torch.is_autocast_enabled()):
            pred_feat = architecture.forward_features(*sequence_to_var(*inputs, device=architecture))
            return to_np(pred_feat)
        
        
def get_resizing_features_modules(ndim: int, resize_features_to: str):
    if ndim not in (2, 3, ):
        raise ValueError(f'`ndim` should be in (2, 3). However, {ndim} is given.')

    ds16 = nn.AvgPool2d(16, 16, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(16, 16, ceil_mode=True)
    ds8 = nn.AvgPool2d(8, 8, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(8, 8, ceil_mode=True)
    ds4 = nn.AvgPool2d(4, 4, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(4, 4, ceil_mode=True)
    ds2 = nn.AvgPool2d(2, 2, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(2, 2, ceil_mode=True)
    identity = nn.Identity()
    us2 = nn.Upsample(scale_factor=2, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us4 = nn.Upsample(scale_factor=4, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us8 = nn.Upsample(scale_factor=8, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us16 = nn.Upsample(scale_factor=16, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)

    if resize_features_to == 'x16':
        return ds16, ds8, ds4, ds2, identity
    elif resize_features_to == 'x8':
        return ds8, ds4, ds2, identity, us2
    elif resize_features_to == 'x4':
        return ds4, ds2, identity, us2, us4
    elif resize_features_to == 'x2':
        return ds2, identity, us2, us4, us8
    elif resize_features_to == 'x1':
        return identity, us2, us4, us8, us16
    else:
        resize_features_to__options = ('x1', 'x2', 'x4', 'x8', 'x16')
        raise ValueError(f'`resize_features_to` should be in {resize_features_to__options}. '
                         f'However, {resize_features_to} is given.')
        
        
def enable_dropout(model: nn.Module):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
       
       
def inference_step_mc_dropout(*inputs: np.ndarray, architecture: nn.Module, activation: Callable = identity, 
                              amp: bool = False) -> np.ndarray:
    """
    Returns the prediction for the given ``inputs`` with all dropout layers turned to a train mode.
    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """
    architecture.eval()
    enable_dropout(architecture)
    with torch.no_grad():
        with torch.cuda.amp.autocast(amp or torch.is_autocast_enabled()):
            return to_np(activation(architecture(*sequence_to_var(*inputs, device=architecture))))