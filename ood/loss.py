import numpy as np
import torch

from dpipe.torch.functional import dice_loss_with_logits, weighted_cross_entropy_with_logits


def combined_loss_weighted(pred: torch.Tensor, target: torch.Tensor, alpha=0.2, beta=0.7, adaptive_bce=False):
    return (1 - alpha) * dice_loss_with_logits(pred, target) \
           + alpha * weighted_cross_entropy_with_logits(pred, target, alpha=beta, adaptive=adaptive_bce)


def cosine_lr_schedule_fn(epoch, lr_max=1e-2, lr_min=1e-6, last_linear_epoch=10, n_epochs=100):
    if epoch <= last_linear_epoch:
        lr = lr_max * epoch / last_linear_epoch
    else:
        lr = lr_max * 0.5 * (1 + np.cos((epoch - last_linear_epoch) / (n_epochs - last_linear_epoch) * np.pi))
    return lr if lr > 0 else lr_min


def find_lr_schedule_fn(epoch, lr_max=1e-2, lr_min=1e-6, n_epochs=100):
    return lr_min * (lr_max / lr_min) ** (epoch / n_epochs)