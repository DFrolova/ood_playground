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


def focal_tversky_loss(proba, target, spatial_dims, beta, gamma):
    intersection = torch.sum(proba * target, dim=spatial_dims)
    tp = torch.sum(proba ** 2 * target, dim=spatial_dims)
    fp = torch.sum(proba ** 2 * (1 - target), dim=spatial_dims)
    fn = torch.sum((1 - proba ** 2) * target, dim=spatial_dims)
    tversky_index = intersection / (tp + beta * fn + (1 - beta) * fp + 1)
    loss = (1 - tversky_index) ** gamma
    return loss.mean()


def focal_tversky_loss_with_nans(proba, target, spatial_dims, beta, gamma):
    proba = torch.where(~torch.isnan(target), proba, torch.tensor(0).to(proba))
    target = torch.nan_to_num(target)
    return focal_tversky_loss(proba, target, spatial_dims, beta, gamma)
