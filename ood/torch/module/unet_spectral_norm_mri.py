import warnings
from functools import partial

import numpy as np
import torch
from torch import nn

from dpipe import layers
from dpipe.im.utils import identity
from dpipe.im.shape_ops import crop_to_shape
from ood.torch.model import get_resizing_features_modules
from .spectral_batchnorm import SpectralBatchNorm3d
from .spectral_norm_conv import spectral_norm_conv
from .spectral_norm_fc import spectral_norm_fc


class PreActivationSpectralNorm(nn.Module):
    """
    Runs a sequence of batch_norm, activation, and spectral normalized convolutional ``layer``.
        in -> (BN -> activation -> layer) -> out
    Parameters
    ----------
    in_features: int
        the number of incoming features/channels.
    out_features: int
        the number of the output features/channels.
    batch_norm_module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    activation_module
        module to build up activation layer. Default is ``torch.nn.ReLU``.
    layer_module: Callable(in_features, out_features, **kwargs)
        module to build up the main layer, e.g. ``torch.nn.Conv3d`` or ``torch.nn.Linear``.
    kwargs
        additional arguments passed to ``layer_module``.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 layer_module, batch_norm_module=None, activation_module=nn.ReLU,
                 input_size, n_power_iterations=1, coeff=3, **kwargs):
        super().__init__()
        if batch_norm_module is not None:
            self.bn = batch_norm_module(in_features)
        else:
            self.bn = identity
        self.activation = activation_module()
        self.layer = layer_module(in_features, out_features, **kwargs)

        kernel_size = kwargs['kernel_size']

        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            self.layer = spectral_norm_fc(self.layer, coeff, n_power_iterations)
        else:
            # Otherwise use spectral norm conv, with loose bound
            input_dim = (in_features, input_size, input_size, input_size)
            self.layer = spectral_norm_conv(
                self.layer, coeff, input_dim, n_power_iterations
            )

    def forward(self, x):
        return self.layer(self.activation(self.bn(x)))


class ResBlockSpectralNorm(nn.Module):
    """
    Performs a sequence of two spectral normalized convolutions with residual connection (Residual Block).
    ..
        in ---> (BN --> activation --> Conv) --> (BN --> activation --> Conv) -- + --> out
            |                                                                    ^
            |                                                                    |
             --------------------------------------------------------------------
    Parameters
    ----------
    in_channels: int
        the number of incoming channels.
    out_channels: int
        the number of the `ResBlock` output channels.
        Note, if ``in_channels`` != ``out_channels``, then linear transform will be applied to the shortcut.
    kernel_size: int, tuple
        size of the convolving kernel.
    stride: int, tuple, optional
        stride of the convolution. Default is 1.
        Note, if stride is greater than 1, then linear transform will be applied to the shortcut.
    padding: int, tuple, optional
        zero-padding added to all spatial sides of the input. Default is 0.
    dilation: int, tuple, optional
        spacing between kernel elements. Default is 1.
    bias: bool
        if ``True``, adds a learnable bias to the output. Default is ``False``.
    activation_module: None, nn.Module, optional
        module to build up activation layer.  Default is ``torch.nn.ReLU``.
    conv_module: nn.Module
        module to build up convolution layer with given parameters, e.g. ``torch.nn.Conv3d``.
    batch_norm_module: nn.Module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    kwargs
        additional arguments passed to ``conv_module``.
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 activation_module=nn.ReLU, conv_module, batch_norm_module, input_size, coeff, n_power_iterations,
                 **kwargs):
        super().__init__()
        # ### Features path ###
        pre_activation = partial(
            PreActivationSpectralNorm, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, layer_module=conv_module, batch_norm_module=batch_norm_module,
            input_size=input_size, coeff=coeff, n_power_iterations=n_power_iterations, **kwargs
        )

        self.conv_path = nn.Sequential(pre_activation(in_channels, out_channels, stride=stride, bias=False),
                                       pre_activation(out_channels, out_channels, bias=bias))

        # ### Shortcut ###
        spatial_difference = np.floor(
            np.asarray(dilation) * (np.asarray(kernel_size) - 1) - 2 * np.asarray(padding)
        ).astype(int)
        if not (spatial_difference >= 0).all():
            raise ValueError(f"The output's shape cannot be greater than the input's shape. ({spatial_difference})")

        if in_channels != out_channels or stride != 1:
            self.adjust_to_stride = conv_module(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
            self.adjust_to_stride = spectral_norm_fc(self.adjust_to_stride, coeff=coeff,
                                                     n_power_iterations=n_power_iterations)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        x_conv = self.conv_path(x)
        shape = x_conv.shape[2:]
        axes = range(-len(shape), 0)
        x_skip = crop_to_shape(self.adjust_to_stride(x), shape=shape, axis=axes)
        return x_conv + x_skip


class UNetSpectralNorm(nn.Module):
    def __init__(self, ndim: int = 3, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16,
                 return_features_from: tuple = (3,), x_patch_size: int = 80, coeff: float = 3.,
                 n_power_iterations: int = 1):
        super().__init__()
        if ndim not in (2, 3,):
            raise ValueError(f'`ndim` should be in (2, 3). However, {ndim} is given.')
        self.ndim = ndim

        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out
        self.n_features = n_filters_init

        self.return_features_from = return_features_from

        print(f'Features will be returned from {return_features_from}', flush=True)

        n = n_filters_init

        filters = (
        n, n, 2 * n, 2 * n, 4 * n, 4 * n, 8 * n, 8 * n, 16 * n, 16 * n, 8 * n, 8 * n, 4 * n, 4 * n, 2 * n, 2 * n, n, n,
        n, n_chans_out)
        scales = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 1)
        self.layer2filters = {i: f for i, f in enumerate(filters, start=1)}
        self.layer2scale = {i: d for i, d in enumerate(scales, start=1)}

        self.s1, self.s2, self.s4, self.s8, self.s16 = get_resizing_features_modules(ndim, 'x1')
        self.scale2module = {d: m for d, m in zip((1, 2, 4, 8, 16), (self.s1, self.s2, self.s4, self.s8, self.s16))}

        resblock = ResBlockSpectralNorm
        downsample = nn.MaxPool2d if (ndim == 2) else nn.MaxPool3d
        upsample = partial(nn.Upsample, mode='bilinear' if (ndim == 2) else 'trilinear')

        resblock_kwargs = {'kernel_size': 3, 'padding': 1, 'coeff': coeff, 'n_power_iterations': n_power_iterations,
                           'conv_module': nn.Conv3d, 'batch_norm_module': partial(SpectralBatchNorm3d, coeff=coeff)}

        self.init_path = nn.Sequential(
            resblock(n_chans_in, n, input_size=x_patch_size, **resblock_kwargs),  # 1
            resblock(n, n, input_size=x_patch_size, **resblock_kwargs),  # 2
        )

        self.down1 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n, n * 2, input_size=x_patch_size // 2, **resblock_kwargs),  # 3
            resblock(n * 2, n * 2, input_size=x_patch_size // 2, **resblock_kwargs)  # 4
        )

        self.down2 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 2, n * 4, input_size=x_patch_size // 4, **resblock_kwargs),  # 5
            resblock(n * 4, n * 4, input_size=x_patch_size // 4, **resblock_kwargs)  # 6
        )

        self.down3 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 4, n * 8, input_size=x_patch_size // 8, **resblock_kwargs),  # 7
            resblock(n * 8, n * 8, input_size=x_patch_size // 8, **resblock_kwargs)  # 8
        )

        self.bottleneck = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 8, n * 16, input_size=x_patch_size // 16, **resblock_kwargs),  # 9
            resblock(n * 16, n * 16, input_size=x_patch_size // 16, **resblock_kwargs),  # 10
            resblock(n * 16, n * 8, input_size=x_patch_size // 16, **resblock_kwargs),  # 11
            upsample(scale_factor=2, align_corners=True),
        )

        self.up3 = nn.Sequential(
            resblock(n * 8, n * 8, input_size=x_patch_size // 8, **resblock_kwargs),  # 12
            resblock(n * 8, n * 4, input_size=x_patch_size // 8, **resblock_kwargs),  # 13
            upsample(scale_factor=2, align_corners=True),
        )

        self.up2 = nn.Sequential(
            resblock(n * 4, n * 4, input_size=x_patch_size // 4, **resblock_kwargs),  # 14
            resblock(n * 4, n * 2, input_size=x_patch_size // 4, **resblock_kwargs),  # 15
            upsample(scale_factor=2, align_corners=True),
        )

        self.up1 = nn.Sequential(
            resblock(n * 2, n * 2, input_size=x_patch_size // 2, **resblock_kwargs),  # 16
            resblock(n * 2, n, input_size=x_patch_size // 2, **resblock_kwargs),  # 17
            upsample(scale_factor=2, align_corners=True),
        )

        self.out_path = nn.Sequential(
            resblock(n, n, input_size=x_patch_size, **resblock_kwargs),  # 18
            resblock(n, n, input_size=x_patch_size, **resblock_kwargs),  # 19
            # TODO replace with gaussian process???
            resblock(n, n_chans_out, input_size=x_patch_size, kernel_size=1, padding=0,
                     bias=True, conv_module=resblock_kwargs['conv_module'],
                     batch_norm_module=resblock_kwargs['batch_norm_module'],
                     coeff=resblock_kwargs['coeff'],
                     n_power_iterations=resblock_kwargs['n_power_iterations']),  # 20
        )

    @staticmethod
    def forward_block(x, block):
        outputs = []
        for layer in block:
            x = layer(x)
            if isinstance(layer, ResBlockSpectralNorm):
                outputs.append(x)
        return outputs, x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if return_features:
            warnings.filterwarnings('ignore')
            xs_init, x_init = self.forward_block(x, self.init_path)
            xs_down1, x_down1 = self.forward_block(x_init, self.down1)
            xs_down2, x_down2 = self.forward_block(x_down1, self.down2)
            xs_down3, x_down3 = self.forward_block(x_down2, self.down3)
            xs_bottleneck, x_bottleneck = self.forward_block(x_down3, self.bottleneck)
            xs_up3, x_up3 = self.forward_block(x_bottleneck + x_down3, self.up3)
            xs_up2, x_up2 = self.forward_block(x_up3 + x_down2, self.up2)
            xs_up1, x_up1 = self.forward_block(x_up2 + x_down1, self.up1)
            xs_out, x_out = self.forward_block(x_up1 + x_init, self.out_path)
            warnings.filterwarnings('default')

            xs = [_x for _xs in (xs_init, xs_down1, xs_down2, xs_down3, xs_bottleneck, xs_up3, xs_up2, xs_up1, xs_out)
                  for _x in _xs]
            return_features = torch.cat([self.scale2module[self.layer2scale[l]](xs[l - 1])
                                         for l in self.return_features_from], dim=1)

            return x_out, return_features

        else:
            warnings.filterwarnings('ignore')
            x0 = self.init_path(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x3_up = self.up3(self.bottleneck(x3) + x3)
            x2_up = self.up2(x3_up + x2)
            x1_up = self.up1(x2_up + x1)
            x_out = self.out_path(x1_up + x0)
            warnings.filterwarnings('default')

            return x_out


class UNetSpectralNormFeatureExtractor(UNetSpectralNorm):

    def forward(self, x: torch.Tensor):
        return super().forward(x, return_features=True)[1]
