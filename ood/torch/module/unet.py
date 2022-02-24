import warnings
from functools import partial

import numpy as np
import torch
from torch import nn

from dpipe import layers
from dpipe.layers.resblock import ResBlock, ResBlock2d, ResBlock3d
from ood.torch.model import get_resizing_features_modules
from dpipe.layers import PreActivationND
from dpipe.im.utils import identity
from dpipe.im.shape_ops import crop_to_shape
from dpipe.itertools import zip_equal


class UNet(nn.Module):
    def __init__(self, ndim: int = 3, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16,
                 return_features_from: tuple = (3, )):
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

        filters = (n, n, 2*n, 2*n, 4*n, 4*n, 8*n, 8*n, 16*n, 16*n, 8*n, 8*n, 4*n, 4*n, 2*n, 2*n, n, n, n, n_chans_out)
        scales = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 1)
        self.layer2filters = {i: f for i, f in enumerate(filters, start=1)}
        self.layer2scale = {i: d for i, d in enumerate(scales, start=1)}

        self.s1, self.s2, self.s4, self.s8, self.s16 = get_resizing_features_modules(ndim, 'x1')
        self.scale2module = {d: m for d, m in zip((1, 2, 4, 8, 16), (self.s1, self.s2, self.s4, self.s8, self.s16))}

        resblock = ResBlock2d if (ndim == 2) else ResBlock3d
        downsample = nn.MaxPool2d if (ndim == 2) else nn.MaxPool3d
        upsample = partial(nn.Upsample, mode='bilinear' if (ndim == 2) else 'trilinear')

        self.init_path = nn.Sequential(
            resblock(n_chans_in, n, kernel_size=3, padding=1),                  # 1
            resblock(n, n, kernel_size=3, padding=1),                           # 2
        )

        self.down1 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n, n * 2, kernel_size=3, padding=1),                       # 3
            resblock(n * 2, n * 2, kernel_size=3, padding=1)                    # 4
        )

        self.down2 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 2, n * 4, kernel_size=3, padding=1),                   # 5
            resblock(n * 4, n * 4, kernel_size=3, padding=1)                    # 6
        )

        self.down3 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 4, n * 8, kernel_size=3, padding=1),                   # 7
            resblock(n * 8, n * 8, kernel_size=3, padding=1)                    # 8
        )

        self.bottleneck = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 8, n * 16, kernel_size=3, padding=1),                  # 9
            resblock(n * 16, n * 16, kernel_size=3, padding=1),                 # 10
            resblock(n * 16, n * 8, kernel_size=3, padding=1),                  # 11
            upsample(scale_factor=2, align_corners=True),
        )

        self.up3 = nn.Sequential(
            resblock(n * 8, n * 8, kernel_size=3, padding=1),                   # 12
            resblock(n * 8, n * 4, kernel_size=3, padding=1),                   # 13
            upsample(scale_factor=2, align_corners=True),
        )

        self.up2 = nn.Sequential(
            resblock(n * 4, n * 4, kernel_size=3, padding=1),                   # 14
            resblock(n * 4, n * 2, kernel_size=3, padding=1),                   # 15
            upsample(scale_factor=2, align_corners=True),
        )

        self.up1 = nn.Sequential(
            resblock(n * 2, n * 2, kernel_size=3, padding=1),                   # 16
            resblock(n * 2, n, kernel_size=3, padding=1),                       # 17
            upsample(scale_factor=2, align_corners=True),
        )

        self.out_path = nn.Sequential(
            resblock(n, n, kernel_size=3, padding=1),                           # 18
            resblock(n, n, kernel_size=3, padding=1),                           # 19
            resblock(n, n_chans_out, kernel_size=1, padding=0, bias=True),      # 20
        )

    @staticmethod
    def forward_block(x, block):
        outputs = []
        for layer in block:
            x = layer(x)
            if isinstance(layer, ResBlock):
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

        
class UNet3D_MC_Dropout(nn.Module):
    def __init__(self, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16, p_dropout: float = 0.1):
        super().__init__()
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out
        self.n_features = n_filters_init
        self.p_dropout = p_dropout

        n = n_filters_init

        self.init_path = nn.Sequential(
            ResBlock3d(n_chans_in, n, kernel_size=3, padding=1),
            ResBlock3d(n, n, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n, n * 2, kernel_size=3, padding=1),
            ResBlock3d(n * 2, n * 2, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 2, n * 4, kernel_size=3, padding=1),
            ResBlock3d(n * 4, n * 4, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 4, n * 8, kernel_size=3, padding=1),
            ResBlock3d(n * 8, n * 8, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 8, n * 16, kernel_size=3, padding=1),
            ResBlock3d(n * 16, n * 16, kernel_size=3, padding=1),
            ResBlock3d(n * 16, n * 8, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up3 = nn.Sequential(
            ResBlock3d(n * 8, n * 8, kernel_size=3, padding=1),
            ResBlock3d(n * 8, n * 4, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up2 = nn.Sequential(
            ResBlock3d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock3d(n * 4, n * 2, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up1 = nn.Sequential(
            ResBlock3d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock3d(n * 2, n, kernel_size=3, padding=1),
            nn.Dropout(p=p_dropout),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.out_path = nn.Sequential(
            ResBlock3d(n, n, kernel_size=3, padding=1),
            ResBlock3d(n, n, kernel_size=3, padding=1),
            ResBlock3d(n, n_chans_out, kernel_size=1, padding=0, bias=True),
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    

class UNet3DLuna(nn.Module):
    def __init__(self, init_bias: float = None):
        super().__init__()
        self.init_bias = init_bias

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=layers.ResBlock3d,
                downsample=nn.MaxPool3d(2, ceil_mode=True),
                upsample=nn.Identity,
                merge=lambda left, down: torch.add(
                    *layers.interpolate_to_left(left, down, 'trilinear')),
                structure=[
                    [[8, 8, 8], [8, 8, 8]],
                    [[8, 16, 16], [16, 16, 8]],
                    [[16, 32, 32], [32, 32, 16]],
                    [[32, 64, 64], [64, 64, 32]],
                    [[64, 128, 128], [128, 128, 64]],
                    [[128, 256, 256], [256, 256, 128]],
                    [[256, 512, 512], [512, 512, 256]],
                    [[512, 1024, 1024], [1024, 1024, 512]],
                    [1024, 1024]
                ],
                kernel_size=3,
                padding=1
            ),
        )

        self.head = layers.PreActivation3d(8, 1, kernel_size=1)
        if init_bias is not None:
            self.head.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_features(self, x):
        return self.unet(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ResBlockDropout(nn.Module):
    """
    Performs a sequence of two convolutions with residual connection (Residual Block) and Dropout.
    ..
        in ---> (BN --> activation --> Conv) --> (BN --> activation --> Conv) --> Dropout + --> out
            |                                                                             ^
            |                                                                             |
             -----------------------------------------------------------------------------
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 activation_module=nn.ReLU, conv_module, batch_norm_module, dropout_module, p_dropout, **kwargs):
        super().__init__()
        # ### Features path ###
        pre_activation = partial(
            PreActivationND, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, conv_module=conv_module, 
            batch_norm_module=batch_norm_module, **kwargs)

        self.conv_path = nn.Sequential(pre_activation(in_channels, out_channels, stride=stride, bias=False),
                                       pre_activation(out_channels, out_channels, bias=bias),
                                       dropout_module(p=p_dropout))

        # ### Shortcut ###
        spatial_difference = np.floor(
            np.asarray(dilation) * (np.asarray(kernel_size) - 1) - 2 * np.asarray(padding)
        ).astype(int)
        if not (spatial_difference >= 0).all():
            raise ValueError(f"The output's shape cannot be greater than the input's shape. ({spatial_difference})")

        if in_channels != out_channels or stride != 1:
            self.adjust_to_stride = conv_module(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        x_conv = self.conv_path(x)
        shape = x_conv.shape[2:]
        axes = range(-len(shape), 0)
        x_skip = crop_to_shape(self.adjust_to_stride(x), shape=shape, axis=axes)
        return x_conv + x_skip
        
    
class UNet3DLunaMCDropout(nn.Module):
    def __init__(self, init_bias: float = None, p_dropout: float = 0.1):
        super().__init__()
        self.init_bias = init_bias
        self.p_dropout = p_dropout

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=partial(ResBlockDropout, conv_module=nn.Conv3d, batch_norm_module=nn.BatchNorm3d, 
                              dropout_module=nn.Dropout, p_dropout=p_dropout),
#                 layer=lambda in_, out, *args, **kwargs: nn.Sequential(layers.ResBlock3d(in_, out, *args, **kwargs),
#                                                                       nn.Dropout(p=self.p_dropout)),
                downsample=nn.MaxPool3d(2, ceil_mode=True),
                upsample=nn.Identity,
                merge=lambda left, down: torch.add(
                    *layers.interpolate_to_left(left, down, 'trilinear')),
                structure=[
                    [[8, 8, 8], [8, 8, 8]],
                    [[8, 16, 16], [16, 16, 8]],
                    [[16, 32, 32], [32, 32, 16]],
                    [[32, 64, 64], [64, 64, 32]],
                    [[64, 128, 128], [128, 128, 64]],
                    [[128, 256, 256], [256, 256, 128]],
                    [[256, 512, 512], [512, 512, 256]],
                    [[512, 1024, 1024], [1024, 1024, 512]],
                    [1024, 1024]
                ],
                kernel_size=3,
                padding=1
            ),
        )

        self.head = layers.PreActivation3d(8, 1, kernel_size=1)
        if init_bias is not None:
            self.head.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_features(self, x):
        return self.unet(x)

    def forward(self, x):
        return self.head(self.forward_features(x))
    
    
class FPN_WithFeatures(layers.FPN):
    
    def forward(self, x):
        print(x.shape, flush=True)
        levels, results = [], []
        for i, (layer, down, middle) in enumerate(zip_equal(self.down_path, self.downsample, self.middle_path)):
            x1 = layer(x)
            levels.append(middle(x1))
            x = down(x)
            if i == 1:
                break

        print(x1.shape, flush=True)
        return x1
    
    
class UNet3DLunaWithFeatures(nn.Module):
    def __init__(self, init_bias: float = None):
        super().__init__()
        self.init_bias = init_bias

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            FPN_WithFeatures(
                layer=layers.ResBlock3d,
                downsample=nn.MaxPool3d(2, ceil_mode=True),
                upsample=nn.Identity,
                merge=lambda left, down: torch.add(
                    *layers.interpolate_to_left(left, down, 'trilinear')),
                structure=[
                    [[8, 8, 8], [8, 8, 8]],
                    [[8, 16, 16], [16, 16, 8]],
                    [[16, 32, 32], [32, 32, 16]],
                    [[32, 64, 64], [64, 64, 32]],
                    [[64, 128, 128], [128, 128, 64]],
                    [[128, 256, 256], [256, 256, 128]],
                    [[256, 512, 512], [512, 512, 256]],
                    [[512, 1024, 1024], [1024, 1024, 512]],
                    [1024, 1024]
                ],
                kernel_size=3,
                padding=1
            ),
        )

        self.head = layers.PreActivation3d(8, 1, kernel_size=1)
        if init_bias is not None:
            self.head.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_features(self, x):
        return self.unet(x)

    def forward(self, x):
        return self.head(self.forward_features(x))