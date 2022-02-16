import warnings

import torch
from torch import nn

from dpipe import layers
from dpipe.layers.resblock import ResBlock2d, ResBlock3d


class UNet2D(nn.Module):
    def __init__(self, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16):
        super().__init__()
        self.n_filters_init = n_filters_init
        n = n_filters_init

        self.init_path = nn.Sequential(
            ResBlock2d(n_chans_in, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            ResBlock2d(n, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            ResBlock2d(n * 2, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            ResBlock2d(n * 4, n * 8, kernel_size=3, padding=1),
            ResBlock2d(n * 8, n * 8, kernel_size=3, padding=1)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            ResBlock2d(n * 8, n * 16, kernel_size=3, padding=1),
            ResBlock2d(n * 16, n * 16, kernel_size=3, padding=1),
            ResBlock2d(n * 16, n * 8, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up3 = nn.Sequential(
            ResBlock2d(n * 8, n * 8, kernel_size=3, padding=1),
            ResBlock2d(n * 8, n * 4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up2 = nn.Sequential(
            ResBlock2d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock2d(n * 4, n * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.up1 = nn.Sequential(
            ResBlock2d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock2d(n * 2, n, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.out_path = nn.Sequential(
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n, kernel_size=3, padding=1),
            ResBlock2d(n, n_chans_out, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
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


class UNet3D(nn.Module):
    def __init__(self, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16):
        super().__init__()
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out
        self.n_features = n_filters_init

        n = n_filters_init

        self.init_path = nn.Sequential(
            ResBlock3d(n_chans_in, n, kernel_size=3, padding=1),
            ResBlock3d(n, n, kernel_size=3, padding=1),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n, n * 2, kernel_size=3, padding=1),
            ResBlock3d(n * 2, n * 2, kernel_size=3, padding=1)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 2, n * 4, kernel_size=3, padding=1),
            ResBlock3d(n * 4, n * 4, kernel_size=3, padding=1)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 4, n * 8, kernel_size=3, padding=1),
            ResBlock3d(n * 8, n * 8, kernel_size=3, padding=1)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2, 2, ceil_mode=True),
            ResBlock3d(n * 8, n * 16, kernel_size=3, padding=1),
            ResBlock3d(n * 16, n * 16, kernel_size=3, padding=1),
            ResBlock3d(n * 16, n * 8, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up3 = nn.Sequential(
            ResBlock3d(n * 8, n * 8, kernel_size=3, padding=1),
            ResBlock3d(n * 8, n * 4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up2 = nn.Sequential(
            ResBlock3d(n * 4, n * 4, kernel_size=3, padding=1),
            ResBlock3d(n * 4, n * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.up1 = nn.Sequential(
            ResBlock3d(n * 2, n * 2, kernel_size=3, padding=1),
            ResBlock3d(n * 2, n, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )

        self.out_path = nn.Sequential(
            ResBlock3d(n, n, kernel_size=3, padding=1),
            ResBlock3d(n, n, kernel_size=3, padding=1),
        )
        
        self.final_conv = ResBlock3d(n, n_chans_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        warnings.filterwarnings('ignore')
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3_up = self.up3(self.bottleneck(x3) + x3)
        x2_up = self.up2(x3_up + x2)
        x1_up = self.up1(x2_up + x1)
        x_final = self.out_path(x1_up + x0)
        x_out = self.final_conv(x_final)
        warnings.filterwarnings('default')
        return x_out

    
class UNet3DWithActivations(UNet3D):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        warnings.filterwarnings('ignore')
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3_up = self.up3(self.bottleneck(x3) + x3)
        x2_up = self.up2(x3_up + x2)
        x1_up = self.up1(x2_up + x1)
        x_final = self.out_path(x1_up + x0)
#         x_out = self.final_conv(x_final)
        warnings.filterwarnings('default')
        return x_final


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

    
class UNet3DLunaMCDropout(nn.Module):
    def __init__(self, init_bias: float = None, p_dropout: float = 0.1):
        super().__init__()
        self.init_bias = init_bias
        self.p_dropout = p_dropout

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=lambda in_, out, *args, **kwargs: nn.Sequential(layers.ResBlock3d(in_, out, *args, **kwargs),
                                                                      nn.Dropout(p=self.p_dropout)),
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