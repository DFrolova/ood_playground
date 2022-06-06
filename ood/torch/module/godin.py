import warnings
from functools import partial

import numpy as np
import torch
from torch import nn

from dpipe import layers
from dpipe.layers.resblock import ResBlock, ResBlock2d, ResBlock3d
from dpipe.layers import PreActivationND
from dpipe.im.utils import identity
from dpipe.im.shape_ops import crop_to_shape
from dpipe.itertools import zip_equal
from ood.torch.model import get_resizing_features_modules
from ood.torch.module.unet import UNet, UNet3DLuna


class UNet_GODIN(UNet):
    def __init__(self, ndim: int = 3, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16,
                 return_features_from: tuple = (3, )):
        super().__init__()
        
        n = n_filters_init
        resblock = ResBlock2d if (ndim == 2) else ResBlock3d

        self.out_path = nn.Sequential(
            resblock(n, n, kernel_size=3, padding=1),                           # 18
            resblock(n, n, kernel_size=3, padding=1),                           # 19
        )
        
        self.out_path_h = resblock(n, n_chans_out, kernel_size=1, padding=0, bias=True)      # 20
        self.out_path_g = resblock(n, 1, kernel_size=1, padding=0, bias=True)      # 20
        

    def forward(self, x: torch.Tensor, return_num_and_denom: bool = False):

            warnings.filterwarnings('ignore')
            x0 = self.init_path(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x3_up = self.up3(self.bottleneck(x3) + x3)
            x2_up = self.up2(x3_up + x2)
            x1_up = self.up1(x2_up + x1)
            x_out = self.out_path(x1_up + x0)
            h_out = self.out_path_h(x_out)
            g_out = torch.sigmoid(self.out_path_g(x_out))
            warnings.filterwarnings('default')
                        
            if return_num_and_denom:
                return h_out / g_out, h_out, g_out
            
            return h_out / g_out
        
        
class UNet3DLuna_GODIN(UNet3DLuna):
    def __init__(self, init_bias: float = None):
        super().__init__()

        self.head = layers.PreActivation3d(8, 1, kernel_size=1)
        self.head_g = layers.PreActivation3d(8, 1, kernel_size=1)
        if init_bias is not None:
            self.head.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))
            self.head_g.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_features(self, x):
        return self.unet(x)

    def forward(self, x: torch.Tensor, return_num_and_denom: bool = False):
        features = self.forward_features(x)
        h = self.head(features)
        g = torch.sigmoid(self.head_g(features))
        
        if return_num_and_denom:
                return h / g, h, g
            
        return h / g