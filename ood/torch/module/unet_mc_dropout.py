import warnings

import torch
from torch import nn

from dpipe.layers.resblock import ResBlock3d


class UNet3D_MC_Dropout(nn.Module):
    def __init__(self, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16, p_dropout: float = 0.1):
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
            ResBlock3d(n, n_chans_out, kernel_size=1, padding=0, bias=True),
        )
        
        self.dropout = nn.Dropout(p=p_dropout)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        warnings.filterwarnings('ignore')
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3_up = self.up3(self.dropout(self.bottleneck(x3) + x3))
        x2_up = self.up2(self.dropout(x3_up + x2))
        x1_up = self.up1(self.dropout(x2_up + x1))
        x_out = self.out_path(self.dropout(x1_up + x0))
        warnings.filterwarnings('default')
        return x_out

    
def enable_dropout(model: nn.Module):
    """ Function to enable the dropout layers during test-time """
    print('ENABLING DROPOUT')
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(m.__class__.__name__)
            m.train()