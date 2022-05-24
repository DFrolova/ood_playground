# import warnings
# from functools import partial

import numpy as np
import torch
from torch import nn

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from .dkl import initial_values, GP, DKL


# from dpipe import layers
# from dpipe.im.utils import identity
# from dpipe.im.shape_ops import crop_to_shape
# from ood.torch.model import get_resizing_features_modules
# from .spectral_batchnorm import SpectralBatchNorm3d
# from .spectral_norm_conv import spectral_norm_conv
# from .spectral_norm_fc import spectral_norm_fc

# from .unet_spectral_norm_mri import UNetSpectralNormFeatureExtractor#, PreActivationSpectralNorm, ResBlockSpectralNorm




# feature_extraxtor = UNetSpectralNormFeatureExtractor()





class UNetDUE(nn.Module):
    def __init__(self, feature_extractor, batch_iter, device, num_train_voxels, 
                 num_classes=1, n_inducing_points=2, kernel='RBF'):
        
        super().__init__()
        
        self.feature_extractor = feature_extractor
        
        print('preparing initial points')
        
        initial_inducing_points, initial_lengthscale = initial_values(
            batch_iter=batch_iter, 
            feature_extractor=feature_extractor, 
            device=device, 
            n_inducing_points=n_inducing_points
        )
        
        gp = GP(
            num_outputs=num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=kernel,
        )
        
        model = DKL(feature_extractor, gp)

        likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
        likelihood = likelihood.to(device)

        elbo_fn = VariationalELBO(likelihood, gp, num_data=num_train_voxels)
        loss_fn = lambda x, y: -elbo_fn(x, y)
        
        
        

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