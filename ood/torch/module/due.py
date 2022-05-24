import torch
from torch import nn

from dpipe import layers
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
                 n_power_iterations=1, coeff=3, **kwargs):
        super().__init__()
        if batch_norm_module is not None:
            self.bn = batch_norm_module(in_features)
        else:
            self.bn = identity
        self.activation = activation_module()
        self.layer = layer_module(in_features, out_features, **kwargs)
        
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            self.layer = spectral_norm_fc(self.layer, coeff, n_power_iterations)
        else:
            # Otherwise use spectral norm conv, with loose bound
            input_dim = (in_c, input_size, input_size) # TODO check
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
                 activation_module=nn.ReLU, conv_module, batch_norm_module, coeff, n_power_iterations, **kwargs):
        super().__init__()
        # ### Features path ###
        pre_activation = partial(
            PreActivationSpectralNorm, kernel_size=kernel_size, padding=padding, dilation=dilation,
            activation_module=activation_module, layer_module=conv_module, batch_norm_module=batch_norm_module, 
            coeff=coeff, n_power_iterations=n_power_iterations, **kwargs
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
            self.adjust_to_stride = spectral_norm_fc(self.adjust_to_stride, coeff=coeff, n_power_iterations=n_power_iterations)
        else:
            self.adjust_to_stride = identity

    def forward(self, x):
        x_conv = self.conv_path(x)
        shape = x_conv.shape[2:]
        axes = range(-len(shape), 0)
        x_skip = crop_to_shape(self.adjust_to_stride(x), shape=shape, axis=axes)
        return x_conv + x_skip


def wrapped_conv(conv_module, input_size, in_c, out_c, kernel_size, stride):

    conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

    if not spectral_conv:
        return conv

    if kernel_size == 1:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
    else:
        # Otherwise use spectral norm conv, with loose bound
        input_dim = (in_c, input_size, input_size)
        wrapped_conv = spectral_norm_conv(
            conv, coeff, input_dim, n_power_iterations
        )

    return wrapped_conv


class UNet3DLunaDUE(nn.Module):
    def __init__(self, init_bias: float = None, coeff: float = 3., n_power_iterations: int = 1):
        super().__init__()
        self.init_bias = init_bias
        self.batchnorm = partial(SpectralBatchNorm3d, coeff=coeff)
        
        input_dim = (in_c, input_size, input_size)
        wrapped_conv = spectral_norm_conv(
            conv, coeff, input_dim, n_power_iterations
        )
        self.conv = partial(nn.Conv3d(in_c, out_c, kernel_size, stride, padding, bias=False)

        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
        else:
            # Otherwise use spectral norm conv, with loose bound
            input_dim = (in_c, input_size, input_size)
            wrapped_conv = spectral_norm_conv(
                conv, coeff, input_dim, n_power_iterations
            )

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=partial(layers.ResBlock3d, conv_module=nn.Conv3d, batch_norm_module=self.batchnorm),
                
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