import numpy as np
from .module import Module
from . import conv_functional as F
from ..tensor import Tensor, Parameter

class ConvTranspose1D(Module):
    '''Applies a 1D transposed convolution operator over an input image composed of several input planes.'''
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.factor       = upsampling_factor

        self.W = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels, in_channels, kernel_size)))
        
        self.b = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels,)))
        

    def init_weights(self, W: Tensor, b: Tensor):
        self.W = Parameter(W)
        self.b = Parameter(b)

    def forward(self, x : Tensor) -> Tensor :
        is_batched = x.ndim == 3
        if not is_batched:
            x = x.unsqueeze(0)
        
        y  = F.Upsample1D()(x, self.factor)
        y  = F.Conv1D_stride1()(y, self.W, self.b)

        if not is_batched:
            y = y.squeeze(0)
        return y 