import numpy as np
from .module import Module
from ..tensor import Tensor, Parameter
from . import conv_functional as F

class Conv1D(Module):
    '''Applies a 1D convolution over an input signal composed of several input planes.'''
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.W = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels, in_channels, kernel_size)))
        
        self.b = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels,)))
        

    def init_weights(self, W: Tensor, b: Tensor) -> None:
        self.W = Parameter(W)
        self.b = Parameter(b)

    def forward(self, x : Tensor) -> Tensor :
        is_batched = x.ndim == 3
        if not is_batched:
            x = x.unsqueeze(0)
        
        y  = F.Pad1D()(x, self.padding)
        y  = F.Conv1D_stride1()(y, self.W, self.b)
        y  = F.Downsample1D()(y, self.stride)

        if not is_batched:
            y = y.squeeze(0)
        return y 