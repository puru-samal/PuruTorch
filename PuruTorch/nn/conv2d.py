import numpy as np
from .module import Module
from .resampling import *
from ..tensor import Tensor, Parameter
from ..functional import *

class Conv2D(Module):
    '''Applies a 2D convolution over an input signal composed of several input planes.'''
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding=0) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.W = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels, in_channels, kernel_size, kernel_size)))
        
        self.b = Parameter(Tensor.random_uniform(-np.sqrt(1 / (in_channels * kernel_size)),
                                                np.sqrt(1 / (in_channels * kernel_size)),
                                                size=(out_channels,)))
        

    def init_weights(self, W: Tensor, b: Tensor):
        self.W = Parameter(W)
        self.b = Parameter(b)

    def forward(self, x : Tensor) -> Tensor :
        is_batched = x.ndim == 4
        if not is_batched:
            x = x.unsqueeze(0)
        
        y  = Pad2D()(x, self.padding)
        y  = Conv2D_stride1()(y, self.W, self.b)
        y  = Downsample2D()(y, self.stride)

        if not is_batched:
            y = y.squeeze(0)
        return y 