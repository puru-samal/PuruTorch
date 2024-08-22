import numpy as np
from .module import Module
from .resampling import Downsample1D, Downsample2D
from ..tensor import Tensor
from ..functional import *
from ..utils import Function

class MaxPool1D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = MaxPool1D_stride1()(x, self.kernel_size)
        Z = Downsample1D()(Z, factor=self.stride)
        return Z
    
class MeanPool1D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = MeanPool1D_stride1()(x, self.kernel_size)
        Z = Downsample1D()(Z, factor=self.stride)
        return Z

class MaxPool2D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = MaxPool2D_stride1()(x, self.kernel_size)
        Z = Downsample2D()(Z, factor=self.stride)
        return Z

class MeanPool2D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = MeanPool2D_stride1()(x, self.kernel_size)
        Z = Downsample2D()(Z, factor=self.stride)
        return Z
