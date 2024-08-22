from .module import Module
from . import conv_functional as F
from ..tensor import Tensor

class MaxPool1D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = F.MaxPool1D_stride1()(x, self.kernel_size)
        Z = F.Downsample1D()(Z, factor=self.stride)
        return Z
    
class MeanPool1D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = F.MeanPool1D_stride1()(x, self.kernel_size)
        Z = F.Downsample1D()(Z, factor=self.stride)
        return Z

class MaxPool2D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = F.MaxPool2D_stride1()(x, self.kernel_size)
        Z = F.Downsample2D()(Z, factor=self.stride)
        return Z

class MeanPool2D(Module):
    def __init__(self, kernel_size:int, stride:int=1) -> None:
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x:Tensor) -> Tensor:
        Z = F.MeanPool2D_stride1()(x, self.kernel_size)
        Z = F.Downsample2D()(Z, factor=self.stride)
        return Z
