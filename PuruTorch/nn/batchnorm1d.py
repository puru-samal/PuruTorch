import numpy as np
from .module import Module
from ..tensor import Tensor, Parameter
from ..functional import *

class BatchNorm1d(Module):
    """
    Applies batch normalization over 2D and 3D inputs. 
    Tested against Pytorch's Batchnorm1D implementation.
    """
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # Learnable affine parameters
        self.gamma = Parameter(Tensor.ones((self.num_features,)))
        self.beta  = Parameter(Tensor.zeros((self.num_features,)))

        # Running mean and var
        self.running_mean = Tensor.zeros((self.num_features,))
        self.running_var  = Tensor.ones((self.num_features,))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0] # batch size
        mean = x.mean(axis=0)
        var = x.var(axis=0, keepdims=False, corr=0)          # Used for output
        unbiased_var = x.var(axis=0, keepdims=False, corr=1) # Used for moving avg
        if self.mode == 'train':
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1. - self.momentum) * self.running_var  + self.momentum * unbiased_var
            Nx = (x - mean) / (var + self.eps) ** 0.5
        else:
            Nx = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
        
        y = Nx * self.gamma + self.beta
        return y