import numpy as np
from .module import Module
from ..tensor import Tensor, Parameter
from ..functional import *

class Linear(Module):
    """
    Applies affine linear transformation. 
    y = x @ W.T + b.T
    """
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = Parameter(Tensor.random_uniform(-np.sqrt(1 / in_features),
                                                np.sqrt(1 / in_features),
                                                size=(out_features, in_features)))
        
        if bias:
            self.b = Parameter(Tensor.random_uniform(-np.sqrt(1 / in_features),
                                                    np.sqrt(1 / in_features),
                                                    size=(out_features,)))
        else:
            self.b = Tensor.zeros((out_features,))
        

    def init_weights(self, W: Tensor, b: Tensor):
        self.W = Parameter(W)
        self.b = Parameter(b)

    def forward(self, x : Tensor) -> Tensor :
        is_batched = x.ndim == 2
        if not is_batched:
            x = x.unsqueeze(0)
        
        y = x @ self.W.T + self.b.T

        if not is_batched:
            y = y.squeeze(0)
        return y 