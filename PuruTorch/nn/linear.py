from typing import Any
import numpy as np
from .module import Module
from ..tensor import Tensor, Parameter
from ..functional import *

class Linear(Module):
    
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = Parameter(Tensor.random_uniform(-np.sqrt(1 / in_features),
                                                np.sqrt(1 / in_features),
                                                size=(out_features, in_features)))
        
        self.b = Parameter(Tensor.random_uniform(-np.sqrt(1 / in_features),
                                                np.sqrt(1 / in_features),
                                                size=(out_features,)))
        
        self.momentum_W = Tensor.zeros_like(self.W)
        self.momentum_b = Tensor.zeros_like(self.b)

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