from .optimizer import Optimizer
from ..tensor import Tensor
from typing import Tuple
import numpy as np

class AdamW(Optimizer):

    def __init__(self, params, lr=0.001, betas:Tuple[float, float]=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.t = 0
        self.betas = betas
        self.m = [Tensor.zeros_like(param) for param in self.params] # Mean derivative state for each param
        self.v = [Tensor.zeros_like(param) for param in self.params] # Mean sq. derivative state for each param
        self.weight_decay = weight_decay
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
    
            # weight decay update
            param.data = param.data - param.data * self.weight_decay * self.lr

            self.m[i].data = self.betas[0] * self.m[i].data + (1. - self.betas[0]) * param.grad.data
            self.v[i].data = self.betas[1] * self.v[i].data + (1. - self.betas[1]) * (param.grad.data**2)

            m = self.m[i].data / (1. - self.betas[0] ** self.t)
            v = self.v[i].data / (1. - self.betas[1] ** self.t)

            param.data = param.data - self.lr * m / (np.sqrt(v) + self.eps)
