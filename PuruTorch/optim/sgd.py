from .optimizer import Optimizer
from ..tensor import Tensor

class SGD(Optimizer):

    def __init__(self, params, lr=0.001, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.momentums = [Tensor.zeros_like(param) for param in self.params] # Momentum state for each param
    
    def step(self):
        for i, param in enumerate(self.params):
            self.momentums[i].data = self.momentum * self.momentums[i].data + param.grad.data
            param.data = param.data - self.lr * self.momentums[i].data


