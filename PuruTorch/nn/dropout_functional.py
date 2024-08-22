import numpy as np
from ..tensor import Tensor
from ..utils import Function, unbroadcast
from typing import Tuple, List, Union, Optional, Literal

# ------------------------------------------
#  Dropout Functionals
# ------------------------------------------

class dropout(Function):
    """
    An explicit dropout functional to reuse values computed in forward.
    Used in the dropout module in nn.
    """ 

    def __call__(self, x:Tensor, p:float, train:bool) -> Tensor:
        return self.forward(x, p, train)
    
    def forward(self, x:Tensor, p:float, train:bool) -> Tensor:
        self.ctx.save_for_backward(x)
        self.ctx.train = train
        self.ctx.p = p
        if train:
            mask = np.random.binomial(1., 1. - p, x.shape)
            self.ctx.mask = mask 
            data = x.data * mask
            data /= (1. - p)
            return Tensor(data, x.requires_grad, self if x.requires_grad else None)
        else:
            return x
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.train:
            return [Tensor.tensor(grad_output.data * self.ctx.mask / (1. - self.ctx.p))]
        else:
            x = self.ctx.saved_tensors[0]
            return [Tensor.ones_like(x)]


class dropout1d(Function):
    """
    An explicit dropout1d functional to reuse values computed in forward.
    Used in the Dropout1d module in nn.
    """ 
    def __call__(self, x:Tensor, p:float, train:bool) -> Tensor:
        return self.forward(x, p, train)
       
    def forward(self, x:Tensor, p:float, train:bool) -> Tensor:
        self.ctx.save_for_backward(x)
        self.ctx.train = train
        if train:
            ch_selector = np.random.binomial(1., 1. - p, size=(x.shape[0], x.shape[1], 1))
            mask = np.tile(ch_selector, (1, 1, x.shape[2]))
            self.ctx.mask = mask 
            self.ctx.p = p
            data = x.data * mask
            data /= (1. - p)
            return Tensor(data, x.requires_grad, self if x.requires_grad else None)
        else:
            return x
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.train:
            return [Tensor.tensor(grad_output.data * self.ctx.mask / (1. - self.ctx.p))]
        else:
            x = self.ctx.saved_tensors[0]
            return [Tensor.ones_like(x)]


class dropout2d(Function):
    """
    An explicit dropout2d functional to reuse values computed in forward.
    Used in the Dropout2d module in nn.
    """
    def __call__(self, x:Tensor, p:float, train:bool) -> Tensor:
        return self.forward(x, p, train)
        
    def forward(self, x:Tensor, p:float, train:bool) -> Tensor:
        self.ctx.save_for_backward(x)
        self.ctx.train = train
        if train:
            ch_selector = np.random.binomial(1., 1. - p, size=(x.shape[0], x.shape[1], 1, 1))
            mask = np.tile(ch_selector, (1, 1, x.shape[2], x.shape[3]))
            self.ctx.mask = mask 
            self.ctx.p = p
            data = x.data * mask
            data /= (1. - p)
            return Tensor(data, x.requires_grad, self if x.requires_grad else None)
        else:
            return x
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.train:
            return [Tensor.tensor(grad_output.data * self.ctx.mask / (1. - self.ctx.p))]
        else:
            x = self.ctx.saved_tensors[0]
            return [Tensor.ones_like(x)]
