import numpy as np
from ..tensor import Tensor
from ..utils import Function, unbroadcast
from typing import Tuple, List, Union, Optional, Literal

# ------------------------------------------
#  Activation Functionals
#  Objects superclassed by Function.
#  Handle fwd/bwd passes of certain
#  activation functions.
# ------------------------------------------

class Identity(Function):
    """
    Tensor Identity operation.
    """ 
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
     
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data
        return [Tensor.tensor(a_grad)]
    

class Sigmoid(Function):
    """
    Tensor Sigmoid operation.
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = 1 / (1 + np.exp(-a.data))
        self.ctx.fwd_out = data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data * (self.ctx.fwd_out - self.ctx.fwd_out**2)
        return [Tensor.tensor(a_grad)]
    

class Tanh(Function):
    """
    Tensor Tanh operation.
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.tanh(a.data)
        self.ctx.fwd_out = data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data * (1 - self.ctx.fwd_out**2)
        return [Tensor.tensor(a_grad)]


class ReLU(Function):
    """
    Tensor ReLU operation.
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.maximum(0, a.data)
        self.ctx.fwd_out = data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data * np.where(self.ctx.fwd_out > 0.0, 1.0, 0.0)
        return [Tensor.tensor(a_grad)]


class Softmax(Function):
    """
    Tensor Softmax operation.
    NOTE: Softmax is taken over the last axis, so inputs must be (*, num_classes)
    Pytorch equivalent: torch.nn.Softmax(dim=-1)
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        self.ctx.is_batched = a.ndim == 2

        if not self.ctx.is_batched:
            a_data = a.data.reshape(1, -1) # (k,) -> (1, k)
        else:
            a_data = a.data # (N, k)
            
        data = np.exp(a_data) / np.sum(np.exp(a_data), axis=-1, keepdims=True) # (N, k)
        
        if not self.ctx.is_batched:
            data = data.reshape(-1) # (1,k) -> (k,)
        
        self.ctx.fwd_out = data # Use to calculate the jacobian
        return Tensor(data, requires_grad, self if requires_grad else None)
    

    def backward(self, grad_output: Tensor) -> List[Tensor]:
        if not self.ctx.is_batched:
            fwd_data = self.ctx.fwd_out.reshape(1, -1)
            g_out    = grad_output.data.reshape(1, -1)
        else:
            fwd_data = self.ctx.fwd_out
            g_out    = grad_output.data

        # shape: (N, C, 1) * (N, 1, C) -> (N, C, C)
        # elem: -a_i * a_j 
        jacobian = -fwd_data[..., None] * fwd_data[:, None, :]
        di, dj = np.diag_indices_from(jacobian[0])

        # replace diag elems: a_i * (1 - a_i)
        jacobian[:, di, dj] = fwd_data * (1. - fwd_data)

        # (N, C) , (N, C, C)
        a_grad = np.tensordot(g_out, jacobian, axes=(-1,-2))

        if not self.ctx.is_batched:
            a_grad = a_grad.reshape(-1)
        return [Tensor.tensor(a_grad)]

