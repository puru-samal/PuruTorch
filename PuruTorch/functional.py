# ------------------------------------------
# Function Superclass
# ------------------------------------------
import numpy as np
from .tensor import Tensor
from .utils import Function, unbroadcast
from typing import Tuple, List, Union, Optional

# ------------------------------------------
# Tensor Functions
# ------------------------------------------

class Add(Function):
    """
    Tensor addition operation.
    """    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data + b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Neg(Function):
    """
    Tensor negation operation.
    """    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad 
        data = -a.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(-grad_output.data, a.shape)
        return [Tensor.tensor(a_grad)]


class Sub(Function):
    """
    Tensor subtraction operation.
    """    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data - b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(-grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Mul(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data * b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * b.data, a.shape)
        b_grad = unbroadcast(grad_output.data * a.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Pow(Function):
    """
    Tensor power operation.
    """    
    def forward(self, a: Tensor, exponent: Union[int, float, Tensor]) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(exponent, Tensor):
            exponent = Tensor.tensor(np.array(exponent))

        self.ctx.exponent = exponent
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data**exponent.data 
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        exponent = self.ctx.exponent
        a_grad = unbroadcast(grad_output.data * (exponent.data * (a.data**(exponent.data-1))), a.shape)
        return [Tensor.tensor(a_grad)]


class Div(Function):
    """
    Tensor division operation.
    """    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data / b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * (1 / b.data), a.shape)
        b_grad = unbroadcast(grad_output.data * (-a.data /(b.data**2.0)), b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class MatMul(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a: Tensor, b: Tensor):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data @ b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data @ b.data.T, a.shape)
        b_grad = unbroadcast(a.data.T @ grad_output.data  , b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Slice(Function):
    """
    Tensor Slice operation.
    """    
    def forward(self, a: Tensor, key: Tuple[slice, ...]) -> Tensor:
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        self.ctx.save_for_backward(a)
        self.ctx.key = key
        requires_grad = a.requires_grad
        data = a.data[key]
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = np.zeros_like(a.data)
        if grad_output.data.size != 0 and a_grad[self.ctx.key].size != 0:
           a_grad[self.ctx.key] = grad_output.data
        return [Tensor.tensor(a_grad)]

class Reshape(Function):
    """
    Tensor Reshape operation.
    """    
    def forward(self, a: Tensor, shape: Tuple[int, ...]) -> Tensor:
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        self.ctx.save_for_backward(a)
        self.ctx.shape = a.shape
        requires_grad = a.requires_grad
        data = a.data.reshape(shape)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data.reshape(self.ctx.shape)
        return [Tensor.tensor(a_grad)]


class Squeeze(Function):
    """
    Tensor Squeeze operation.
    """    
    def forward(self, a: Tensor, axis: int) -> Tensor:
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        requires_grad = a.requires_grad
        data = np.squeeze(a.data, axis=axis)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data
        if a.shape != a_grad.shape:
            a_grad = np.expand_dims(a_grad, axis=self.ctx.axis)
        return [Tensor.tensor(a_grad)]


class Unsqueeze(Function):
    """
    Tensor Unsqueeze operation.
    """    
    def forward(self, a: Tensor, axis: int) -> Tensor:
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")
        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        requires_grad = a.requires_grad
        data = np.expand_dims(a.data, axis=axis)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data
        if a.shape != a_grad.shape:
            a_grad = np.squeeze(a_grad, axis=self.ctx.axis)
        return [Tensor.tensor(a_grad)]
    

class Transpose(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            print(f"Can't transpose a non-Tensor. Recieved {a} of type: {type(a)}")
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data.T
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(grad_output.data.T, a.shape)
        return [Tensor.tensor(a_grad)]
    

class Log(Function):
    """
    Tensor log operation.
    """    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.log(a.data)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data * (1/a.data)
        return [Tensor.tensor(a_grad)]
    

class Exp(Function):
    """
    Tensor exp operation.
    """    
    def forward(self, a: Tensor) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.exp(a.data)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        self.ctx.output = out # use in backward
        return out
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a_grad = grad_output.data * self.ctx.output.data
        return [Tensor.tensor(a_grad)]


class Max(Function):
    """
    Tensor Max operation.
    """    
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False ) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        data = np.max(a.data, axis=axis, keepdims=keepdims)
        self.ctx.max = data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        a = self.ctx.saved_tensors[0]

        # broadcast if not keepdims
        if not self.ctx.keepdims: 
            if self.ctx.axis is None: # scalar
                self.ctx.max = self.ctx.max * np.ones_like(a.data)
                a_grad = grad_output.data * np.ones_like(a.data)
            else: # else axis reduction
                self.ctx.max = np.expand_dims(self.ctx.max, self.ctx.axis) * np.ones_like(a.data)
                a_grad = np.expand_dims(grad_output.data, self.ctx.axis) * np.ones_like(a.data)
        
        a_grad = a_grad * np.where(a.data == self.ctx.max, 1.0, 0.0)
        return [Tensor.tensor(a_grad)]


class Sum(Function):
    """
    Tensor Max operation.
    """    
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        data = np.sum(a.data, axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output: Tensor) -> Tensor:
        a = self.ctx.saved_tensors[0]

        # if asix reduction happened
        if not self.ctx.keepdims and self.ctx.axis is not None: 
            a_grad = np.expand_dims(grad_output.data, self.ctx.axis)
        else:
            a_grad = grad_output.data

        a_grad = a_grad * np.ones_like(a.data)
        return [Tensor.tensor(a_grad)]


# ------------------------------------------
# Tensor Bindings
# ------------------------------------------

Tensor.__add__  = lambda self, other : Add()(self, other)
Tensor.__radd__ = lambda self, other : Add()(other, self)
Tensor.__iadd__ = lambda self, other : Add()(self, other)
Tensor.__sub__  = lambda self, other : Sub()(self, other)
Tensor.__rsub__ = lambda self, other : Sub()(other, self)
Tensor.__isub__ = lambda self, other : Sub()(self, other)
Tensor.__neg__  = lambda self : Neg()(self)
Tensor.__mul__  = lambda self, other : Mul()(self, other)
Tensor.__rmul__ = lambda self, other : Mul()(other, self)
Tensor.__imul__ = lambda self, other : Mul()(self, other)
Tensor.__pow__  = lambda self, exponent : Pow()(self, exponent)
Tensor.__truediv__ = lambda self, other : Div()(self, other)
Tensor.__matmul__  = lambda self, other : MatMul()(self, other)
Tensor.__getitem__ = lambda self, key : Slice()(self, key)
Tensor.T    = property(lambda self : Transpose()(self))
Tensor.reshape    = lambda self, *shape : Reshape()(self, shape)
Tensor.squeeze    = lambda self, axis : Squeeze()(self, axis)
Tensor.unsqueeze  = lambda self, axis : Unsqueeze()(self, axis)
Tensor.log  = lambda self : Log()(self)
Tensor.exp  = lambda self : Exp()(self)
Tensor.max  = lambda self, axis=None, keepdims=False : Max()(self, axis, keepdims)
Tensor.sum  = lambda self, axis=None, keepdims=False : Sum()(self, axis, keepdims)
