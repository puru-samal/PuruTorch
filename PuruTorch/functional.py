# ------------------------------------------
# Function Superclass
# ------------------------------------------
import numpy as np
from .tensor import Tensor
from .utils import Function, unbroadcast

# ------------------------------------------
# Tensor Functions
# ------------------------------------------

class Add(Function):
    """
    Tensor addition operation.
    """    
    def forward(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data + b.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Neg(Function):
    """
    Tensor negation operation.
    """    
    def forward(self, a):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad 
        data = -a.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(-grad_output.data, a.shape)
        return [Tensor.tensor(a_grad)]


class Sub(Function):
    """
    Tensor subtraction operation.
    """    
    def forward(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data - b.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(-grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Mul(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data * b.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * b.data, a.shape)
        b_grad = unbroadcast(grad_output.data * a.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Pow(Function):
    """
    Tensor power operation.
    """    
    def forward(self, a, exponent):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(exponent, Tensor):
            exponent = Tensor.tensor(np.array(exponent))

        self.ctx.exponent = exponent
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data**exponent.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a = self.ctx.saved_tensors[0]
        exponent = self.ctx.exponent
        a_grad = unbroadcast(grad_output.data * (exponent.data * (a.data**(exponent.data-1))), a.shape)
        return [Tensor.tensor(a_grad)]


class Div(Function):
    """
    Tensor division operation.
    """    
    def forward(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data / b.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * (1 / b.data), a.shape)
        b_grad = unbroadcast(grad_output.data * (-a.data /(b.data**2.0)), b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Transpose(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a):
        if not isinstance(a, Tensor):
            print(f"Can't transpose a non-Tensor. Recieved {a} of type: {type(a)}")
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data.T
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(grad_output.data.T, a.shape)
        return [Tensor.tensor(a_grad)]


class MatMul(Function):
    """
    Tensor multiplication operation.
    """    
    def forward(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data @ b.data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data @ b.data.T, a.shape)
        b_grad = unbroadcast(a.data.T @ grad_output.data  , b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Slice(Function):
    """
    Tensor Slice operation.
    """    
    def forward(self, a, key):
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        self.ctx.save_for_backward(a)
        self.ctx.key = key
        requires_grad = a.requires_grad
        data = a.data[key]
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a = self.ctx.saved_tensors[0]
        a_grad = np.zeros_like(a.data)
        if grad_output.data.size != 0 and a_grad[self.ctx.key].size != 0:
           a_grad[self.ctx.key] = grad_output.data
        return [Tensor.tensor(a_grad)]


class Log(Function):
    """
    Tensor log operation.
    """    
    def forward(self, a):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.log(a.data)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output):
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data * (1/a.data)
        return [Tensor.tensor(a_grad)]
    

class Exp(Function):
    """
    Tensor exp operation.
    """    
    def forward(self, a):
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        self.ctx.save_for_backward(a)
        
        requires_grad = a.requires_grad
        data = np.exp(a.data)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        self.output = out # use in backward
        return out
    
    def backward(self, grad_output):
        a_grad = grad_output.data * self.output.data
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
Tensor.reshape    = lambda self : "NOT IMPLEMENTED!"
Tensor.squeeze    = lambda self : "NOT IMPLEMENTED!"
Tensor.unsqueeze  = lambda self : "NOT IMPLEMENTED!"
Tensor.cat    = lambda self : "NOT IMPLEMENTED!"
Tensor.stack  = lambda self : "NOT IMPLEMENTED!"
Tensor.masked_fill = lambda self : "NOT IMPLEMENTED!"
Tensor.log  = lambda self : Log()(self)
Tensor.exp  = lambda self : Exp()(self)
Tensor.max  = lambda self : "NOT IMPLEMENTED!"
Tensor.mean = lambda self : "NOT IMPLEMENTED!"
Tensor.var  = lambda self : "NOT IMPLEMENTED!"
Tensor.sum  = lambda self : "NOT IMPLEMENTED!"
Tensor.tanh = lambda self : "NOT IMPLEMENTED!"