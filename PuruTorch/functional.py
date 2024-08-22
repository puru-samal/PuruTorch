# ------------------------------------------
# Function Superclass
# ------------------------------------------
import numpy as np
from .tensor import Tensor
from .utils import Function, unbroadcast
from typing import Tuple, List, Union, Optional

# ------------------------------------------
# General Functionals (w/ Tensor bindings)
# ------------------------------------------
# Collection of objects superclassed by Function.
# These objects handle fwd/backward passes of some
# primitive Tensor operations and are bound as
# class methods to the Tensor object.
# ------------------------------------------

class Add(Function):
    """
    Tensor addition operation.
    """
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)
        
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        data = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a, b)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Neg(Function):
    """
    Tensor negation operation.
    """
    def __call__(self, a: Tensor) -> Tensor:
        return self.forward(a)   
        
    def forward(self, a: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = -a.data
        requires_grad = a.requires_grad 
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(-grad_output.data, a.shape)
        return [Tensor.tensor(a_grad)]


class Sub(Function):
    """
    Tensor subtraction operation.
    """
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)   
        
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        data = a.data - b.data
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a, b)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data, a.shape)
        b_grad = unbroadcast(-grad_output.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Mul(Function):
    """
    Tensor multiplication operation.
    """
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)   
        
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        data = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a, b)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * b.data, a.shape)
        b_grad = unbroadcast(grad_output.data * a.data, b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Pow(Function):
    """
    Tensor power operation.
    """
    def __call__(self, a: Tensor, exponent: Union[int, float, Tensor]) -> Tensor:
        return self.forward(a, exponent) 
        
    def forward(self, a: Tensor, exponent: Union[int, float, Tensor]) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(exponent, Tensor):
            exponent = Tensor.tensor(np.array(exponent))

        data = a.data**exponent.data
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.exponent = exponent 
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        exponent = self.ctx.exponent
        a_grad = unbroadcast(grad_output.data * (exponent.data * (a.data**(exponent.data-1))), a.shape)
        return [Tensor.tensor(a_grad)]


class Div(Function):
    """
    Tensor division operation.
    """
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)   
        
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))
        
        data = a.data / b.data
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a, b)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * (1 / b.data), a.shape)
        b_grad = unbroadcast(grad_output.data * (-a.data /(b.data**2.0)), b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class MatMul(Function):
    """
    Tensor multiplication operation.
    """
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.forward(a, b)   
    
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))
        if not isinstance(b, Tensor):
            b = Tensor.tensor(np.array(b))

        data = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a, b)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data @ b.data.T, a.shape)
        b_grad = unbroadcast(a.data.T @ grad_output.data  , b.shape)
        return [Tensor.tensor(a_grad), Tensor.tensor(b_grad)]


class Slice(Function):
    """
    Tensor Slice operation.
    """
    def __call__(self, a: Tensor, key: Tuple[slice, ...]) -> Tensor:
        return self.forward(a, key)
        
    def forward(self, a: Tensor, key: Tuple[slice, ...]) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        data = a.data[key]
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.key = key
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = np.zeros_like(a.data)
        if grad_output.data.size != 0 and a_grad[self.ctx.key].size != 0:
           a_grad[self.ctx.key] = grad_output.data
        return [Tensor.tensor(a_grad)]

class Reshape(Function):
    """
    Tensor Reshape operation.
    """
    def __call__(self, a: Tensor, shape: Tuple[int, ...]) -> Tensor:
        return self.forward(a, shape)
        
    def forward(self, a: Tensor, shape: Tuple[int, ...]) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        data = a.data.reshape(shape)
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.shape = a.shape 
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a_grad = grad_output.data.reshape(self.ctx.shape)
        return [Tensor.tensor(a_grad)]


class Squeeze(Function):
    """
    Tensor Squeeze operation.
    """
    def __call__(self, a: Tensor, axis: int) -> Tensor:
        return self.forward(a, axis)
        
    def forward(self, a: Tensor, axis: int) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")

        data = np.squeeze(a.data, axis=axis)
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data
        if a.shape != a_grad.shape:
            a_grad = np.expand_dims(a_grad, axis=self.ctx.axis)
        return [Tensor.tensor(a_grad)]


class Unsqueeze(Function):
    """
    Tensor Unsqueeze operation.
    """
    def __call__(self, a: Tensor, axis: int) -> Tensor:
        return self.forward(a, axis)
        
    def forward(self, a: Tensor, axis: int) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            raise ValueError(f"Called slice on non-Tensor. Recieved {a} of type: {type(a)}")
        
        data = np.expand_dims(a.data, axis=axis)
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data
        if a.shape != a_grad.shape:
            a_grad = np.squeeze(a_grad, axis=self.ctx.axis)
        return [Tensor.tensor(a_grad)]
    

class Transpose(Function):
    """
    Tensor multiplication operation.
    """
    def __call__(self, a: Tensor) -> Tensor:
        return self.forward(a)
        
    def forward(self, a: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            print(f"Can't transpose a non-Tensor. Recieved {a} of type: {type(a)}")
        
        data = a.data.T
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = unbroadcast(grad_output.data.T, a.shape)
        return [Tensor.tensor(a_grad)]
    

class Log(Function):
    """
    Tensor log operation.
    """
    def __call__(self, a: Tensor) -> Tensor:
        return self.forward(a)
        
    def forward(self, a: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.log(a.data)
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]
        a_grad = grad_output.data * (1./a.data)
        return [Tensor.tensor(a_grad)]
    

class Exp(Function):
    """
    Tensor exp operation.
    """ 
    def __call__(self, a: Tensor) -> Tensor:
        return self.forward(a)
       
    def forward(self, a: Tensor) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.exp(a.data)
        requires_grad = a.requires_grad
        out = Tensor(data, requires_grad, self if requires_grad else None)
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.fwd_out = out # use in backward
        return out
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a_grad = grad_output.data * self.ctx.fwd_out.data
        return [Tensor.tensor(a_grad)]


class Max(Function):
    """
    Tensor Max operation.
    """
    def __call__(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return self.forward(a, axis, keepdims)
        
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False ) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.max(a.data, axis=axis, keepdims=keepdims)
        requires_grad = a.requires_grad
        if requires_grad: # save for bwd
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
            self.ctx.keepdims = keepdims
            self.ctx.fwd_out = data
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]

        # broadcast if not keepdims
        if not self.ctx.keepdims:
            # if scalar, broadcast max,grad to original shape 
            if self.ctx.axis is None: 
                self.ctx.fwd_out = self.ctx.fwd_out * np.ones_like(a.data)
                g_out = grad_output.data * np.ones_like(a.data)
            else: # else axis reduction, expand then broadcast
                self.ctx.fwd_out = np.expand_dims(self.ctx.fwd_out, self.ctx.axis) * np.ones_like(a.data)
                g_out = np.expand_dims(grad_output.data, self.ctx.axis) * np.ones_like(a.data)
        else:
            g_out = grad_output
        
        a_grad = g_out * np.where(a.data == self.ctx.fwd_out, 1.0, 0.0)
        return [Tensor.tensor(a_grad)]


class Sum(Function):
    """
    Tensor Sum operation.
    """  
    def __call__(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return self.forward(a, axis, keepdims)
      
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.sum(a.data, axis=axis, keepdims=keepdims)
        requires_grad = a.requires_grad
        if requires_grad:
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
            self.ctx.keepdims = keepdims
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]

        # if axis reduction happened
        if not self.ctx.keepdims and self.ctx.axis is not None:
            # expand reduced axis 
            a_grad = np.expand_dims(grad_output.data, self.ctx.axis)
        else: # Scalar
            a_grad = grad_output.data.copy()

        a_grad = a_grad * np.ones_like(a.data)
        return [Tensor.tensor(a_grad)]
    

class Mean(Function):
    """
    Tensor Mean operation.
    """
    def __call__(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        return self.forward(a, axis, keepdims)
        
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.mean(a.data, axis=axis, keepdims=keepdims)
        requires_grad = a.requires_grad
        if requires_grad:
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
            self.ctx.keepdims = keepdims
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]

        # if axis reduction happened
        if not self.ctx.keepdims and self.ctx.axis is not None:
            # expand reduced axis 
            a_grad = np.expand_dims(grad_output.data, self.ctx.axis)
        else: # Scalar
            a_grad = grad_output.data
        
        # NOTE: Handled case for when ctx.axis is a tuple
        denom = np.prod(a.shape) if self.ctx.axis is None else \
                (np.prod(a.shape[self.ctx.axis]) if not isinstance(self.ctx.axis, tuple) else \
                 np.prod(tuple(a.shape[i] for i in self.ctx.axis)))
        a_grad = a_grad * np.ones_like(a.data) / denom
        return [Tensor.tensor(a_grad)]
    

class Var(Function):
    """
    Tensor Mean operation. Divisor used in calculation is N - corr.
    """
    def __call__(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False, corr=1) -> Tensor:
        return self.forward(a, axis, keepdims, corr)
        
    def forward(self, a: Tensor, axis: Optional[int] = None, keepdims: bool = False, corr=1) -> Tensor:
        super().forward()
        if not isinstance(a, Tensor):
            a = Tensor.tensor(np.array(a))

        data = np.var(a.data, axis=axis, keepdims=keepdims, ddof=corr)
        requires_grad = a.requires_grad
        if requires_grad:
            self.ctx.save_for_backward(a)
            self.ctx.axis = axis
            self.ctx.keepdims = keepdims
            self.ctx.corr = corr
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a = self.ctx.saved_tensors[0]

        # if axis reduction happened
        if not self.ctx.keepdims and self.ctx.axis is not None:
            # expand reduced axis 
            a_grad = np.expand_dims(grad_output.data, self.ctx.axis) * np.ones_like(a.data)
        else: # Scalar
            a_grad = grad_output.data * np.ones_like(a.data)
        
        a_grad *=  2. * (a.data - np.mean(a.data, axis=self.ctx.axis, keepdims=True))
        # NOTE: Handled case for when ctx.axis is a tuple
        denom = np.prod(a.shape) if self.ctx.axis is None else \
                (np.prod(a.shape[self.ctx.axis]) if not isinstance(self.ctx.axis, tuple) else \
                 np.prod(tuple(a.shape[i] for i in self.ctx.axis)))
        a_grad /= (denom - self.ctx.corr) 
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
Tensor.max  = lambda self, axis=None, keepdims=False : Max() (self, axis, keepdims)
Tensor.sum  = lambda self, axis=None, keepdims=False : Sum() (self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False : Mean()(self, axis, keepdims)
Tensor.var  = lambda self, axis=None, keepdims=False, corr=1 : Var() (self, axis, keepdims, corr)
