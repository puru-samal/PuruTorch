# ------------------------------------------
# Function Superclass
# ------------------------------------------
import numpy as np
from .tensor import Tensor
from .utils import Function, unbroadcast
from typing import Tuple, List, Union, Optional, Literal

# ------------------------------------------
# General Functionals (w/ Tensor bindings)
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

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data + b.data
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

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad 
        data = -a.data
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

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data - b.data
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

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data * b.data
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

        self.ctx.exponent = exponent
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data**exponent.data 
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
        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data / b.data
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a, b = self.ctx.saved_tensors
        a_grad = unbroadcast(grad_output.data * (1 / b.data), a.shape)
        b_grad = unbroadcast(grad_output.data * (-a.data /(b.data**2.0)), b.shape)
        '''
        print("Div: ")
        print(f"a_grad: {a_grad}")
        print(f"b_grad: {b_grad}")
        print(f"b.requires_grad: {b.requires_grad}")
        '''
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

        self.ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        data = a.data @ b.data
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

        self.ctx.save_for_backward(a)
        self.ctx.key = key
        requires_grad = a.requires_grad
        data = a.data[key]
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

        self.ctx.save_for_backward(a)
        self.ctx.shape = a.shape
        requires_grad = a.requires_grad
        data = a.data.reshape(shape)
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

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        requires_grad = a.requires_grad
        data = np.squeeze(a.data, axis=axis)
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
        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        requires_grad = a.requires_grad
        data = np.expand_dims(a.data, axis=axis)
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
        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = a.data.T
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

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.log(a.data)
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

        self.ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        data = np.exp(a.data)
        out = Tensor(data, requires_grad, self if requires_grad else None)
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

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        data = np.max(a.data, axis=axis, keepdims=keepdims)
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

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        data = np.sum(a.data, axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
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

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims

        requires_grad = a.requires_grad
        data = np.mean(a.data, axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
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

        self.ctx.save_for_backward(a)
        self.ctx.axis = axis
        self.ctx.keepdims = keepdims
        self.ctx.corr = corr

        requires_grad = a.requires_grad
        data = np.var(a.data, axis=axis, keepdims=keepdims, ddof=corr)
        out = Tensor(data, requires_grad, self if requires_grad else None)
        return out
    
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

# ------------------------------------------
#  Activation Functional
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


# ------------------------------------------
#  Loss Functional
# ------------------------------------------

class SoftmaxCrossEntropy(Function):
    """
    An explicit SoftmaxCrossEntropy functional to reuse values computed in forward.
    """    

    def __call__(self, predictions: Tensor, targets: Tensor, 
                 reduction: Union[None, Literal['mean', 'sum']]="mean") -> Tensor:
        return self.forward(predictions, targets, reduction)

    def forward(self, predictions: Tensor, targets: Tensor, 
                reduction: Union[None, Literal['mean', 'sum']]="mean") -> Tensor:
        super().forward()
        self.ctx.save_for_backward(predictions)
        self.ctx.reduction = reduction
        self.ctx.targets = targets.data 
        self.ctx.softmax = np.exp(predictions.data) / np.sum(np.exp(predictions.data), axis=-1, keepdims=True)
        data = np.sum(-targets.data * np.log(self.ctx.softmax), axis=-1)
        if reduction is None:
            data = data
        elif reduction == "mean":
            data = np.mean(data)
            *N_dim, _ = predictions.shape
            self.ctx.N = np.prod(N_dim)
        elif reduction == "sum":
            data = np.sum(data)
        return Tensor(data, predictions.requires_grad, self if predictions.requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        if self.ctx.reduction is None:
            a_grad = grad_output.data * (self.ctx.softmax - self.ctx.targets)
        elif self.ctx.reduction == "mean":
            a_grad = grad_output.data * (self.ctx.softmax - self.ctx.targets) / self.ctx.N
        elif self.ctx.reduction == "sum":
            a_grad = grad_output.data * (self.ctx.softmax - self.ctx.targets)
        return [Tensor.tensor(a_grad)]

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


# ------------------------------------------
#  Conv Functionals
# ------------------------------------------

class Pad1D(Function):
    """
    Functional to handle forward/backward passes of 
    padding a A 3D input for 1D convolution.
    """
    def __call__(self, A:Tensor, padding:int=0) -> Tensor:
        return self.forward(A, padding)
    
    def forward(self, A:Tensor, padding:int=0) -> Tensor:
        self.ctx.save_for_backward(A)
        self.ctx.padding = padding # for backward
        if padding > 0:
            padded_A = np.pad(A.data, pad_width=((0,), (0,), (padding,)), mode='constant')
        else:
            padded_A = A.data.copy()
        return Tensor(padded_A, A.requires_grad, self if A.requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.padding > 0:
            dLdA = grad_output.data[:, :, self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding)]
        else:
            dLdA = grad_output.data.copy()
        return [Tensor.tensor(dLdA)]
        

class Conv1D_stride1(Function):
    """
    Functional to handle forward/backward passes of 
    a 1D stride1 convolution 
    """
    def __call__(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        return self.forward(A, W, b)
    
    def forward(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        # A : N * C_in  * W_in
        # Z : N * C_out * W_out
        # W : C_out * C_in * K
        # b : C_out,

        self.ctx.save_for_backward(A, W, b)
        N, _, W_in  = A.shape      # batch size x in_chans x in_width
        C_out, _, K = W.shape      # out_chans x in_chans x kernel
        W_out       = W_in - K + 1 # out_width

        Z = np.zeros((N, C_out, W_out))

        for w in range(W_out):
            axs = ([1, 2], [1, 2])
            Z[:, :, w] += np.tensordot(A.data[:, :, w : w+K], W.data, axes=axs)
            Z[:, :, w] += b.data
        
        requires_grad = A.requires_grad or W.requires_grad or b.requires_grad
        return Tensor(Z, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        # dLdA : N * C_in  * W_in
        # dLdZ : N * C_out * W_out (grad_output)
        # dLdW : C_out * C_in * K
        # dLdb : C_out,

        A, W, _ = self.ctx.saved_tensors
        _, _, K     = W.shape
        _, _, W_out = grad_output.shape
        _, _, W_in  = A.shape

        dLdb = np.sum(grad_output.data, axis=(0, 2))

        dLdW = np.zeros_like(W.data)
        for k in range(K):
            axs = ([0, 2], [0, 2])
            dLdW[:, :,k] += np.tensordot(grad_output.data, A.data[:, :, k:k+W_out], axes=axs)

        dLdA = np.zeros_like(A.data)
        # Padding only on laxt axis
        pwidths = ((0,), (0,), (K-1,))
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')
        flipW = np.flip(W.data, axis=2)  # Flip only on last axis

        for w in range(W_in):
            axs = ([1, 2], [0, 2])
            dLdA[:, :, w] = np.tensordot(pdLdZ[:, :, w:w+K], flipW, axes=axs)

        return [Tensor.tensor(dLdA), Tensor.tensor(dLdW), Tensor.tensor(dLdb)]

class Pad2D(Function):
    """
    Functional to handle forward/backward passes of 
    padding a A 4D input for 2D convolution.
    """
    def __call__(self, A:Tensor, padding:int=0) -> Tensor:
        return self.forward(A, padding)
    
    def forward(self, A:Tensor, padding:int=0) -> Tensor:
        self.ctx.save_for_backward(A)
        self.ctx.padding = padding # for backward
        if  padding > 0:
            padded_A = np.pad(A.data, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant')
        else:
            padded_A = A.data.copy()
        return Tensor(padded_A, A.requires_grad, self if A.requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.padding > 0:
            dLdA = grad_output.data[:, :, 
                                    self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding), 
                                    self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding)]
        else:
            dLdA = grad_output.data.copy()
        return [Tensor.tensor(dLdA)]


class Conv2D_stride1(Function):
    """
    Functional to handle forward/backward passes of 
    a 2D stride1 convolution 
    """
    def __call__(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        return self.forward(A, W, b)
    
    def forward(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        # A : N * C_in  * H_in * W_in
        # Z : N * C_out * H_in * W_out
        # W : C_out * C_in * K * K
        # b : C_out,

        self.ctx.save_for_backward(A, W, b)
        N, _, H_in, W_in  = A.shape      # batch size x in_chans x in_width
        C_out, _, _, K    = W.shape      # out_chans x in_chans x kernel
        H_out, W_out      = H_in - K + 1, W_in - K + 1 # out_width

        Z = np.zeros((N, C_out, H_out, W_out))

        for h in range(H_out):
            for w in range(W_out):
                axs = ([1, 2, 3], [1, 2, 3])
                Z[:, :, h, w] += np.tensordot(A.data[:, :, h:h+K, w:w+K], W.data, axes=axs)
                Z[:, :, h, w] += b.data
        
        requires_grad = A.requires_grad or W.requires_grad or b.requires_grad
        return Tensor(Z, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        # dLdA : N * C_in  * W_in
        # dLdZ : N * C_out * W_out (grad_output)
        # dLdW : C_out * C_in * K
        # dLdb : C_out * 1

        A, W, _ = self.ctx.saved_tensors
        _, _, _, K         = W.shape
        _, _, H_out, W_out = grad_output.shape
        _, _, H_in,  W_in  = A.shape

        dLdb = np.sum(grad_output.data, axis=(0, 2, 3))

        dLdW = np.zeros_like(W.data)
        for kh in range(K):
            for kw in range(K):
                axs = ([0, 2, 3], [0, 2, 3])
                dLdW[:, :, kh, kw] += np.tensordot(grad_output.data, A.data[:, :, kh:kh+H_out, kw:kw+W_out], axes=axs)

        dLdA = np.zeros_like(A.data)
        # Padding only on laxt axis
        pwidths = ((0,), (0,), (K-1,), (K-1,))
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')
        flipW = np.flip(W.data, axis=(2,3))  # Flip only on last two axis

        for h in range(H_in):
            for w in range(W_in):
                axs = ([1, 2, 3], [0, 2, 3])
                dLdA[:, :, h, w] = np.tensordot(pdLdZ[:, :, h:h+K, w:w+K], flipW, axes=axs)

        return [Tensor.tensor(dLdA), Tensor.tensor(dLdW), Tensor.tensor(dLdb)]

# ------------------------------------------
#  Pool Functionals
# ------------------------------------------

class MaxPool1D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_width)
        """

        self.ctx.save_for_backward(A)
        N, C_in, W_in = A.shape
        C_out = C_in
        W_out = W_in - kernel + 1

        self.ctx.maxindex = np.empty((N, C_out, W_out), dtype=tuple)
        Z = np.zeros((N, C_out, W_out))

        for batch in range(N):
            for ch in range(C_out):
                for w in range(W_out):
                    scan = A.data[batch, ch, w:w+kernel]
                    Z[batch, ch, w] = np.max(scan)
                    self.ctx.maxindex[batch, ch, w] = np.unravel_index(np.argmax(scan), scan.shape)
                    self.ctx.maxindex[batch, ch, w] = tuple(np.add((w), self.ctx.maxindex[batch, ch, w]))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, W_out = grad_output.shape

        for batch in range(N):
            for ch in range(C_out):
                for w in range(W_out):
                    i1  = self.ctx.maxindex[batch, ch, w]
                    dLdA[batch, ch, i1] = grad_output.data[batch, ch, w]
        return [Tensor.tensor(dLdA)]


class MeanPool1D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_width)
        """
        self.ctx.save_for_backward(A)
        self.ctx.kernel = kernel
        N, C_in, W_in = A.shape
        C_out = C_in
        W_out = W_in - kernel + 1
        Z = np.zeros((N, C_out, W_out))

        for w in range(W_out):
            Z[:, :, w] = np.mean(A.data[:, :, w:w+kernel], axis=2)

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, W_out = grad_output.shape

        pwidths = ((0,), (0,), (self.ctx.kernel-1,))
        # Pad with zeroes to shape match
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')

        for w in range(W_out):
            dLdA[:, :, w] = np.mean(pdLdZ[:, :, w:w+self.ctx.kernel], axis=2)
                
        return [Tensor.tensor(dLdA)]
       

class MaxPool2D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_height,  input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_height, output_height)
        """

        self.ctx.save_for_backward(A)
        N, C_in, H_in, W_in = A.shape
        C_out = C_in
        H_out, W_out = H_in - kernel + 1, W_in - kernel + 1

        self.ctx.maxindex = np.empty((N, C_out, H_out, W_out), dtype=tuple)
        Z = np.zeros((N, C_out, H_out, W_out))

        for batch in range(N):
            for ch in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        scan = A.data[batch, ch, h:h+kernel, w:w+kernel]
                        Z[batch, ch, h, w] = np.max(scan)
                        self.ctx.maxindex[batch, ch, h, w] = np.unravel_index(np.argmax(scan), scan.shape)
                        self.ctx.maxindex[batch, ch, h, w] = tuple(np.add((h, w), self.ctx.maxindex[batch, ch, h, w]))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_height, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, H_out, W_out = grad_output.shape

        for batch in range(N):
            for ch in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        i1, i2 = self.ctx.maxindex[batch, ch, h, w]
                        dLdA[batch, ch, i1, i2] = grad_output.data[batch, ch, h, w]
        return [Tensor.tensor(dLdA)]


class MeanPool2D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_height,  input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_height, output_width)
        """
        self.ctx.save_for_backward(A)
        self.ctx.kernel = kernel
        N, C_in, H_in, W_in = A.shape
        C_out = C_in
        H_out, W_out = H_in - kernel + 1, W_in - kernel + 1
        Z = np.zeros((N, C_out, H_out, W_out))

        for h in range(H_out):
            for w in range(W_out):
                Z[:, :, h, w] = np.mean(A.data[:, :, h:h+kernel, w:w+kernel], axis=(2, 3))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_height, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, H_out, W_out = grad_output.shape

        pwidths = ((0,), (0,), (self.ctx.kernel-1,), (self.ctx.kernel-1,))
        # Pad with zeroes to shape match
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')

        for h in range(H_out):
            for w in range(W_out):
                dLdA[:, :, h, w] = np.mean(pdLdZ[:, :, h:h+self.ctx.kernel, w:w+self.ctx.kernel], axis=(2, 3))
                
        return [Tensor.tensor(dLdA)]
       