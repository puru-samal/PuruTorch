import numpy as np
from typing import Tuple, Optional

# ------------------------------------------
# Function Superclass
# ------------------------------------------

class ContextManager:
    """
    Helper class to save Tensors required for bachward computations during the forward pass.
    -> store tensors for backprop with:
        ctx.save_for_backward(tensor)
    -> store other parameters useful for gradient computation (eg, shape..) with:
        ctx.attr = attr
    """   
    def __init__(self):
        self.saved_tensors = []
    
    def save_for_backward(self, *args):
        """
        Save Tensor's during forward pass.
        """        
        for arg in args:
            if not type(arg).__name__ == "Tensor":
                raise ValueError(f"Only Tensors can be saved. Recieved {arg} of type: {type(arg)} instead.")
            self.saved_tensors.append(arg)

    def clear(self):
        """
        Clear context. To be used during zero_grad.
        """        
        self.saved_tensors.clear()


class Function:
    """
    Superclass for all tensor operation types.
    """    
    def __init__(self, debug=False) -> None:
        self.ctx = ContextManager()
        self.debug = debug
    
    def __call__(self, *args):
        return self.forward(*args)
        
    def forward(self, *args):
        return
    
    def backward(self, *args):
        if self.debug:
            print("")
            print(self.__class__.__name__+"_backward")
        

# ------------------------------------------
# Tensor Class
# ------------------------------------------

class Tensor :
    """
    The fundamental datastructure used in this library. 
    Implemented as wrapper around the np.ndarray class to facilitate gradient tracking.
    """    

    def __init__(self, data : np.ndarray, requires_grad : bool = False, op : Optional[Function] = None):
        """ 
        Args:
            data (np.ndarray): The underlying np.ndarray
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.
            op (Function, optional): The operation used to generate a new Tensor. 
                                     Tracked only if requires_grad=True. Defaults to None.
        """            
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = op
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = Tensor.tensor(np.zeros_like(data)) if self.requires_grad else None
        self.dtype = data.dtype
        self.is_leaf = self.grad_fn == None
    
    def __repr__(self):
        """
        Handles calls to print(Tensor)
        """        
        rg_str = f"requires_grad={self.requires_grad}," if self.requires_grad else ""
        gf_str = f"grad_fn={self.grad_fn.__class__.__name__}," if self.grad_fn is not None else ""
        return f"Tensor({self.data}, {rg_str} {gf_str})"
    
    def backward(self, grad : Optional['Tensor'] = None):
        """
        Updates Tensor's grad attribute and calles the backward methods on the 
        Tensor's used to create it.
        Args:
            grad (Tensor, optional): Gradient of output. Defaults to None.

        Raises:
            RuntimeError: if backward is called on a Tensor with requires_grad=False.
        """        
        if not self.requires_grad:
            raise RuntimeError(f"Tried to call backward on a Tensor with requires_grad=False!")

        if grad == None:
            grad = Tensor.ones_like(self)

        self.grad += grad

        if not self.is_leaf:
            inp_grads = self.grad_fn.backward(grad_output=self.grad)

            for (inp_tensor, inp_grad) in zip(self.grad_fn.ctx.saved_tensors, inp_grads):
                if isinstance(inp_tensor, Tensor) and inp_tensor.requires_grad:
                    inp_tensor.backward(inp_grad)
            
            self.grad = None
        return

    # ------------------------------------------
    # Tensor Initializers
    # ------------------------------------------

    @staticmethod
    def tensor(data : np.ndarray, requires_grad = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class containing 'data'

        Args:
            data (np.ndarray): The underlying np.ndarray
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.

        Returns:
            Tensor
        """    
        return Tensor(data, requires_grad)

    @staticmethod
    def zeros(shape : Tuple[int, ...], requires_grad = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class filled with zeros.

        Args:
            shape (Tuple[int, ...]): Shape of the Tensor
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.

        Returns:
            Tensor
        """    
        return Tensor(np.zeros(shape), requires_grad)

    @staticmethod
    def ones(shape : Tuple[int, ...], requires_grad = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class filled with zeros.

        Args:
            shape (Tuple[int, ...]): Shape of the Tensor
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.

        Returns:
            Tensor
        """    
        return Tensor(np.ones(shape), requires_grad)

    @staticmethod
    def zeros_like(tensor : 'Tensor', requires_grad = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class of the same shape as the
        given Tensor instance filled with zeros.

        Args:
            tensor (Tensor): Tensor to copy shape from
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.

        Returns:
            Tensor
        """    
        return Tensor(np.zeros_like(tensor.data), requires_grad)

    @staticmethod
    def ones_like(tensor : 'Tensor', requires_grad = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class of the same shape as the
        given Tensor instance filled with ones.

        Args:
            tensor (Tensor): Tensor to copy shape from
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.

        Returns:
            Tensor
        """    
        return Tensor(np.ones_like(tensor.data), requires_grad)

    # ------------------------------------------
    # Tensor Operations
    # ------------------------------------------
   
    def __add__(self, other):
        """ New = self + other """
        op = Add()
        return op(self, other)

    def __radd__(self, other):
        """ New = other + self """
        op = Add()
        return op(other, self)

    def __iadd__(self, other):
        """ self += other """
        op = Add()
        return op(self, other)

    def __sub__(self, other):
        """ New = self - other """
        op = Sub()
        return op(self, other)
    
    def __rsub__(self, other):
        """ New = other - self """
        op = Sub()
        return op(other, self)

    def __isub__(self, other):
        """ self -= other """
        op = Sub()
        return op(self, other)
    
    def __neg__(self):
        """ self = -self """
        op = Neg()
        return op(self)

    def __mul__(self, other):
        """ New = self * other """
        op = Mul()
        return op(self, other)

    def __rmul__(self, other):
        """ New = other * self """
        op = Mul()
        return op(other, self)

    def __imul__(self, other):
        """ self *= other """
        op = Mul()
        return op(self, other)
    
    def __pow__(self, other):
        """ New = self ** other """
        op = Pow()
        return op(self, other)

    def __matmul__(self, other):
        """ New = self @ other """
        op = MatMul()
        return op(self, other)

    def __truediv__(self, other):
        """ New = self / other """
        op = Div()
        return op(self, other)
    
    def __getitem__(self, key): 
        """ New = self[index] """
        """ Can use self.data[key] """
        pass

    def __gt__(self, other):
        """ New = self > other """
        pass

    @property    
    def T(self):
        """ New = self.T """
        op = Transpose()
        return op(self)

    def reshape(self, shape):
        pass

    def squeeze(self, axis=0):
        pass

    def unsqueeze(self, axis=0):
        pass

    def sum(self, axis=None, keepdims=False):
        pass

    def mean(self, axis=None, keepdims=False):
        pass


# ------------------------------------------
# Tensor Functions
# ------------------------------------------

def unbroadcast(tensor_data : np.ndarray, shape_to_match, to_keep=0):
    """
    Helper function to handle broadcasting.
    """
    for _ in range(len(tensor_data.shape) - len(shape_to_match)):
        tensor_data = tensor_data.sum(axis=0)
    
    for i, dim in enumerate(shape_to_match):
        if dim == 1:
            tensor_data = tensor_data.sum(axis=i, keepdims=True)
    
    return tensor_data


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

