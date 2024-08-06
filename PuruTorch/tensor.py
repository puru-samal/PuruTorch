import numpy as np
from typing import Tuple, Optional
from .utils import Function
from collections import deque
from dataclasses import dataclass

# ------------------------------------------
# Tensor Class
# ------------------------------------------

class Tensor :
    """
    The fundamental datastructure used in this library. 
    Implemented as wrapper around the np.ndarray class to facilitate gradient tracking.
    """    

    def __init__(self, data: np.ndarray, requires_grad: bool = False, op: Optional[Function] = None):
        """ 
        Args:
            data (np.ndarray): The underlying np.ndarray
            requires_grad (bool, optional): Whether to track a Tensor's gradients. Defaults to False.
            op (Function, optional): The operation used to generate a new Tensor. 
                                     Tracked only if requires_grad=True. Defaults to None.
        """
        if not isinstance(data, np.ndarray):            
            self.data = np.array(data)
        else:
            self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = op
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = None
        self.dtype = self.data.dtype
        self.is_leaf = self.grad_fn == None

    def __repr__(self):
        """
        Handles calls to print(Tensor)
        """
        dt_str = f"dtype={self.dtype}"        
        rg_str = f"requires_grad={self.requires_grad}," if (self.requires_grad and self.is_leaf) else ""
        gf_str = f"grad_fn={self.grad_fn.__class__.__name__}.backward," if self.grad_fn is not None else ""
        return f"Tensor({self.data}, {dt_str}, {rg_str} {gf_str})"
    
    def backward(self, grad: Optional['Tensor'] = None) -> None:
        """
        Updates Tensor's grad attribute and recursively calls the backward methods on the 
        Tensor's used to create it (children).
        Args:
            grad (Tensor, optional): Gradient of output. Defaults to None.

        Raises:
            RuntimeError: if backward is called on a Tensor with requires_grad=False.
        """        
        if not self.requires_grad:
            raise RuntimeError(f"Tried to call backward on a Tensor with requires_grad=False!")
        
        # Init grad
        if self.requires_grad and self.grad is None:
            self.grad = Tensor.tensor(np.zeros_like(self.data))

        if grad is None:
            grad = Tensor.ones_like(self)

        self.grad += grad

        if not self.is_leaf:
            inp_grads = self.grad_fn.backward(grad_output=self.grad)
            for (inp_tensor, inp_grad) in zip(self.grad_fn.ctx.saved_tensors, inp_grads):
                if isinstance(inp_tensor, Tensor) and inp_tensor.requires_grad:
                    inp_tensor.backward(inp_grad)
            self.grad = None # Only leaf tensors retain gradients
        return
    
    def print_op_tree(self, depth=0,):
        print( "".join('*' for _ in range(depth)))
        print("____________________")
        print(f"| out_shape: {self.shape}")
        print(f"| out: {self}")
        print(f"| op: {self.grad_fn}")
       
        if not self.grad_fn is None:
            for (i, inp) in enumerate(self.grad_fn.ctx.saved_tensors):
                print(f"| inp{i+1}_shape: {inp.shape}")
                print(f"| inp{i+1}: {inp}")
        
        print("____________________")
           

    @staticmethod
    def bfs(root : 'Tensor'):
        
        @dataclass
        class AutogradNode:
            tensor: 'Tensor'
            depth : int

        queue = deque([AutogradNode(tensor=root, depth=0)])
        visited = set()

        while queue:
            parent = queue.popleft()
            parent.tensor.print_op_tree(parent.depth)
            visited.add(parent.tensor)
            if parent.tensor.grad_fn is not None:
                for child in parent.tensor.grad_fn.ctx.saved_tensors:
                    if child not in visited:
                        queue.append(AutogradNode(tensor=child, depth=parent.depth+1))

    # ------------------------------------------
    # Tensor Initializers
    # ------------------------------------------

    @staticmethod
    def tensor(data: np.ndarray, requires_grad: bool = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class containing 'data'
        """    
        return Tensor(data, requires_grad)

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class filled with zeros.
        """    
        return Tensor(np.zeros(shape), requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class filled with zeros.
        """    
        return Tensor(np.ones(shape), requires_grad)

    @staticmethod
    def zeros_like(tensor: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class of the same shape as the
        given Tensor instance filled with zeros.
        """    
        return Tensor(np.zeros_like(tensor.data), requires_grad)

    @staticmethod
    def ones_like(tensor: 'Tensor', requires_grad: bool = False) -> 'Tensor':
        """
        Creates an instance of a Tensor class of the same shape as the
        given Tensor instance filled with ones.
        """    
        return Tensor(np.ones_like(tensor.data), requires_grad)
    
    @staticmethod
    def random_uniform(lo: float, hi: float, size: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor' :
        """
        Create's a Tensor by filled with samples drawn from a uniform distribution.
        """        
        return Tensor(np.random.uniform(lo, hi, size), requires_grad)

# ------------------------------------------
# Parameter Class
# ------------------------------------------

class Parameter(Tensor):
    """
    A Tensor subclass that will always track gradients. 
    """    
    def __init__(self, tensor : Tensor):
        super().__init__(tensor.data, requires_grad=True, op=None)
    
    def __repr__(self):
        """
        Handles calls to print(Parameter)
        """
        dt_str = f"dtype={self.dtype}"        
        return f"Parameter({self.data}, {dt_str})"