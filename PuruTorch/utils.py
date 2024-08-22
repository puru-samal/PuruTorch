
import numpy as np
from typing import Tuple, Literal

class ContextManager:
    """
    Helper class to save Tensors required for bachward computations during the forward pass.
    During forward calls:
    -> Store tensors for backprop with:
        ctx.save_for_backward(tensor)
    -> Store other parameters useful for gradient computation (eg, shape..) with:
        ctx.attr = attr
    During backward calls:
    -> Access stored tensors by indexing into 
        ctx.saved_tensors
    -> Access other parameters useful for gradient computation (eg, shape..) with:
        ctx.attr = attr
    """   
    def __init__(self):
        self.saved_tensors = []
    
    def save_for_backward(self, *args) -> None:
        """
        Save Tensor's during forward pass.
        """        
        for arg in args:
            if not (type(arg).__name__ == "Tensor" or type(arg).__name__ == "Parameter"):
                raise ValueError(f"Only Tensors can be saved. Recieved {arg} of type: {type(arg)} instead.")
            self.saved_tensors.append(arg)

    def clear(self) -> None:
        """
        Clear context. To be used during zero_grad.
        """        
        self.saved_tensors.clear()


class Function:
    """
    Superclass for all tensor operation types.
    """    
    def __init__(self, debug=False, debug_mode:Literal['fwd', 'bwd', 'both'] = 'bwd') -> None:
        self.ctx = ContextManager()
        self.debug = debug
        self.debug_mode = debug_mode
    
    def __call__(self, *args):
        return self.forward(*args)
        
    def forward(self, *args):
        # Call super().foward() to enable debugging
        if self.debug and (self.debug_mode=='fwd' or self.debug_mode=='both'):
            print(self.__class__.__name__+"_forward")
        return
    
    def backward(self, *args):
        # Call super().backward to enable debugging
        if self.debug and (self.debug_mode=='bwd' or self.debug_mode=='both'):
            print("")
            print(self.__class__.__name__+"_backward")
        if len(self.ctx.saved_tensors) == 0:
            raise RuntimeError("Called backward with no saved Tensors")
        return


def unbroadcast(tensor_data : np.ndarray, shape_to_match : Tuple[int, ...], to_keep=0) -> np.ndarray:
    """
    Helper function to handle broadcasting.
    """
    for _ in range(len(tensor_data.shape) - len(shape_to_match)):
        tensor_data = tensor_data.sum(axis=0)
    
    for i, dim in enumerate(shape_to_match):
        if dim == 1:
            tensor_data = tensor_data.sum(axis=i, keepdims=True)
    
    return tensor_data
