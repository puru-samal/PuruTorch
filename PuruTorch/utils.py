
import numpy as np

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
    def __init__(self, debug=True) -> None:
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