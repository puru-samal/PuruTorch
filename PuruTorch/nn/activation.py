from .module import Module
from ..tensor import Tensor, Parameter
from .. import functional as F

class Identity(Module):
    """
    Identity Activation Function
    """    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Identity()(x)


class Sigmoid(Module):
    """
    Sigmoid Activation Function
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Sigmoid()(x)


class Tanh(Module):
    """
    Tanh Activation Function
    """ 
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Tanh()(x)


class ReLU(Module):
    """
    ReLU Activation Function
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.ReLU()(x)

    
class Softmax(Module):
    """
    Softmax Activation Function
    """  
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.Softmax()(x)

            