from .module import Module
from ..tensor import Tensor, Parameter
from .. import functional as F

class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Identity()(x)

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Sigmoid()(x)
    
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Tanh()(x)
    
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.ReLU()(x)
    
class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.Softmax()(x)