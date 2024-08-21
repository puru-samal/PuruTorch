from .module import Module
from ..tensor import Tensor
from .. import functional as F

class Dropout(Module):
    ''' Randomly zeros out elements in an input tensor'''
    def __init__(self, p:float=0.5) -> None:
        """
        Args:
            p (float, optional): Probalitily of an element to be zeroed. Defaults to 0.5.
        """  
        super().__init__()
        self.p = p
    
    def forward(self, x:Tensor) -> Tensor:
        return F.dropout()(x, self.p, self.mode == 'train')
    

class Dropout1D(Module):
    ''' Randomly zeros out entire channels in a 1D feature map '''
    def __init__(self, p:float=0.5) -> None:
        """
        Args:
            p (float, optional): Probalitily of an element to be zeroed. Defaults to 0.5.
        """  
        super().__init__()
        self.p = p
    
    def forward(self, x:Tensor) -> Tensor:
        return F.dropout1d()(x, self.p, self.mode == 'train')
    

class Dropout2D(Module):
    ''' Randomly zeros out entire channels in a 2D feature map '''
    def __init__(self, p:float=0.5) -> None:
        """
        Args:
            p (float, optional): Probalitily of an element to be zeroed. Defaults to 0.5.
        """        
        super().__init__()
        self.p = p
    
    def forward(self, x:Tensor) -> Tensor:
        return F.dropout2d()(x, self.p, self.mode == 'train')
    
    
