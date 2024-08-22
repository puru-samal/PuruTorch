from ..nn.module import Module
from ..nn import Linear, Identity
from ..tensor import Tensor
from typing import List, Optional

class MLP(Module):
    """
    A MLP model build with PuruTorch library.
    Network output and parameters are compared with
    equivalent PyTorch model to verify correctness.
    """    
    def __init__(self, dims: List[int], act_fn:Optional[Module] = None):
        """
        Args:
            dims (List[int]): A list of network dims. First elem is input dim, last elem is output dim, 
                              intermediate elems are hidden dims.
            act_fn (Optional[Module], optional): Activation function. If None then Identity() is used.
        """        
        super().__init__()
        self.layers = [Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]
        self.act_fn = Identity() if act_fn is None else act_fn

    def forward(self, x : Tensor) -> Tensor:
        for (i, layer) in enumerate(self.layers):
            y = layer.forward(x if i == 0 else y)
            y = self.act_fn(y)
        return y