from ..nn.module import Module
from ..nn import Linear
from ..nn.activation import *
from typing import List, Optional

class MLP(Module):
    def __init__(self, dims: List[int], act_fn:Optional[Module] = None):
        self.layers = [Linear(i-1, i) for i in range(1, len(dims))]
        self.act    = Identity() if act_fn is None else act_fn

    def forward(self, x : Tensor) -> Tensor:
        for (i, layer) in enumerate(self.layers):
            y = layer.forward(x if i == 0 else y)
        return y