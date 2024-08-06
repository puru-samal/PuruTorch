from ..tensor import Tensor
from .. import functional as F
from typing import Union, Literal
import numpy as np

class Loss():
    """
    Superclass for Loss Functions
    """    
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, *args):
        raise NotImplementedError


class MSELoss(Loss):
    def __init__(self, reduction: Union[None, Literal['mean', 'sum']]=None):
        super().__init__()
        self.reduction = reduction
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        SE = (targets - predictions) ** 2 # N, C
        if self.reduction is None:
            MSE = SE
        elif self.reduction == 'sum':
            MSE = SE.sum()
        elif self.reduction == 'mean':
            MSE = SE.mean()
        return MSE

class CrossEntropyLoss(Loss):
    def __init__(self, reduction: Union[None, Literal['mean', 'sum']]='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        RCE = F.SoftmaxCrossEntropy()(predictions, targets, self.reduction)
        return RCE


class CTCLoss(Loss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        pass
        