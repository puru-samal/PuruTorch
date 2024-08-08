from ..tensor import Tensor
from .. import functional as F
from typing import Union, Literal
from ._CTC import _CTCLoss

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
    '''
    Computes squared L2 Norm between each element in predictions and targets.  
    '''
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
    '''Computes the cross entropy loss between input logits and targets'''
    def __init__(self, reduction: Union[None, Literal['mean', 'sum']]='mean'):
        super().__init__()
        self.reduction = reduction

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        RCE = F.SoftmaxCrossEntropy()(predictions, targets, self.reduction)
        return RCE


class CTCLoss(Loss):
    '''An implementation of the Connectionist Temporal Classification loss'''
    def __init__(self, BLANK:int=0, reduction: Union[None, Literal['mean', 'sum']]='mean'):
        super().__init__()
        self.BLANK = BLANK
        self.reduction = reduction
    
    def __call__(self, logits:Tensor, target:Tensor, input_lengths:Tensor, target_lengths:Tensor) -> Tensor:
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits:Tensor, target:Tensor, input_lengths:Tensor, target_lengths:Tensor) -> Tensor:
        RCTC = _CTCLoss()(logits, target, input_lengths, target_lengths, self.BLANK, self.reduction)
        return RCTC