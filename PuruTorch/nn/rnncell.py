from .module import Module
from ..tensor import Tensor, Parameter
from .linear import Linear
from .activation import * 
from typing import Union, Optional

class RNNCell(Module):
    """
    An Elman RNN cell with non-linearity.
    h_t = tanh(W_ih x_t + b_ih + W_hh h_t−1 + b_hh) 
    """
    def __init__(self, in_features:int, hidden_features:int, act_fn:Module=Tanh()) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.ih = Linear(self.in_features, self.hidden_features)
        self.hh = Linear(self.hidden_features, self.hidden_features)
        self.act_fn = act_fn

    def init_weights(self, W_ih: Tensor, W_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> None:
        self.ih.init_weights(W_ih, b_ih)
        self.hh.init_weights(W_hh, b_hh)

    def __call__(self, x:Tensor, h_prev_t:Tensor, scale_hidden:Union[float, Tensor]=1.0) -> Tensor:
        """
        RNN Cell forward (single time step).
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer
        
        scale_hidden: (float, optional)
            an optional scale factor to hidden. useful to create GRUCells from RNNCells

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """
        return self.forward(x, h_prev_t, scale_hidden)
    

    def forward(self, x:Tensor, h_prev_t:Optional[Tensor]=None, scale_hidden:Union[float, Tensor]=1.0) -> Tensor:
        """
        RNN Cell forward (single time step).
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer
        
        scale_hidden: (float, optional)
            an optional scale factor to hidden. useful to create GRUCells from RNNCells

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """
        batch_size, _ = x.shape
        if h_prev_t is None:
            h_prev_t = Tensor.zeros((batch_size, self.hidden_features))

        ih = self.ih(x)
        hh = self.hh(h_prev_t)
        y = ih + scale_hidden * hh
        h_t = self.act_fn(y)
        return h_t


        