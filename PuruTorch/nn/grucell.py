from .module import Module
from ..tensor import Tensor
from .rnncell import RNNCell
from .activation import * 

class GRUCell(Module):
    """
    A gated recurrent unit (GRU) cell.
    """
    def __init__(self, in_features:int, hidden_features:int) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        self.r_cell = RNNCell(self.in_features, self.hidden_features, act_fn=Sigmoid())
        self.z_cell = RNNCell(self.in_features, self.hidden_features, act_fn=Sigmoid())
        self.n_cell = RNNCell(self.in_features, self.hidden_features, act_fn=Tanh())


    def init_weights(self, Wrx:Tensor, Wzx:Tensor, Wnx:Tensor, Wrh:Tensor, 
                     Wzh:Tensor, Wnh:Tensor, brx:Tensor, bzx:Tensor, 
                     bnx:Tensor, brh:Tensor, bzh:Tensor, bnh:Tensor) -> None:
        self.r_cell.init_weights(Wrx, Wrh, brx, brh)
        self.z_cell.init_weights(Wzx, Wzh, bzx, bzh)
        self.n_cell.init_weights(Wnx, Wnh, bnx, bnh)


    def __call__(self, x:Tensor, h_prev_t:Tensor) -> Tensor:
        """
        GRU cell forward.
        
        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        return self.forward(x, h_prev_t)


    def forward(self, x:Tensor, h_prev_t:Tensor) -> Tensor:
        """
        GRU cell forward.
        
        -----
        Using RNNCells to compute the transformations below:
        r = σ (W_ir x + b_ir + W_hr h + b_hr)
        z = σ (W_iz x + b_iz + W_hz h + b_hz)
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        -----
        
        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        batch_size, _ = x.shape
        if h_prev_t is None:
            h_prev_t = Tensor.zeros((batch_size, self.hidden_features))
            
        r = self.r_cell(x, h_prev_t)
        z = self.z_cell(x, h_prev_t)
        n = self.n_cell(x, h_prev_t, scale_hidden=r)
        h_t = (1. - z) * n + z * h_prev_t
        return h_t


        