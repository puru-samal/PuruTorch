from ..nn.module import Module
from ..nn import GRUCell, Linear
from ..nn.activation import *
from typing import List, Optional

class GRUClassifier(Module):
    """
    A GRU Classifier Model build with PuruTorch library.
    Applies a  multi-layer Elman RNN with non-linearity to an input sequence,
    Output from last layer, last timestep is passed to a linear classifier.
    Network output and parameters are compared with
    equivalent PyTorch model to verify correctness.
    """ 

    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int, num_hidden_layers:int=2, act_fn:Module=Tanh()):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.grus :List[GRUCell] = [GRUCell(in_dim if i == 0 else hidden_dim, 
                                    hidden_dim) for i in range(self.num_hidden_layers)]
        
        self.proj = Linear(hidden_dim, out_dim)
    
    def __call__(self, x:Tensor, h_0:Optional[Tensor]=None) -> Tensor:
        """
        Forward computation over multiple layers and multiple time-steps.

        Args:
            x         (Tensor): Input Tensor of shape (batch_size, seq_len, input_dim)
            h_0 (List[Tensor]): List of length num-layers containing 
                              hidden_state of shape (batch_size, hidden_dim)

        Returns:
            output       (Tensor): Output of last layer across all timesteps, of shape (batch_size, seq_len, hidden_dim)
            h_n          (Tensor): Final hidden state of hidden sequence, of shape (num_layers, batch_size, hidden_dim)
        """
        return self.forward(x, h_0)
        

    def forward(self, x:Tensor, h_0:Optional[List[Tensor]]=None) -> Tensor:
        """
        Forward computation over multiple layers and multiple time-steps.

        Args:
            x         (Tensor): Input Tensor of shape (batch_size, seq_len, input_dim)
            h_0 (List[Tensor]): List of length num-layers containing 
                              hidden_state of shape (batch_size, hidden_dim)

        Returns:
            output       (Tensor): Output of last layer across all timesteps, of shape (batch_size, seq_len, hidden_dim)
            h_n          (Tensor): Final hidden state of hidden sequence, of shape (num_layers, batch_size, hidden_dim)
        """

        batch_size, seq_len, _ = x.shape
        hiddens = [] # (seq_len+1) * (num_layers) * (batch_size, hidden_dim)
        
        if h_0 is  None:
            hidden = [Tensor.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_hidden_layers)]
        else:
            hidden = h_0
        
        hiddens.append(hidden)
        
        for t in range(seq_len):
            hidden[0] = self.grus[0](x[:, t, :], hiddens[-1][0])
            for L in range(1, self.num_hidden_layers):
                hidden[L] = self.grus[L](hidden[L - 1], hiddens[-1][L])
            hiddens.append(hidden)
        
        logits = self.proj(hiddens[-1][-1])
        return logits



        

