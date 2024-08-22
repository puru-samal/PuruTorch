from ..nn import Module, Identity, Tanh, ReLU, Sigmoid, Linear, Conv1D
from ..tensor import Tensor

class CNN(Module):
    """
    A simple convolutional neural network
    """
    def __init__(self, in_width=128, conv_dims=[24, 56, 28, 14], kernel_sizes=[5, 6, 2], strides=[1, 2, 2], 
                 linear_dim=10, activations=[Tanh(), ReLU(), Sigmoid()]):
        """
        conv_dims             : [int]     : List containing number of (input-output) channels for each conv layer
        kernel_sizes          : [int]     : List containing kernel width for each conv layer
        strides               : [int]     : List containing stride size for each conv layer
        linear_dim            : int       : Number of neurons in the linear layer
        activations           : [Module]  : List of objects corresponding to the activation fn for each conv layer

        len(conv_dims) == len(activations) + 1  == len(kernel_sizes) + 1 == len(strides) + 1
        """
        super().__init__()
        self.activations = activations
        self.conv_layers = [Conv1D(in_channels=conv_dims[i-1], out_channels=conv_dims[i], 
                              kernel_size=kernel_sizes[i-1], stride=strides[i-1], 
                              padding=kernel_sizes[i-1]//2) for i in range(1, len(conv_dims))]

        w_out = in_width
        for i in range(len(self.conv_layers)):
            w_out = (w_out + 2*(kernel_sizes[i]//2) -  kernel_sizes[i]) // strides[i] + 1

        self.linear = Linear(w_out * conv_dims[-1], linear_dim)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """
        for conv, act in zip(self.conv_layers, self.activations):
            A = conv(A)
            A = act(A)

        N = A.shape[0] # batch_size
        A = self.linear(A.reshape(N, -1))
        return A
