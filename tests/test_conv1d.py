import torch
from torch import nn
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.models import CNN
from PuruTorch.nn import Conv1D
import numpy as np
from helpers import *

# Reference Pytorch CNN Model
class ReferenceModel(nn.Module):
    def __init__(self, in_width=128, conv_dims=[24, 56, 28, 14], kernel_sizes=[5, 6, 2], strides=[1, 2, 2], 
                 linear_dim=10, activations=[nn.Tanh(), nn.ReLU(), nn.Sigmoid()]):
        super(ReferenceModel, self).__init__()
        self.activations = activations

        self.conv_layers = [nn.Conv1d(in_channels=conv_dims[i-1], out_channels=conv_dims[i], 
                            kernel_size=kernel_sizes[i-1], stride=strides[i-1], 
                            padding=kernel_sizes[i-1]//2) for i in range(1, len(conv_dims))]
        
        w_out = in_width
        for i in range(len(self.conv_layers)):
            w_out = (w_out + 2*(kernel_sizes[i]//2) -  kernel_sizes[i]) // strides[i] + 1
        
        self.linear = nn.Linear(w_out * conv_dims[-1], linear_dim)

    def forward(self, A):
        for conv, act in zip(self.conv_layers, self.activations):
            A = act(conv(A))

        N = A.shape[0] # batch_size
        A = self.linear(A.reshape(N, -1))
        return A


def test_conv1d():
    np.random.seed(11785)
    for i in range(3):
        
        print(f"** test{i+1}:", end=' ')
        
        # hyperparams
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        kernel, stride = rint(1, 10), rint(1, 10)
        batch, width = rint(1, 4), rint(20, 300)

        # init usr, pyt model
        pyt_layer = torch.nn.Conv1d(in_c, out_c, kernel, stride=stride)
        usr_layer = Conv1D(in_c, out_c, kernel, stride=stride)

        # pyt weights -> usr weights
        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                            Tensor.tensor(pyt_layer.bias.detach().numpy()))
        
        # construct inputs
        npx = np.random.randn(batch, in_c, width)
        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()
        
        # forward
        usr_result = usr_layer(usr_x)
        pyt_result = pyt_layer(pyt_x)

        name = "conv1d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = "conv1_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        cmp3 = (cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'type',      name+": dW")
            and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'shape',     name+": dW")
            and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'closeness', name+": dW"))
        
        cmp4 = (cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'type',      name+": db")
            and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'shape',     name+": db")
            and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'closeness', name+": db"))
        
        if not (cmp1 and cmp2 and cmp3 and cmp4):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True

def test_cnn():

    print(f"** testing CNN:", end=' ')
    batch, in_c, width = 10, 24, 128
    npx = np.random.randn(batch, in_c, width)
    
    usr_x = Tensor.tensor(npx, requires_grad=True)
    pyt_x = torch.FloatTensor(npx).requires_grad_()

    usr_model = CNN()
    pyt_model = ReferenceModel()
    
    #init conv weights
    for (pyt_layer, usr_layer) in zip(pyt_model.conv_layers, usr_model.conv_layers):
        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                               Tensor.tensor(pyt_layer.bias.detach().numpy()))

    # init linear weights
    usr_model.linear.init_weights(Tensor.tensor(pyt_model.linear.weight.detach().numpy()), 
                                  Tensor.tensor(pyt_model.linear.bias.detach().numpy()))

    pyt_result = pyt_model(pyt_x)
    usr_result = usr_model(usr_x)

    name = "cnn_forward"
    fwd_cmp = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
           and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
           and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
        
    #backward
    usr_result.sum().backward()
    pyt_result.sum().backward()

    name = "cnn_backward"
    bwd_cmp = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
           and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
           and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
    
    for (pyt_layer, usr_layer) in zip(pyt_model.conv_layers, usr_model.conv_layers):
        w_cmp = (cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'type',      name+": dW")
            and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'shape',     name+": dW")
            and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'closeness', name+": dW"))
        
        b_cmp = (cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'type',      name+": db")
             and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'shape',     name+": db")
             and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'closeness', name+": db"))
        
        bwd_cmp = bwd_cmp and w_cmp and b_cmp
    
    lW_cmp = (cmp_usr_pyt_tensor(usr_model.linear.W.grad,  pyt_model.linear.weight.grad, 'type',      name+": dW")
           and cmp_usr_pyt_tensor(usr_model.linear.W.grad, pyt_model.linear.weight.grad, 'shape',     name+": dW")
           and cmp_usr_pyt_tensor(usr_model.linear.W.grad, pyt_model.linear.weight.grad, 'closeness', name+": dW"))
    
    lb_cmp = (cmp_usr_pyt_tensor(usr_model.linear.b.grad,  pyt_model.linear.bias.grad, 'type',      name+": dW")
           and cmp_usr_pyt_tensor(usr_model.linear.b.grad, pyt_model.linear.bias.grad, 'shape',     name+": dW")
           and cmp_usr_pyt_tensor(usr_model.linear.b.grad, pyt_model.linear.bias.grad, 'closeness', name+": dW"))

    bwd_cmp = bwd_cmp and lW_cmp and lb_cmp

    if not (fwd_cmp and bwd_cmp):
        print("failed!")
        return False
    else:
        print("passed!")
        return True