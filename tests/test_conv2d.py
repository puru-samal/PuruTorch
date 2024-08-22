import torch
from torch import nn
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Conv2D
from PuruTorch.models import ResBlock
import numpy as np
from helpers import *
import os

class PytConvBn2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, padding:int=0) -> None:
        super(PytConvBn2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x : Tensor) -> Tensor:
        out = self.bn(self.conv(x))
        return out

class PytResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=3, padding=1):
        super(PytResBlock, self).__init__()
        self.cbn1 = PytConvBn2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cbn2 = PytConvBn2D(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        
        if stride != 1 or in_channels != out_channels or kernel_size != 1 or padding != 0:
            self.residual =  PytConvBn2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=1)
        else:
            self.residual =  nn.Identity()  
        
        self.act = nn.ReLU()
    
    def forward(self, x : Tensor) -> Tensor:
        identity = x

        out = self.act(self.cbn1(x))
        out = self.cbn2(out)
        out = self.residual(identity) + out

        out = self.act(out)
        return out

def test_conv2d():
    np.random.seed(11785)
    for i in range(3):
        
        print(f"** test{i+1}:", end=' ')
        
        # hyperparams
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        kernel, stride = rint(3, 7), rint(1, 10)
        batch, width = rint(1, 4), rint(60, 80)

        # init usr, pyt model
        pyt_layer = torch.nn.Conv2d(in_c, out_c, kernel, stride=stride)
        usr_layer = Conv2D(in_c, out_c, kernel, stride=stride)

        # pyt weights -> usr weights
        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                            Tensor.tensor(pyt_layer.bias.detach().numpy()))
        
        # construct inputs
        npx = np.random.randn(batch, in_c, width, width)
        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()
        
        # forward
        usr_result = usr_layer(usr_x)
        pyt_result = pyt_layer(pyt_x)

        name = "conv2d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = "conv2d_backward"
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


def test_resblock():
    print(f"** testing ResBlock:", end=' ')
    batch, in_c, width = 10, 24, 128
    out_c = 3
    npx = np.random.randn(batch, in_c, width, width)
    
    usr_x = Tensor.tensor(npx, requires_grad=True)
    pyt_x = torch.FloatTensor(npx).requires_grad_()

    usr_model = ResBlock(in_channels=in_c, out_channels=out_c, kernel_size=5, stride=5, padding=5//2)
    pyt_model = PytResBlock(in_channels=in_c, out_channels=out_c, kernel_size=5, stride=5, padding=5//2)

    # init weights
    usr_model.cbn1.conv.init_weights(Tensor.tensor(pyt_model.cbn1.conv.weight.detach().numpy()), 
                                     Tensor.tensor(pyt_model.cbn1.conv.bias.detach().numpy()))
    
    usr_model.cbn2.conv.init_weights(Tensor.tensor(pyt_model.cbn2.conv.weight.detach().numpy()), 
                                     Tensor.tensor(pyt_model.cbn2.conv.bias.detach().numpy()))
    
    usr_model.residual.conv.init_weights(Tensor.tensor(pyt_model.residual.conv.weight.detach().numpy()), 
                                         Tensor.tensor(pyt_model.residual.conv.bias.detach().numpy()))
    
    # forward
    usr_result = usr_model(usr_x)
    pyt_result = pyt_model(pyt_x)

    name = "resblock_forward"
    fwd_cmp = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
           and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
           and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
    
    #backward
    usr_result.sum().backward()
    pyt_result.sum().backward()

    name = "resblock_backward"
    bwd_cmp = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
           and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
           and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
    
    cmp1 = (cmp_usr_pyt_tensor(usr_model.cbn1.conv.W.grad, pyt_model.cbn1.conv.weight.grad, 'type',      name+": dW1")
        and cmp_usr_pyt_tensor(usr_model.cbn1.conv.W.grad, pyt_model.cbn1.conv.weight.grad, 'shape',     name+": dW1")
        and cmp_usr_pyt_tensor(usr_model.cbn1.conv.W.grad, pyt_model.cbn1.conv.weight.grad, 'closeness', name+": dW1"))
    

    cmp2 = (cmp_usr_pyt_tensor(usr_model.cbn2.conv.W.grad, pyt_model.cbn2.conv.weight.grad, 'type',      name+": dW2")
        and cmp_usr_pyt_tensor(usr_model.cbn2.conv.W.grad, pyt_model.cbn2.conv.weight.grad, 'shape',     name+": dW2")
        and cmp_usr_pyt_tensor(usr_model.cbn2.conv.W.grad, pyt_model.cbn2.conv.weight.grad, 'closeness', name+": dW2"))
    

    cmp3 = (cmp_usr_pyt_tensor(usr_model.residual.conv.W.grad, pyt_model.residual.conv.weight.grad, 'type',      name+": dW3")
        and cmp_usr_pyt_tensor(usr_model.residual.conv.W.grad, pyt_model.residual.conv.weight.grad, 'shape',     name+": dW3")
        and cmp_usr_pyt_tensor(usr_model.residual.conv.W.grad, pyt_model.residual.conv.weight.grad, 'closeness', name+": dW3"))


    cmp4 = (cmp_usr_pyt_tensor(usr_model.cbn1.conv.b.grad, pyt_model.cbn1.conv.bias.grad, 'type',      name+": db1")
        and cmp_usr_pyt_tensor(usr_model.cbn1.conv.b.grad, pyt_model.cbn1.conv.bias.grad, 'shape',     name+": db1")
        and cmp_usr_pyt_tensor(usr_model.cbn1.conv.b.grad, pyt_model.cbn1.conv.bias.grad, 'closeness', name+": db1"))
    

    cmp5 = (cmp_usr_pyt_tensor(usr_model.cbn2.conv.b.grad, pyt_model.cbn2.conv.bias.grad, 'type',      name+": db2")
        and cmp_usr_pyt_tensor(usr_model.cbn2.conv.b.grad, pyt_model.cbn2.conv.bias.grad, 'shape',     name+": db2")
        and cmp_usr_pyt_tensor(usr_model.cbn2.conv.b.grad, pyt_model.cbn2.conv.bias.grad, 'closeness', name+": db2"))
    

    cmp6 = (cmp_usr_pyt_tensor(usr_model.residual.conv.b.grad, pyt_model.residual.conv.bias.grad, 'type',      name+": db3")
        and cmp_usr_pyt_tensor(usr_model.residual.conv.b.grad, pyt_model.residual.conv.bias.grad, 'shape',     name+": db3")
        and cmp_usr_pyt_tensor(usr_model.residual.conv.b.grad, pyt_model.residual.conv.bias.grad, 'closeness', name+": db3"))
    
    bwd_cmp = bwd_cmp and cmp1 and cmp2 and cmp3 and cmp4 and cmp5 and cmp6

    if not (fwd_cmp and bwd_cmp):
        print("failed!")
        return False
    else:
        print("passed!")
        return True
    
