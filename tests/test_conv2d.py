import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn.conv2d import *
import numpy as np
from helpers import *
import os

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

