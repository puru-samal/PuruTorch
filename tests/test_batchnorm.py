import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn.linear import Linear
from PuruTorch.nn import activation
from PuruTorch.nn.batchnorm1d import *
from PuruTorch.nn.loss import *
import numpy as np
from helpers import *


def test_batchnorm1d():
    np.random.seed(11786)
    for i in range(6, 9):
        print(f"** test{i-5}:", end=' ')
        batch_size = np.random.randint(2, 9)
        npx = np.random.uniform(-1.0, 1.0, size=(batch_size, i))
        npt = np.random.uniform(1.0, 1.0, size=(batch_size, i-2))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_t = Tensor.tensor(npt)
        pyt_t = torch.FloatTensor(npt)

        usr_layer = Linear(i, i-2)
        usr_act   = activation.ReLU()
        usr_bn    = BatchNorm1d(i-2, eps=0.00001, momentum=0.1)
        usr_criterion = MSELoss(reduction='mean')

        usr_layer.train()
        usr_act.train()
        usr_bn.train()

        pyt_layer = torch.nn.Linear(i, i-2)
        pyt_act   = torch.nn.ReLU()
        pyt_bn    = torch.nn.BatchNorm1d(i-2, eps=0.00001, momentum=0.1)
        pyt_criterion = torch.nn.MSELoss(reduction='mean')

        pyt_layer.train()
        pyt_act.train()
        pyt_bn.train()

        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                                Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward-backward
        usr_result = usr_layer(usr_x)
        usr_result = usr_act(usr_result)
        usr_result = usr_bn(usr_result)
        usr_loss   = usr_criterion(usr_result, usr_t)
        usr_loss.backward()

        pyt_result = pyt_layer(pyt_x)
        pyt_result = pyt_act(pyt_result)
        pyt_result = pyt_bn(pyt_result)
        pyt_loss   = pyt_criterion(pyt_result, pyt_t)
        pyt_loss.backward()

        name = "batchnorm_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        name = "batchnorm_backward"
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



