import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Dropout, Dropout1D, Dropout2D
import numpy as np
from helpers import *

def test_dropout():
    fwd_pyt_means, fwd_usr_means = [], []

    for _ in range(100):
        npx = np.random.rand(100, 100)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_dropout = Dropout(p=0.5)
        pyt_dropout = torch.nn.Dropout(p=0.5)

        usr_dropout.train()
        pyt_dropout.train()

        usr_fwd_res = usr_dropout(usr_x)
        pyt_fwd_res = pyt_dropout(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        fwd_usr_means.append(np.mean(usr_fwd_res.data))
        fwd_pyt_means.append(np.mean(pyt_fwd_res.detach().numpy()))

        name = "dropout_backward"
        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1):
            print("failed!")
            return False
    
    print("Testing mean over 100 it/s: ", end="")
    if not np.allclose(np.mean(fwd_usr_means), np.mean(fwd_pyt_means), atol=0.01, rtol=0.001):
        print(f"usr mean: {np.mean(fwd_usr_means)}")
        print(f"pyt mean: {np.mean(fwd_pyt_means)}")
        print("failed!")
        return False
    else:
        print("passed!")
        return True


def test_dropout1d():
    fwd_pyt_means, fwd_usr_means = [], []
    np.random.seed(11786)
    for _ in range(100):
        npx = np.random.rand(5, 100, 100)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_dropout = Dropout1D(p=0.5)
        pyt_dropout = torch.nn.Dropout1d(p=0.5)

        usr_dropout.train()
        pyt_dropout.train()

        usr_fwd_res = usr_dropout(usr_x)
        pyt_fwd_res = pyt_dropout(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        fwd_usr_means.append(np.mean(usr_fwd_res.data))
        fwd_pyt_means.append(np.mean(pyt_fwd_res.detach().numpy()))

        name = "dropout_backward"
        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1):
            print("failed!")
            return False
    
    print("Testing mean over 100 it/s: ", end="")
    if not np.allclose(np.mean(fwd_usr_means), np.mean(fwd_pyt_means), atol=0.01, rtol=0.001):
        print(f"usr mean: {np.mean(fwd_usr_means)}")
        print(f"pyt mean: {np.mean(fwd_pyt_means)}")
        print("failed!")
        return False
    else:
        print("passed!")
        return True


def test_dropout2d():
    fwd_pyt_means, fwd_usr_means = [], []
    np.random.seed(11786)
    for _ in range(100):
        npx = np.random.rand(5, 3, 100, 100)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_dropout = Dropout2D(p=0.5)
        pyt_dropout = torch.nn.Dropout2d(p=0.5)

        usr_dropout.train()
        pyt_dropout.train()

        usr_fwd_res = usr_dropout(usr_x)
        pyt_fwd_res = pyt_dropout(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        fwd_usr_means.append(np.mean(usr_fwd_res.data))
        fwd_pyt_means.append(np.mean(pyt_fwd_res.detach().numpy()))

        name = "dropout_backward"
        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1):
            print("failed!")
            return False
    
    print("Testing mean over 100 it/s: ", end="")
    if not np.allclose(np.mean(fwd_usr_means), np.mean(fwd_pyt_means), atol=0.01, rtol=0.001):
        print(f"usr mean: {np.mean(fwd_usr_means)}")
        print(f"pyt mean: {np.mean(fwd_pyt_means)}")
        print("failed!")
        return False
    else:
        print("passed!")
        return True


