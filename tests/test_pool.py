import torch
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import MaxPool1D, MeanPool1D, MaxPool2D, MeanPool2D
import numpy as np
from helpers import *

def test_maxpool1d():
    seeds = [11485, 11685, 11785]
    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')
        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)

        npx = np.random.randn(batch, in_c, width)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_mp = MaxPool1D(kernel_size=3, stride=1)
        pyt_mp = torch.nn.MaxPool1d(kernel_size=3, stride=1)

        usr_result = usr_mp(usr_x)
        pyt_result = pyt_mp(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        name = "maxpool1d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        name = "maxpool1d_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_meanpool1d():
    seeds = [11485, 11685, 11785]
    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')
        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)

        npx = np.random.randn(batch, in_c, width)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()
                
        usr_mp = MeanPool1D(kernel_size=3, stride=1)
        pyt_mp = torch.nn.AvgPool1d(kernel_size=3, stride=1)

        usr_result = usr_mp(usr_x)
        pyt_result = pyt_mp(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        name = "meanpool1d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        name = "meanpool1d_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_maxpool2d():
    seeds = [11485, 11685, 11785]
    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')
        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)

        npx = np.random.randn(batch, in_c, width, width)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_mp = MaxPool2D(kernel_size=3, stride=1)
        pyt_mp = torch.nn.MaxPool2d(kernel_size=3, stride=1)

        usr_result = usr_mp(usr_x)
        pyt_result = pyt_mp(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        name = "maxpool2d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        name = "maxpool2d_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True

def test_meanpool2d():
    seeds = [11485, 11685, 11785]
    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')
        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)

        npx = np.random.randn(batch, in_c, width, width)

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()
                
        usr_mp = MeanPool2D(kernel_size=3, stride=1)
        pyt_mp = torch.nn.AvgPool2d(kernel_size=3, stride=1)

        usr_result = usr_mp(usr_x)
        pyt_result = pyt_mp(pyt_x)
        usr_x.sum().backward()
        pyt_x.sum().backward()

        name = "meanpool2d_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        name = "meanpool2d_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True
