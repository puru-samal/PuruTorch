import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Linear
from PuruTorch.nn.activation import *
import torch
import numpy as np
from helpers import *

def test_linear_w_act(act_str):
    
    if act_str == "Identity":
        usr_act_fn = Identity()
        pyt_act_fn = torch.nn.Identity()
    elif act_str == "Sigmoid":
        usr_act_fn = Sigmoid()
        pyt_act_fn = torch.nn.Sigmoid()
    elif act_str == "ReLU":
        usr_act_fn = ReLU()
        pyt_act_fn = torch.nn.ReLU()
    elif act_str == "Tanh":
        usr_act_fn = Tanh()
        pyt_act_fn = torch.nn.Tanh()
    elif act_str == "Softmax":
        usr_act_fn = Softmax()
        pyt_act_fn = torch.nn.Softmax(dim=-1)
    else:
        print(f"Invalid activation: {act_str}")
        return False
    
    for i in range(6, 9):
        print(f"** {act_str}: test{i-5}:", end=' ')    
        #np
        npx = np.random.uniform(0.0, 1.0, size=(5, i))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_layer = Linear(i, 5)
        pyt_layer = torch.nn.Linear(i, 5)

        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                               Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_act_fn(usr_layer(usr_x))
        pyt_result = pyt_act_fn(pyt_layer(pyt_x))

        name = f"linear_forward w/ {act_str}"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
        
        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = f"linear_backward w/ {act_str}"
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


def test_identity():
    return test_linear_w_act("Identity")

def test_sigmoid():
    return test_linear_w_act("Sigmoid")

def test_relu():
    return test_linear_w_act("ReLU")

def test_tanh():
    return test_linear_w_act("Tanh")

def test_softmax():
    return test_linear_w_act("Softmax")
