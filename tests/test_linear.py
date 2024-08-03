import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Linear
import torch
import numpy as np
from helpers import *

def test_linear():
    for i in range(6, 9):
        print(f"** test{i-5}:", end=' ')
        #np
        npx = np.random.uniform(0.0, 1.0, size=(i,))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_layer = Linear(i, 2)
        pyt_layer = torch.nn.Linear(i, 2)

        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                               Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_layer(usr_x)
        pyt_result = pyt_layer(pyt_x)

        name = "linear_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
        
        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = "linear_backward"
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

