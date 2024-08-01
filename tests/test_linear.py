import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Linear
import torch
import numpy as np
from helpers import *

def test_linear_forward():
    for i in range(6, 9):
        print(f"** test{i-5}:", end=' ')
        
        #np
        npx = np.random.uniform(0.0, 1.0, size=(i,))
        npW = np.random.uniform(-1.0, 1.0, size=(2,i))
        npb = np.random.uniform(-1.0, 1.0, size=(2,))

        usr_x = Tensor.tensor(npx)
        pyt_x = torch.FloatTensor(npx)

        usr_layer = Linear(i, 2)
        pyt_layer = torch.nn.Linear(i, 2)

        usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                               Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_layer(usr_x)
        pyt_result = pyt_layer(pyt_x)

        name = "linear_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        if not (cmp1):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_linear_backward():
    print(f"** test{1}:", end=' ')
    

    #np
    npx = np.random.uniform(0.0, 1.0, size=(9,))
    npW = np.random.uniform(-1.0, 1.0, size=(4,9))
    npb = np.random.uniform(-1.0, 1.0, size=(4,))

    usr_x = Tensor.tensor(npx, requires_grad=True)
    pyt_x = torch.FloatTensor(npx).requires_grad_()

    usr_layer = Linear(9, 4)
    pyt_layer = torch.nn.Linear(9, 4)

    usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                            Tensor.tensor(pyt_layer.bias.detach().numpy()))

    #forward
    usr_result = usr_layer(usr_x)
    pyt_result = pyt_layer(pyt_x)

    #backward
    usr_result.sum().backward()
    pyt_result.sum().backward()

    name = "linear_backward"
    cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
        and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
        and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
    
    cmp2 = (cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'type',      name+": dW")
        and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'shape',     name+": dW")
        and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'closeness', name+": dW"))
    
    cmp3 = (cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'type',      name+": db")
        and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'shape',     name+": db")
        and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'closeness', name+": db"))
    
    if not (cmp1 and cmp2 and cmp3):
        print("failed!")
        return False
    else:
        print("passed!")
        return True
        

    




