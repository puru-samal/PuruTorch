import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import ReLU
from PuruTorch.models import MLP
import numpy as np
from helpers import *

class pyt_MLP(torch.nn.Module):
    '''Reference model to test correctness'''
    def __init__(self, dims: List[int], act_fn:Optional[torch.nn.Module] = None) -> None:
        super(pyt_MLP, self).__init__()
    
        self.layers = [torch.nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))]
        self.act_fn    = torch.nn.Identity() if act_fn is None else act_fn

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        for (i, layer) in enumerate(self.layers):
            y = layer.forward(x if i == 0 else y)
            y = self.act_fn(y)
        return y

test_dims = [
                [4, 8, 4],
                [4, 6, 6, 2],
                [6, 4, 8, 4, 2],
                [4, 6, 8, 12, 12, 8, 2]
            ]

def test_mlp():

    for i in range(len(test_dims)):
        print(f"** test{i+1}:", end=' ')

        print(f"dims: {[(test_dims[i][j-i],test_dims[i][j]) for j in range(1, len(test_dims[i]))]} ->", end=' ')
        #np
        batch_size = np.random.randint(1, 9)
        npx = np.random.uniform(0.0, 1.0, size=(batch_size, test_dims[i][0]))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_mlp = MLP(test_dims[i], act_fn=ReLU())
        pyt_mlp = pyt_MLP(test_dims[i], act_fn=torch.nn.ReLU())

        for (usr_layer, pyt_layer) in zip(usr_mlp.layers, pyt_mlp.layers):
            usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                                Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_mlp(usr_x)
        pyt_result = pyt_mlp(pyt_x)

        name = "mlp_forward"
        out_cmp = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = "mlp_backward"
        weight_cmp = True
        for (j, (usr_layer, pyt_layer)) in enumerate(zip(usr_mlp.layers, pyt_mlp.layers)):
            cmp2 = (cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'type',      name+f"{j}: dW")
                and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'shape',     name+f"{j}: dW")
                and cmp_usr_pyt_tensor(usr_layer.W.grad, pyt_layer.weight.grad, 'closeness', name+f"{j}: dW"))

            cmp3 = (cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'type',      name+f"{j}: db")
                and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'shape',     name+f"{j}: db")
                and cmp_usr_pyt_tensor(usr_layer.b.grad, pyt_layer.bias.grad, 'closeness', name+f"{j}: db"))
            
            weight_cmp = weight_cmp and cmp2 and cmp3

        x_cmp = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad,    'type',      name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (out_cmp and weight_cmp and x_cmp):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True