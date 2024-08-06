import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn.activation import *
from PuruTorch.models import MLP
from PuruTorch.nn.loss import MSELoss
from PuruTorch.optim import SGD, Adam, AdamW
import numpy as np
from helpers import *
from torchviz import make_dot

class pyt_MLP(torch.nn.Module):
    '''Reference model to test correctness'''
    def __init__(self, dims: List[int], act_fn:Optional[torch.nn.Module] = None) -> None:
        super(pyt_MLP, self).__init__()
    
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(dims[i-1], dims[i]) for i in range(1, len(dims))])
        self.act_fn = torch.nn.Identity() if act_fn is None else act_fn

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

def test_optim(optim_type):
    for i in range(len(test_dims)):
        print(f"** test{i+1}:", end=' ')

        print(f"dims: {[(test_dims[i][j-i],test_dims[i][j]) for j in range(1, len(test_dims[i]))]} ->", end=' ')
        #np
        batch_size = np.random.randint(1, 9)
        npx = np.random.uniform(0.0, 1.0, size=(batch_size, test_dims[i][0]))
        npt = np.random.uniform(0.0, 1.0, size=(batch_size, test_dims[i][-1]))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_t = Tensor.tensor(npt)
        pyt_t = torch.FloatTensor(npt)

        usr_mlp = MLP(test_dims[i], act_fn=ReLU())
        pyt_mlp = pyt_MLP(test_dims[i], act_fn=torch.nn.ReLU())

        for (usr_layer, pyt_layer) in zip(usr_mlp.layers, pyt_mlp.layers):
            usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                                Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_mlp(usr_x)
        pyt_result = pyt_mlp(pyt_x)
        usr_criterion = MSELoss(reduction='mean')
        pyt_criterion = torch.nn.MSELoss(reduction='mean')
        usr_loss = usr_criterion(usr_result, usr_t)
        pyt_loss = pyt_criterion(pyt_result, pyt_t)
        if optim_type == 'sgd':
            usr_optim = SGD(usr_mlp.parameters(), lr=0.001, momentum=0.0)
            pyt_optim = torch.optim.SGD(pyt_mlp.parameters(), lr=0.001, momentum=0.0)
        elif optim_type == 'adam':
            usr_optim = Adam(usr_mlp.parameters(), lr=0.001, betas=[0.9, 0.99])
            pyt_optim = torch.optim.Adam(pyt_mlp.parameters(), lr=0.001, betas=(0.9, 0.999))
        elif optim_type == 'adamW':
            usr_optim = AdamW(usr_mlp.parameters(), lr=0.001, betas=[0.9, 0.99], weight_decay=0.01)
            pyt_optim = torch.optim.AdamW(pyt_mlp.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)


        name = "mlp_mse_forward"
        out_cmp = (cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'type',      name+": loss")
               and cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'shape',     name+": loss")
               and cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'closeness', name+": loss"))

        #backward
        usr_loss.backward()
        pyt_loss.backward()

        name = "mlp_mse_backward"
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
        
        #optim step
        usr_optim.step()
        pyt_optim.step()

        param_cmp = True
        for (j, (usr_layer, pyt_layer)) in enumerate(zip(usr_mlp.layers, pyt_mlp.layers)):
            cmp2 = (cmp_usr_pyt_tensor(usr_layer.W, pyt_layer.weight, 'type',      name+f"{j}: W")
                and cmp_usr_pyt_tensor(usr_layer.W, pyt_layer.weight, 'shape',     name+f"{j}: W")
                and cmp_usr_pyt_tensor(usr_layer.W, pyt_layer.weight, 'closeness', name+f"{j}: W"))

            cmp3 = (cmp_usr_pyt_tensor(usr_layer.b, pyt_layer.bias, 'type',      name+f"{j}: b")
                and cmp_usr_pyt_tensor(usr_layer.b, pyt_layer.bias, 'shape',     name+f"{j}: b")
                and cmp_usr_pyt_tensor(usr_layer.b, pyt_layer.bias, 'closeness', name+f"{j}: b"))
            
            param_cmp = param_cmp and cmp2 and cmp3
        
        usr_optim.zero_grad()
        pyt_optim.zero_grad()

        zero_grad_cmp = True
        for (j, (usr_layer, pyt_layer)) in enumerate(zip(usr_mlp.layers, pyt_mlp.layers)):
            cmp2 = usr_layer.W.grad is None and pyt_layer.weight.grad is None
            cmp3 = usr_layer.b.grad is None and pyt_layer.bias.grad is None
            zero_grad_cmp = zero_grad_cmp and cmp2 and cmp3

        
        if not (out_cmp and weight_cmp and x_cmp and param_cmp and zero_grad_cmp):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True

def test_optim_sgd():
    return test_optim('sgd')

def test_optim_adam():
    return test_optim('adam')

def test_optim_adamW():
    return test_optim('adamW')