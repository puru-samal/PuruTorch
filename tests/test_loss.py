import torch
from typing import List, Optional
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import ReLU, Softmax, MSELoss, CrossEntropyLoss, CTCLoss
from PuruTorch.models import MLP
import numpy as np
from helpers import *
import os


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

def test_loss_mse():
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
        
        if not (out_cmp and weight_cmp and x_cmp):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True

def test_loss_ce():
    np.random.seed(11785)
    for i in range(len(test_dims)):
        print(f"** test{i+1}:", end=' ')

        print(f"dims: {[(test_dims[i][j-i],test_dims[i][j]) for j in range(1, len(test_dims[i]))]} ->", end=' ')
        #np
        batch_size = np.random.randint(1, 9)
        npx = np.random.uniform(0.0, 1.0, size=(batch_size, test_dims[i][0]))
        npt = np.random.uniform(0.0, 1.0, size=(batch_size, test_dims[i][-1]))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_t = Softmax()(Tensor.tensor(npt))
        pyt_t = torch.FloatTensor(npt).softmax(dim=-1)

        usr_mlp = MLP(test_dims[i], act_fn=ReLU())
        pyt_mlp = pyt_MLP(test_dims[i], act_fn=torch.nn.ReLU())

        for (usr_layer, pyt_layer) in zip(usr_mlp.layers, pyt_mlp.layers):
            usr_layer.init_weights(Tensor.tensor(pyt_layer.weight.detach().numpy()), 
                                Tensor.tensor(pyt_layer.bias.detach().numpy()))

        #forward
        usr_result = usr_mlp(usr_x)
        pyt_result = pyt_mlp(pyt_x)
        usr_criterion = CrossEntropyLoss(reduction='mean')
        pyt_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        usr_loss = usr_criterion(usr_result, usr_t)
        pyt_loss = pyt_criterion(pyt_result, pyt_t)

        name = "mlp_ce_forward"
        out_cmp = (cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'type',      name+": loss")
               and cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'shape',     name+": loss")
               and cmp_usr_pyt_tensor(usr_loss, pyt_loss, 'closeness', name+": loss"))

        #backward
        usr_loss.backward()
        pyt_loss.backward()

        name = "mlp_ce_backward"
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

def test_loss_ctc():

    print(f"** test{1}:", end=' ')

    data_path = os.getcwd() + '/tests/data'
    ref_data_path = data_path + '/ctc_ref_data' 
    probs = np.load(os.path.join(data_path, "X.npy"))
    targets = np.load(os.path.join(data_path, "Y.npy"))
    input_lens = np.load(os.path.join(data_path, "X_lens.npy"))
    out_lens = np.load(os.path.join(data_path, "Y_lens.npy"))

    usr_logits = Tensor.tensor(probs, requires_grad=True)
    usr_target =  Tensor.tensor(targets)
    usr_input_lengths  =  Tensor.tensor(input_lens)
    usr_target_lengths =  Tensor.tensor(out_lens)

    usr_ctc_loss = CTCLoss()
    usr_loss = usr_ctc_loss(usr_logits, usr_target, usr_input_lengths, usr_target_lengths)
    ref_loss = torch.from_numpy(np.load(os.path.join(ref_data_path, "ref_loss.npy")))
    ref_dy = torch.from_numpy(np.load(os.path.join(ref_data_path, "ref_dy.npy")))

    name = "ctc_forward"
    out_cmp = (cmp_usr_pyt_tensor(usr_loss,  ref_loss, 'type',      name+": loss")
            and cmp_usr_pyt_tensor(usr_loss, ref_loss, 'shape',     name+": loss")
            and cmp_usr_pyt_tensor(usr_loss, ref_loss, 'closeness', name+": loss"))
    
    usr_loss.backward()
    grad_cmp = (cmp_usr_pyt_tensor(usr_logits.grad,  ref_dy, 'type',      name+": dy")
            and cmp_usr_pyt_tensor(usr_logits.grad,  ref_dy, 'shape',     name+": dy")
            and cmp_usr_pyt_tensor(usr_logits.grad, ref_dy, 'closeness', name+": dy"))
    
    if not (out_cmp and grad_cmp):
            print("failed!")
            return False
    else:
        print("passed!")
        return True
