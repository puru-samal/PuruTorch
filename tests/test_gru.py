import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append("./")
from torch import nn
from PuruTorch import Tensor
from PuruTorch.nn import GRUCell
from PuruTorch.models import GRUClassifier
import torch
import numpy as np
from helpers import *

# Reference Pytorch GRU Model
class ReferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_layers=2):
        super(ReferenceModel, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=rnn_layers, bias=True, batch_first=True
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_h=None):
        out, hidden = self.gru(x, init_h)
        out = self.output(out[:, -1, :])
        return out

def test_grucell():
    np.random.seed(11785)
    for i in range(6, 9):
        print(f"** test{i-5}:", end=' ')

        batch_size = np.random.randint(1, 9)
        npx = np.random.uniform(-1.0, 1.0, size=(batch_size,i))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_grucell = GRUCell(i, i-4)
        pyt_grucell = torch.nn.GRUCell(i, i-4)

        W_ih, b_ih = pyt_grucell.weight_ih.detach().numpy(), pyt_grucell.bias_ih.detach().numpy()
        W_hh, b_hh = pyt_grucell.weight_hh.detach().numpy(), pyt_grucell.bias_hh.detach().numpy()

        Wrx, Wzx, Wnx = np.split(W_ih, 3, axis=0)
        Wrh, Wzh, Wnh = np.split(W_hh, 3, axis=0)
        brx, bzx, bnx = np.split(b_ih, 3, axis=0)
        brh, bzh, bnh = np.split(b_hh, 3, axis=0)

        usr_grucell.init_weights(Tensor.tensor(Wrx), Tensor.tensor(Wzx), Tensor.tensor(Wnx), 
                                 Tensor.tensor(Wrh), Tensor.tensor(Wzh), Tensor.tensor(Wnh),
                                 Tensor.tensor(brx), Tensor.tensor(bzx), Tensor.tensor(bnx), 
                                 Tensor.tensor(brh), Tensor.tensor(bzh), Tensor.tensor(bnh))

        #forward
        usr_result = usr_grucell(usr_x, h_prev_t=None)
        pyt_result = pyt_grucell(pyt_x, hx=None)

        name = "grucell_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        hidden_dim = i - 4
        name = "grucell_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.W.grad,     pyt_grucell.weight_ih.grad[:hidden_dim], 'type',      name+": dWrx")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.W.grad, pyt_grucell.weight_ih.grad[:hidden_dim], 'shape',     name+": dWrx")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.W.grad, pyt_grucell.weight_ih.grad[:hidden_dim], 'closeness', name+": dWrx"))
        
        cmp3 = (cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.W.grad,     pyt_grucell.weight_ih.grad[hidden_dim:2*hidden_dim], 'type',      name+": dWzx")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.W.grad, pyt_grucell.weight_ih.grad[hidden_dim:2*hidden_dim], 'shape',     name+": dWzx")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.W.grad, pyt_grucell.weight_ih.grad[hidden_dim:2*hidden_dim], 'closeness', name+": dWzx"))
        
        cmp4 = (cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.W.grad,     pyt_grucell.weight_ih.grad[2*hidden_dim:], 'type',      name+": dWnx")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.W.grad, pyt_grucell.weight_ih.grad[2*hidden_dim:], 'shape',     name+": dWnx")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.W.grad, pyt_grucell.weight_ih.grad[2*hidden_dim:], 'closeness', name+": dWnx"))
        
        cmp5 = (cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.b.grad,     pyt_grucell.bias_ih.grad[:hidden_dim], 'type',      name+": dbrx")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.b.grad, pyt_grucell.bias_ih.grad[:hidden_dim], 'shape',     name+": dbrx")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.ih.b.grad, pyt_grucell.bias_ih.grad[:hidden_dim], 'closeness', name+": dbrx"))
        
        cmp6 = (cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.b.grad,     pyt_grucell.bias_ih.grad[hidden_dim:2*hidden_dim], 'type',      name+": dbzx")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.b.grad, pyt_grucell.bias_ih.grad[hidden_dim:2*hidden_dim], 'shape',     name+": dbzx")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.ih.b.grad, pyt_grucell.bias_ih.grad[hidden_dim:2*hidden_dim], 'closeness', name+": dbzx"))
        
        cmp7 = (cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.b.grad,     pyt_grucell.bias_ih.grad[2*hidden_dim:], 'type',      name+": dbnx")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.b.grad, pyt_grucell.bias_ih.grad[2*hidden_dim:], 'shape',     name+": dbnx")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.ih.b.grad, pyt_grucell.bias_ih.grad[2*hidden_dim:], 'closeness', name+": dbnx"))

        cmp8 = (cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.W.grad,     pyt_grucell.weight_hh.grad[:hidden_dim], 'type',      name+": dWrh")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.W.grad, pyt_grucell.weight_hh.grad[:hidden_dim], 'shape',     name+": dWrh")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.W.grad, pyt_grucell.weight_hh.grad[:hidden_dim], 'closeness', name+": dWrh"))
        
        cmp9 = (cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.W.grad,     pyt_grucell.weight_hh.grad[hidden_dim:2*hidden_dim], 'type',      name+": dWzh")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.W.grad, pyt_grucell.weight_hh.grad[hidden_dim:2*hidden_dim], 'shape',     name+": dWzh")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.W.grad, pyt_grucell.weight_hh.grad[hidden_dim:2*hidden_dim], 'closeness', name+": dWzh"))
        
        cmp10 = (cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.W.grad,    pyt_grucell.weight_hh.grad[2*hidden_dim:], 'type',      name+": dWnh")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.W.grad, pyt_grucell.weight_hh.grad[2*hidden_dim:], 'shape',     name+": dWnh")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.W.grad, pyt_grucell.weight_hh.grad[2*hidden_dim:], 'closeness', name+": dWnh"))
        
        cmp11 = (cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.b.grad,    pyt_grucell.bias_hh.grad[:hidden_dim], 'type',      name+": dbrh")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.b.grad, pyt_grucell.bias_hh.grad[:hidden_dim], 'shape',     name+": dbrh")
                and cmp_usr_pyt_tensor(usr_grucell.r_cell.hh.b.grad, pyt_grucell.bias_hh.grad[:hidden_dim], 'closeness', name+": dbrh"))
        
        cmp12 = (cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.b.grad,    pyt_grucell.bias_hh.grad[hidden_dim:2*hidden_dim], 'type',      name+": dbzh")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.b.grad, pyt_grucell.bias_hh.grad[hidden_dim:2*hidden_dim], 'shape',     name+": dbzh")
                and cmp_usr_pyt_tensor(usr_grucell.z_cell.hh.b.grad, pyt_grucell.bias_hh.grad[hidden_dim:2*hidden_dim], 'closeness', name+": dbzh"))
        
        cmp13 = (cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.b.grad,    pyt_grucell.bias_hh.grad[2*hidden_dim:], 'type',      name+": dbnh")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.b.grad, pyt_grucell.bias_hh.grad[2*hidden_dim:], 'shape',     name+": dbnh")
                and cmp_usr_pyt_tensor(usr_grucell.n_cell.hh.b.grad, pyt_grucell.bias_hh.grad[2*hidden_dim:], 'closeness', name+": dbnh"))
        
        cmp14 = (cmp_usr_pyt_tensor(usr_x.grad,    pyt_x.grad, 'type',      name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2 and cmp3 and 
                cmp4 and cmp5 and cmp6 and 
                cmp7 and cmp8 and cmp9 and 
                cmp10 and cmp11 and cmp12 
                and cmp13 and cmp14):
            
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_gru_classifier():
        gru_layers = 1
        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 8
        output_size = 20 
        np.random.seed(11785)
        torch.manual_seed(11785)

        data_x = np.random.randn(batch_size, seq_len, input_size)

        # Initialize
        # Reference model
        gru_model = ReferenceModel(input_size, 
                                hidden_size, 
                                output_size, 
                                rnn_layers=gru_layers)
        model_state_dict = gru_model.state_dict()


        #my model
        my_gru_model = GRUClassifier(input_size, hidden_size, output_size, num_hidden_layers=gru_layers)
        gru_weights = [[
                        model_state_dict["gru.weight_ih_l%d" % l].numpy(),
                        model_state_dict["gru.weight_hh_l%d" % l].numpy(),
                        model_state_dict["gru.bias_ih_l%d" % l].numpy(),
                        model_state_dict["gru.bias_hh_l%d" % l].numpy(),
                ] for l in range(gru_layers)]
        fc_weights = [
                        Tensor.tensor(model_state_dict["output.weight"].numpy()),
                        Tensor.tensor(model_state_dict["output.bias"].numpy()),
                        ]
        
        for i, grucell in enumerate(my_gru_model.grus):
                Wrx, Wzx, Wnx = np.split(gru_weights[i][0], 3, axis=0)
                Wrh, Wzh, Wnh = np.split(gru_weights[i][1], 3, axis=0)
                brx, bzx, bnx = np.split(gru_weights[i][2], 3, axis=0)
                brh, bzh, bnh = np.split(gru_weights[i][3], 3, axis=0)

                grucell.init_weights(Tensor.tensor(Wrx), Tensor.tensor(Wzx), Tensor.tensor(Wnx), 
                                     Tensor.tensor(Wrh), Tensor.tensor(Wzh), Tensor.tensor(Wnh),
                                     Tensor.tensor(brx), Tensor.tensor(bzx), Tensor.tensor(bnx), 
                                     Tensor.tensor(brh), Tensor.tensor(bzh), Tensor.tensor(bnh))

        my_gru_model.proj.init_weights(*fc_weights)

        # Test forward pass
        # Reference model
        
        ref_init_h = nn.Parameter(
        torch.zeros(gru_layers, batch_size, hidden_size, dtype=torch.float), requires_grad=True)
        pyt_x = torch.FloatTensor(data_x).requires_grad_()
        pyt_result = gru_model(pyt_x, ref_init_h)
        # My model
        usr_x = Tensor.tensor(data_x, requires_grad=True)
        usr_result = my_gru_model(usr_x)

        name = "gru_classifier_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
        
        # Test backward pass
        # Reference model
        pyt_result.sum().backward()
        usr_result.sum().backward()
        grad_dict = {
            k: v.grad for k, v in zip(gru_model.state_dict(), gru_model.parameters())
        }

        name = "gru_classifier_backward"
        cmp2 = (cmp_usr_pyt_tensor(my_gru_model.proj.W.grad,     grad_dict["output.weight"], 'type',      name+": dW")
                and cmp_usr_pyt_tensor(my_gru_model.proj.W.grad, grad_dict["output.weight"], 'shape',     name+": dW")
                and cmp_usr_pyt_tensor(my_gru_model.proj.W.grad, grad_dict["output.weight"], 'closeness', name+": dW"))

        cmp3 = (cmp_usr_pyt_tensor(my_gru_model.proj.b.grad,     grad_dict["output.bias"], 'type',      name+": dW")
                and cmp_usr_pyt_tensor(my_gru_model.proj.b.grad, grad_dict["output.bias"], 'shape',     name+": dW")
                and cmp_usr_pyt_tensor(my_gru_model.proj.b.grad, grad_dict["output.bias"], 'closeness', name+": dW"))
        
        cmp_rnn = True
        for i, grucell in enumerate(my_gru_model.grus):
             
                cmp2 = (cmp_usr_pyt_tensor(grucell.r_cell.ih.W.grad,     grad_dict[f'gru.weight_ih_l{i}'][:hidden_size], 'type',      name+": dWrx")
                        and cmp_usr_pyt_tensor(grucell.r_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][:hidden_size], 'shape',     name+": dWrx")
                        and cmp_usr_pyt_tensor(grucell.r_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][:hidden_size], 'closeness', name+": dWrx"))
        
                cmp3 = (cmp_usr_pyt_tensor(grucell.z_cell.ih.W.grad,     grad_dict[f'gru.weight_ih_l{i}'][hidden_size:2*hidden_size], 'type',      name+": dWzx")
                        and cmp_usr_pyt_tensor(grucell.z_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][hidden_size:2*hidden_size], 'shape',     name+": dWzx")
                        and cmp_usr_pyt_tensor(grucell.z_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][hidden_size:2*hidden_size], 'closeness', name+": dWzx"))
                
                cmp4 = (cmp_usr_pyt_tensor(grucell.n_cell.ih.W.grad,     grad_dict[f'gru.weight_ih_l{i}'][2*hidden_size:], 'type',      name+": dWnx")
                        and cmp_usr_pyt_tensor(grucell.n_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][2*hidden_size:], 'shape',     name+": dWnx")
                        and cmp_usr_pyt_tensor(grucell.n_cell.ih.W.grad, grad_dict[f'gru.weight_ih_l{i}'][2*hidden_size:], 'closeness', name+": dWnx"))
                
                cmp5 = (cmp_usr_pyt_tensor(grucell.r_cell.ih.b.grad,     grad_dict[f'gru.bias_ih_l{i}'][:hidden_size], 'type',      name+": dbrx")
                        and cmp_usr_pyt_tensor(grucell.r_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][:hidden_size], 'shape',     name+": dbrx")
                        and cmp_usr_pyt_tensor(grucell.r_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][:hidden_size], 'closeness', name+": dbrx"))
                
                cmp6 = (cmp_usr_pyt_tensor(grucell.z_cell.ih.b.grad,     grad_dict[f'gru.bias_ih_l{i}'][hidden_size:2*hidden_size], 'type',      name+": dbzx")
                        and cmp_usr_pyt_tensor(grucell.z_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][hidden_size:2*hidden_size], 'shape',     name+": dbzx")
                        and cmp_usr_pyt_tensor(grucell.z_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][hidden_size:2*hidden_size], 'closeness', name+": dbzx"))
                
                cmp7 = (cmp_usr_pyt_tensor(grucell.n_cell.ih.b.grad,     grad_dict[f'gru.bias_ih_l{i}'][2*hidden_size:], 'type',      name+": dbnx")
                        and cmp_usr_pyt_tensor(grucell.n_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][2*hidden_size:], 'shape',     name+": dbnx")
                        and cmp_usr_pyt_tensor(grucell.n_cell.ih.b.grad, grad_dict[f'gru.bias_ih_l{i}'][2*hidden_size:], 'closeness', name+": dbnx"))

                cmp8 = (cmp_usr_pyt_tensor(grucell.r_cell.hh.W.grad,     grad_dict[f'gru.weight_hh_l{i}'][:hidden_size], 'type',      name+": dWrh")
                        and cmp_usr_pyt_tensor(grucell.r_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][:hidden_size], 'shape',     name+": dWrh")
                        and cmp_usr_pyt_tensor(grucell.r_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][:hidden_size], 'closeness', name+": dWrh"))
                
                cmp9 = (cmp_usr_pyt_tensor(grucell.z_cell.hh.W.grad,     grad_dict[f'gru.weight_hh_l{i}'][hidden_size:2*hidden_size], 'type',      name+": dWzh")
                        and cmp_usr_pyt_tensor(grucell.z_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][hidden_size:2*hidden_size], 'shape',     name+": dWzh")
                        and cmp_usr_pyt_tensor(grucell.z_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][hidden_size:2*hidden_size], 'closeness', name+": dWzh"))
                
                cmp10 = (cmp_usr_pyt_tensor(grucell.n_cell.hh.W.grad,    grad_dict[f'gru.weight_hh_l{i}'][2*hidden_size:], 'type',      name+": dWnh")
                        and cmp_usr_pyt_tensor(grucell.n_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][2*hidden_size:], 'shape',     name+": dWnh")
                        and cmp_usr_pyt_tensor(grucell.n_cell.hh.W.grad, grad_dict[f'gru.weight_hh_l{i}'][2*hidden_size:], 'closeness', name+": dWnh"))
                
                cmp11 = (cmp_usr_pyt_tensor(grucell.r_cell.hh.b.grad,    grad_dict[f'gru.bias_hh_l{i}'][:hidden_size], 'type',      name+": dbrh")
                        and cmp_usr_pyt_tensor(grucell.r_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][:hidden_size], 'shape',     name+": dbrh")
                        and cmp_usr_pyt_tensor(grucell.r_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][:hidden_size], 'closeness', name+": dbrh"))
                
                cmp12 = (cmp_usr_pyt_tensor(grucell.z_cell.hh.b.grad,    grad_dict[f'gru.bias_hh_l{i}'][hidden_size:2*hidden_size], 'type',      name+": dbzh")
                        and cmp_usr_pyt_tensor(grucell.z_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][hidden_size:2*hidden_size], 'shape',     name+": dbzh")
                        and cmp_usr_pyt_tensor(grucell.z_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][hidden_size:2*hidden_size], 'closeness', name+": dbzh"))
                
                cmp13 = (cmp_usr_pyt_tensor(grucell.n_cell.hh.b.grad,    grad_dict[f'gru.bias_hh_l{i}'][2*hidden_size:], 'type',      name+": dbnh")
                        and cmp_usr_pyt_tensor(grucell.n_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][2*hidden_size:], 'shape',     name+": dbnh")
                        and cmp_usr_pyt_tensor(grucell.n_cell.hh.b.grad, grad_dict[f'gru.bias_hh_l{i}'][2*hidden_size:], 'closeness', name+": dbnh"))
                
                cmp_rnn = cmp_rnn and cmp4 and cmp5 and cmp6 and cmp7
        
        cmp_x = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))

        if not (cmp1 and cmp2 and cmp3 and cmp_rnn):
            print("failed!")
            return False
        else:
            print("passed!")
            return True
 
test_gru_classifier()