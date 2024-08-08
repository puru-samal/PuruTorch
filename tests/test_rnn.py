import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import RNNCell
from PuruTorch.models import RNNClassifier
from PuruTorch.nn.activation import *
import torch
from torch import nn
import numpy as np
from helpers import *

# Reference Pytorch RNN Model
class ReferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_layers=2):
        super(ReferenceModel, self).__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers=rnn_layers, bias=True, batch_first=True
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_h=None):
        out, hidden = self.rnn(x, init_h)
        out = self.output(out[:, -1, :])
        return out

def test_rnncell():
    np.random.seed(11785)
    for i in range(6, 9):
        print(f"** test{i-5}:", end=' ')

        batch_size = np.random.randint(1, 9)
        npx = np.random.uniform(-1.0, 1.0, size=(batch_size,i))

        usr_x = Tensor.tensor(npx, requires_grad=True)
        pyt_x = torch.FloatTensor(npx).requires_grad_()

        usr_rnncell = RNNCell(i, i-4, Tanh())
        pyt_rnncell = torch.nn.RNNCell(i, i-4)

        usr_rnncell.init_weights(Tensor.tensor(pyt_rnncell.weight_ih.detach().numpy()),
                                 Tensor.tensor(pyt_rnncell.weight_hh.detach().numpy()),
                                 Tensor.tensor(pyt_rnncell.bias_ih.detach().numpy()),
                                 Tensor.tensor(pyt_rnncell.bias_hh.detach().numpy()))

        #forward
        usr_result = usr_rnncell(usr_x, h_prev_t=None)
        pyt_result = pyt_rnncell(pyt_x, hx=None)

        name = "rncell_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))

        #backward
        usr_result.sum().backward()
        pyt_result.sum().backward()

        name = "rncell_backward"
        cmp2 = (cmp_usr_pyt_tensor(usr_rnncell.ih.W.grad,     pyt_rnncell.weight_ih.grad, 'type',      name+": dW_ih")
                and cmp_usr_pyt_tensor(usr_rnncell.ih.W.grad, pyt_rnncell.weight_ih.grad, 'shape',     name+": dW_ih")
                and cmp_usr_pyt_tensor(usr_rnncell.ih.W.grad, pyt_rnncell.weight_ih.grad, 'closeness', name+": dW_ih"))

        cmp3 = (cmp_usr_pyt_tensor(usr_rnncell.ih.b.grad,     pyt_rnncell.bias_ih.grad, 'type',      name+": db_ih")
                and cmp_usr_pyt_tensor(usr_rnncell.ih.b.grad, pyt_rnncell.bias_ih.grad, 'shape',     name+": db_ih")
                and cmp_usr_pyt_tensor(usr_rnncell.ih.b.grad, pyt_rnncell.bias_ih.grad, 'closeness', name+": db_ih"))

        cmp4 = (cmp_usr_pyt_tensor(usr_rnncell.hh.W.grad,     pyt_rnncell.weight_hh.grad, 'type',      name+": dW_hh")
                and cmp_usr_pyt_tensor(usr_rnncell.hh.W.grad, pyt_rnncell.weight_hh.grad, 'shape',     name+": dW_hh")
                and cmp_usr_pyt_tensor(usr_rnncell.hh.W.grad, pyt_rnncell.weight_hh.grad, 'closeness', name+": dW_hh"))

        cmp5 = (cmp_usr_pyt_tensor(usr_rnncell.hh.b.grad,     pyt_rnncell.bias_hh.grad, 'type',      name+": db_hh")
                and cmp_usr_pyt_tensor(usr_rnncell.hh.b.grad, pyt_rnncell.bias_hh.grad, 'shape',     name+": db_hh")
                and cmp_usr_pyt_tensor(usr_rnncell.hh.b.grad, pyt_rnncell.bias_hh.grad, 'closeness', name+": db_hh"))

        cmp6 = (cmp_usr_pyt_tensor(usr_x.grad,     pyt_x.grad, 'type',      name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+": dx")
                and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+": dx"))
        
        if not (cmp1 and cmp2 and cmp3 and cmp4 and cmp5 and cmp6):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True

def test_rnn_classifier():
        rnn_layers = 2
        batch_size = 5
        seq_len = 10
        input_size = 40
        hidden_size = 32
        output_size = 138 
        np.random.seed(11785)
        torch.manual_seed(11785)

        data_x = np.random.randn(batch_size, seq_len, input_size)

        # Initialize
        # Reference model
        rnn_model = ReferenceModel(input_size, 
                                hidden_size, 
                                output_size, 
                                rnn_layers=rnn_layers)
        model_state_dict = rnn_model.state_dict()


        #my model
        my_rnn_model = RNNClassifier(input_size, hidden_size, output_size, num_hidden_layers=rnn_layers)
        rnn_weights = [[
                Tensor.tensor(model_state_dict["rnn.weight_ih_l%d" % l].numpy()),
                Tensor.tensor(model_state_dict["rnn.weight_hh_l%d" % l].numpy()),
                Tensor.tensor(model_state_dict["rnn.bias_ih_l%d" % l].numpy()),
                Tensor.tensor(model_state_dict["rnn.bias_hh_l%d" % l].numpy()),
                ] for l in range(rnn_layers)]
        fc_weights = [
                        Tensor.tensor(model_state_dict["output.weight"].numpy()),
                        Tensor.tensor(model_state_dict["output.bias"].numpy()),
                        ]
        
        for i, rnncell in enumerate(my_rnn_model.rnns):
                rnncell.init_weights(*rnn_weights[i])

        my_rnn_model.proj.init_weights(*fc_weights)

        # Test forward pass
        # Reference model
        
        ref_init_h = nn.Parameter(
        torch.zeros(rnn_layers, batch_size, hidden_size, dtype=torch.float),
                        requires_grad=True)
        pyt_x = torch.FloatTensor(data_x).requires_grad_()
        pyt_result = rnn_model(pyt_x, ref_init_h)
        # My model
        usr_x = Tensor.tensor(data_x, requires_grad=True)
        usr_result = my_rnn_model(usr_x)

        name = "rnn_classifier_forward"
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name+": y")
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name+": y"))
        
        # Test backward pass
        # Reference model
        pyt_result.sum().backward()
        usr_result.sum().backward()
        grad_dict = {
            k: v.grad for k, v in zip(rnn_model.state_dict(), rnn_model.parameters())
        }

        name = "rnn_classifier_backward"
        cmp2 = (cmp_usr_pyt_tensor(my_rnn_model.proj.W.grad,     grad_dict["output.weight"], 'type',      name+": dW")
                and cmp_usr_pyt_tensor(my_rnn_model.proj.W.grad, grad_dict["output.weight"], 'shape',     name+": dW")
                and cmp_usr_pyt_tensor(my_rnn_model.proj.W.grad, grad_dict["output.weight"], 'closeness', name+": dW"))

        cmp3 = (cmp_usr_pyt_tensor(my_rnn_model.proj.b.grad,     grad_dict["output.bias"], 'type',      name+": dW")
                and cmp_usr_pyt_tensor(my_rnn_model.proj.b.grad, grad_dict["output.bias"], 'shape',     name+": dW")
                and cmp_usr_pyt_tensor(my_rnn_model.proj.b.grad, grad_dict["output.bias"], 'closeness', name+": dW"))
        
        cmp_rnn = True
        for i, rnncell in enumerate(my_rnn_model.rnns):
             
             cmp4 = (cmp_usr_pyt_tensor(rnncell.ih.W.grad, grad_dict["rnn.weight_ih_l%d" % i], 'type',      name+f": dW_ih{i}")
                and cmp_usr_pyt_tensor(rnncell.ih.W.grad,  grad_dict["rnn.weight_ih_l%d" % i], 'shape',     name+f": dW_ih{i}")
                and cmp_usr_pyt_tensor(rnncell.ih.W.grad,  grad_dict["rnn.weight_ih_l%d" % i], 'closeness', name+f": dW_ih{i}"))
             
             cmp5 = (cmp_usr_pyt_tensor(rnncell.ih.b.grad, grad_dict["rnn.bias_ih_l%d" % i], 'type',      name+f": db_ih{i}")
                and cmp_usr_pyt_tensor(rnncell.ih.b.grad,  grad_dict["rnn.bias_ih_l%d" % i], 'shape',     name+f": db_ih{i}")
                and cmp_usr_pyt_tensor(rnncell.ih.b.grad,  grad_dict["rnn.bias_ih_l%d" % i], 'closeness', name+f": db_ih{i}"))
             
             cmp6 = (cmp_usr_pyt_tensor(rnncell.hh.W.grad, grad_dict["rnn.weight_hh_l%d" % i], 'type',      name+f": dW_hh{i}")
                and cmp_usr_pyt_tensor(rnncell.hh.W.grad,  grad_dict["rnn.weight_hh_l%d" % i], 'shape',     name+f": dW_hh{i}")
                and cmp_usr_pyt_tensor(rnncell.hh.W.grad,  grad_dict["rnn.weight_hh_l%d" % i], 'closeness', name+f": dW_hh{i}"))
             
             cmp7 = (cmp_usr_pyt_tensor(rnncell.hh.b.grad, grad_dict["rnn.bias_hh_l%d" % i], 'type',      name+f": db_hh{i}")
                and cmp_usr_pyt_tensor(rnncell.hh.b.grad,  grad_dict["rnn.bias_hh_l%d" % i], 'shape',     name+f": db_hh{i}")
                and cmp_usr_pyt_tensor(rnncell.hh.b.grad,  grad_dict["rnn.bias_hh_l%d" % i], 'closeness', name+f": db_hh{i}"))
             
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
    