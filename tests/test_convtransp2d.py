import torch
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import ConvTranspose2D
import numpy as np
from helpers import *
import os

def test_convtransp2d():
    data_path = os.getcwd() + '/tests/data'
    ref_data_path = os.path.join(data_path,'convtransp_ref_data')
    expected_res_list = np.load(os.path.join(ref_data_path, 'convTranspose_2d_res.npz'), allow_pickle=True)
    seeds = [11485, 11685, 11785]

    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')

        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)
        kernel, upsampling_factor = rint(1, 3), rint(1, 3)
        x = Tensor.tensor(np.random.randn(batch, in_c, width, width), requires_grad=True)

        usr_layer = ConvTranspose2D(in_c, out_c, kernel, upsampling_factor=upsampling_factor)

        np.random.seed(11785)
        usr_layer.init_weights(Tensor.tensor(np.random.normal(0.0, 1.0, (out_c, in_c, kernel, kernel))), 
                               Tensor.tensor(np.zeros(out_c)))

        forward_res = usr_layer(x)
        b, c, w, h = forward_res.shape
        delta   = np.random.randn(b, c, w, h)
        forward_res.backward(grad=Tensor.tensor(delta))
        backward_res = x.grad 

        fwd_exp_res = torch.from_numpy(expected_res_list['conv_forward_res_list'][i])
        bwd_exp_res = torch.from_numpy(expected_res_list['conv_backward_res_list'][i])

        name = "convtranspose2d"
        out_cmp = (cmp_usr_pyt_tensor(forward_res,  fwd_exp_res, 'type',      name+": fwd")
                and cmp_usr_pyt_tensor(forward_res, fwd_exp_res, 'shape',     name+": fwd")
                and cmp_usr_pyt_tensor(forward_res, fwd_exp_res, 'closeness', name+": fwd"))
        
        grad_cmp = (cmp_usr_pyt_tensor(backward_res, bwd_exp_res, 'type',      name+": bwd")
                and cmp_usr_pyt_tensor(backward_res, bwd_exp_res, 'shape',     name+": bwd")
                and cmp_usr_pyt_tensor(backward_res, bwd_exp_res, 'closeness', name+": bwd"))
        
        if not (out_cmp and grad_cmp):
                print("failed!")
                return False
        else:
            print("passed!")
    
    return True

