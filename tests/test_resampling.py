import torch
import sys
sys.path.append("./")
from PuruTorch import Tensor
from PuruTorch.nn import Upsample1D, Downsample1D, Upsample2D, Downsample2D
import numpy as np
from helpers import *
import os

def test_upsampling1d():
    data_path = os.getcwd() + '/tests/data'
    ref_data_path = os.path.join(data_path,'resampling_ref_data')
    expected_res_list = np.load(os.path.join(ref_data_path, 'upsample_1d_res.npz'), allow_pickle=True)
    seeds = [11333, 11235, 11785]

    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')

        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        batch, width = rint(1, 4), rint(20, 300)
        kernel, upsampling_factor = rint(1, 10), rint(1, 10)
        x = Tensor.tensor(np.random.randn(batch, in_c, width), requires_grad=True)

        upsampling1d = Upsample1D()
        forward_res = upsampling1d(x, upsampling_factor)
        forward_res.backward(grad=Tensor.tensor(forward_res.data.copy()))
        backward_res = x.grad 

        fwd_exp_res = torch.from_numpy(expected_res_list['forward_res_list'][i])
        bwd_exp_res = torch.from_numpy(expected_res_list['backward_res_list'][i])

        name = "upsample1d"
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



def test_downsampling1d():
    data_path = os.getcwd() + '/tests/data'
    ref_data_path = os.path.join(data_path,'resampling_ref_data')
    expected_res_list = np.load(os.path.join(ref_data_path, 'downsample_1d_res.npz'), allow_pickle=True)
    seeds = [11333, 11235, 11785]

    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')

        np.random.seed(seed)
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        batch, width = rint(1, 4), rint(20, 300)
        kernel, downsampling_factor = rint(1, 10), rint(1, 10)
        x = Tensor.tensor(np.random.randn(batch, in_c, width), requires_grad=True)

        downsample1d = Downsample1D()
        forward_res = downsample1d(x, downsampling_factor)
        forward_res.backward(grad=Tensor.tensor(forward_res.data.copy()))
        backward_res = x.grad 

        fwd_exp_res = torch.from_numpy(expected_res_list['forward_res_list'][i])
        bwd_exp_res = torch.from_numpy(expected_res_list['backward_res_list'][i])

        name = "downsample1d"
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



def test_upsampling2d():
    data_path = os.getcwd() + '/tests/data'
    ref_data_path = os.path.join(data_path,'resampling_ref_data')
    expected_res_list = np.load(os.path.join(ref_data_path, 'upsample_2d_res.npz'), allow_pickle=True)
    seeds = [11333, 11235, 11785]

    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')

        np.random.seed(seed)
        rint = np.random.randint
        in_c = np.random.randint(5, 15)
        out_c = np.random.randint(5, 15)
        kernel = np.random.randint(3, 7)

        width = np.random.randint(60, 80)
        batch = np.random.randint(1, 4)
        upsampling_factor = rint(1, 10)
        x = Tensor.tensor(np.random.randn(batch, in_c, width, width), requires_grad=True)

        upsampling2d = Upsample2D()
        forward_res = upsampling2d(x, upsampling_factor)
        forward_res.backward(grad=Tensor.tensor(forward_res.data.copy()))
        backward_res = x.grad 

        fwd_exp_res = torch.from_numpy(expected_res_list['forward_res_list'][i])
        bwd_exp_res = torch.from_numpy(expected_res_list['backward_res_list'][i])

        name = "upsample1d"
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



def test_downsampling2d():
    data_path = os.getcwd() + '/tests/data'
    ref_data_path = os.path.join(data_path,'resampling_ref_data')
    expected_res_list = np.load(os.path.join(ref_data_path, 'downsample_2d_res.npz'), allow_pickle=True)
    seeds = [11593, 11785, 34567]

    for i, seed in enumerate(seeds):
        print(f"** test{i+1}:", end=' ')

        np.random.seed(seed)
        rint = np.random.randint
        in_c = np.random.randint(5, 15)
        out_c = np.random.randint(5, 15)
        kernel = np.random.randint(3, 7)
        downsampling_factor = rint(1, 10)
        width = np.random.randint(60, 80)
        batch = np.random.randint(1, 4)
        x = Tensor.tensor(np.random.randn(batch, in_c, width, width), requires_grad=True)

        downsample2d = Downsample2D()
        forward_res = downsample2d(x, downsampling_factor)
        forward_res.backward(grad=Tensor.tensor(forward_res.data.copy()))
        backward_res = x.grad 

        fwd_exp_res = torch.from_numpy(expected_res_list['forward_res_list'][i])
        bwd_exp_res = torch.from_numpy(expected_res_list['backward_res_list'][i])

        name = "downsample1d"
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
