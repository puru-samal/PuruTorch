import sys
sys.path.append("./")
import PuruTorch
import torch
import numpy as np
from helpers import *

def test_tensor_2op_forward(test_name, op):
    for i in range(1, 4):
        j = np.random.randint(1, 6)

        print(f"** test{i}:", end=' ')
        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        if test_name == "matmul":
            npy = np.random.uniform(0.1, 1.0, size=(j,i))
            npa = np.random.uniform(0.1, 1.0, size=(j,1))
            npb = np.random.uniform(0.1, 1.0, size=(i,1))
        else:
            npy = np.random.uniform(0.1, 1.0, size=(i,j))
            npa = np.random.uniform(0.1, 1.0, size=(i,1))
            npb = np.random.uniform(0.1, 1.0, size=(j,))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx)
        usr_y = PuruTorch.Tensor.tensor(npy)
        usr_a = PuruTorch.Tensor.tensor(npa)
        usr_b = PuruTorch.Tensor.tensor(npb)

        #pytorch
        pyt_x = torch.from_numpy(npx)
        pyt_y = torch.from_numpy(npy)
        pyt_a = torch.from_numpy(npa)
        pyt_b = torch.from_numpy(npb)

        # op
        name = test_name
        usr_result = op(usr_x, usr_y)
        pyt_result = op(pyt_x, pyt_y)
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        # broadcasting
        name = test_name+"_broadcasting1" if test_name != "matmul" else test_name
        usr_result = op(usr_x, usr_a)
        pyt_result = op(pyt_x, pyt_a)
        cmp2 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', test_name))
        
        test_name = test_name+"broadcasting2" if test_name != "matmul" else test_name
        usr_result = op(usr_y, usr_b)
        pyt_result = op(pyt_y, pyt_b)
        cmp3 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', test_name))

        if not (cmp1 and cmp2 and cmp3):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_tensor_2op_backward(test_name, op):

    for i in range(1, 4):
        j = np.random.randint(2, 6)

        print(f"** test{i}:", end=' ')
        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        if test_name == "matmul":
            npy = np.random.uniform(0.1, 1.0, size=(i,1))
            npa = np.random.uniform(0.1, 1.0, size=(j,i))
            npb = np.random.uniform(0.1, 1.0, size=(1,i))
        else:
            npy = np.random.uniform(0.1, 1.0, size=(i,j))
            npa = np.random.uniform(0.1, 1.0, size=(i,1))
            npb = np.random.uniform(0.1, 1.0, size=(j,))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx, requires_grad=True)
        usr_y = PuruTorch.Tensor.tensor(npy, requires_grad=True)
        usr_a = PuruTorch.Tensor.tensor(npa, requires_grad=True)
        usr_b = PuruTorch.Tensor.tensor(npb, requires_grad=True)
        
        #pytorch
        pyt_x = torch.from_numpy(npx).requires_grad_()
        pyt_y = torch.from_numpy(npy).requires_grad_()
        pyt_a = torch.from_numpy(npa).requires_grad_()
        pyt_b = torch.from_numpy(npb).requires_grad_()
        
        # complex op
        name = test_name
        ui1 = op(usr_x, usr_a)
        ui2 = op(ui1,   usr_y)
        usr_result = op(ui2, usr_b)
        usr_result.backward(grad=PuruTorch.Tensor.ones_like(usr_result))

        pi1 = op(pyt_x, pyt_a)
        pi2 = op(pi1,   pyt_y)  
        pyt_result = op(pi2, pyt_b)
        pyt_result.backward(gradient=torch.ones_like(pyt_result))

        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+': x_grad'))
        
        
        cmp2 = (cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'type',      name+': y_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'shape',     name+': y_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'closeness', name+': y_grad'))
        
        cmp3 = (cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'type',      name+': a_grad')
            and cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'shape',     name+': a_grad')
            and cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'closeness', name+': a_grad'))
        
        
        cmp4 = (cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'type',      name+': b_grad')
            and cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'shape',     name+': b_grad')
            and cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'closeness', name+': b_grad'))
        
        if not (cmp1 and cmp2 and cmp3 and cmp4):
            print("failed!")
            return False
        else:
            print("passed!")

    return True


def test_tensor_1op_forward(test_name, op):
    for i in range(1, 4):
        j = np.random.randint(1, 6)

        print(f"** test{i}:", end=' ')
        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        npa = np.random.uniform(0.1, 1.0, size=(i,1))
        if test_name == "transpose":
            npb = np.random.uniform(0.1, 1.0, size=(i,))
        else:
            npb = np.random.uniform(0.1, 1.0, size=(j,))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx)
        usr_a = PuruTorch.Tensor.tensor(npa)
        usr_b = PuruTorch.Tensor.tensor(npb)

        #pytorch
        pyt_x = torch.from_numpy(npx)
        pyt_a = torch.from_numpy(npa)
        pyt_b = torch.from_numpy(npb)

        # add
        name = test_name
        usr_result = op(usr_x)
        pyt_result = op(pyt_x)
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        # broadcasting
        name = test_name+"_broadcasting1"
        usr_result = op(usr_a)
        pyt_result = op(pyt_a)
        cmp2 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', test_name))
        
        test_name = test_name+"broadcasting2"
        usr_result = op(usr_b)
        pyt_result = op(pyt_b)
        cmp3 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     test_name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', test_name))

        if not (cmp1 and cmp2 and cmp3):
            print("failed!")
            return False
        else:
            print("passed!")
    
    return True


def test_tensor_1op_backward(test_name, op):

    for i in range(1, 4):
        j = np.random.randint(2, 6)

        print(f"** test{i}:", end=' ')
        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        npa = np.random.uniform(0.1, 1.0, size=(i,1))
        if test_name == "transpose":
            npb = np.random.uniform(0.1, 1.0, size=(i,))
        else:
            npb = np.random.uniform(0.1, 1.0, size=(j,))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx, requires_grad=True)
        usr_a = PuruTorch.Tensor.tensor(npa, requires_grad=True)
        usr_b = PuruTorch.Tensor.tensor(npb, requires_grad=True)
        
        #pytorch
        pyt_x = torch.from_numpy(npx).requires_grad_()
        pyt_a = torch.from_numpy(npa).requires_grad_()
        pyt_b = torch.from_numpy(npb).requires_grad_()
        
        # complex op
        name = test_name
        ui1 = op(usr_x)
        ui2 = op(usr_a) + ui1
        usr_result = op(usr_b) + ui2
        usr_result.backward(grad=PuruTorch.Tensor.ones_like(usr_result))

        pi1 = op(pyt_x)
        pi2 = op(pyt_a) + pi1  
        pyt_result = op(pyt_b) + pi2
        pyt_result.backward(gradient=torch.ones_like(pyt_result))

        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+': x_grad'))
        
        
        cmp2 = (cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'type',      name+': a_grad')
            and cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'shape',     name+': a_grad')
            and cmp_usr_pyt_tensor(usr_a.grad, pyt_a.grad, 'closeness', name+': a_grad'))
        
        
        cmp3 = (cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'type',      name+': b_grad')
            and cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'shape',     name+': b_grad')
            and cmp_usr_pyt_tensor(usr_b.grad, pyt_b.grad, 'closeness', name+': b_grad'))
        
        if not (cmp1 and cmp2 and cmp3):
            print("failed!")
            return False
        else:
            print("passed!")

    return True


## Operator Tests
def test_tensor_add_forward():
    return test_tensor_2op_forward("add", lambda a,b: a+b)

def test_tensor_add_backward():
    return test_tensor_2op_backward("add", lambda a,b: a+b)

def test_tensor_neg_forward():
    return test_tensor_1op_forward("neg", lambda a: -a)

def test_tensor_neg_backward():
    return test_tensor_1op_backward("neg", lambda a: -a)

def test_tensor_sub_forward():
    return test_tensor_2op_forward("sub", lambda a,b: a-b)

def test_tensor_sub_backward():
    return test_tensor_2op_backward("sub", lambda a,b: a-b)

def test_tensor_mul_forward():
    return test_tensor_2op_forward("mul", lambda a,b: a*b)

def test_tensor_mul_backward():
    return test_tensor_2op_backward("mul", lambda a,b: a*b)

def test_tensor_div_forward():
    return test_tensor_2op_forward("div", lambda a,b: a/b)

def test_tensor_div_backward():
    return test_tensor_2op_backward("div", lambda a,b: a/b)

def test_tensor_pow_forward():
    return test_tensor_1op_forward("pow", lambda a: a**2.0)

def test_tensor_pow_backward():
    return test_tensor_1op_backward("pow", lambda a: a**2.0)

def test_tensor_transp_forward():
    return test_tensor_1op_forward("transpose", lambda a: a.T)

def test_tensor_transp_backward():
    return test_tensor_1op_backward("transpose", lambda a: a.T)

def test_tensor_matmul_forward():
    return test_tensor_2op_forward("matmul", lambda a,b: a @ b)

def test_tensor_matmul_backward():
    return test_tensor_2op_backward("matmul", lambda a,b: a @ b)