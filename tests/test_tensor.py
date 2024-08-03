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
        if test_name == "max":
            pyt_result, _ = op(pyt_x)
        else:
            pyt_result = op(pyt_x)
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        # broadcasting
        name = test_name+"_broadcasting1"
        usr_result = op(usr_a)
        if test_name == "max":
            pyt_result, _ = op(pyt_a)
        else:
            pyt_result = op(pyt_a)
        cmp2 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        name = test_name+"broadcasting2"
        usr_result = op(usr_b)
        if test_name == "max":
            pyt_result, _ = op(pyt_b)
        else:
            pyt_result = op(pyt_b)
        cmp3 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))

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
        pi1 = op(pyt_x) 
        if test_name == "reshape":
            ui2 = usr_a.reshape(1, -1) @ ui1.reshape(i, j)
            usr_result = op(usr_b) + ui2
            pi2 = pyt_a.reshape(1, -1) @ pi1.reshape(i, j)
            pyt_result = op(pyt_b) + pi2
        else:
            ui2 = op(usr_a) + ui1
            usr_result = op(usr_b) + ui2
            pi2 = op(pyt_a) + pi1  
            pyt_result = op(pyt_b) + pi2
        
        usr_result.backward(grad=PuruTorch.Tensor.ones_like(usr_result))
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


def test_tensor_slice_forward_(test_name):
    for i in range(2, 5):
        j = np.random.randint(2, 6)

        k = np.random.randint(0, i)
        l = np.random.randint(0, j)

        print(f"** test{i-1}:", end=' ')

        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        npy = np.random.uniform(0.1, 1.0, size=(j,i))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx)
        usr_y = PuruTorch.Tensor.tensor(npy)

        #pytorch
        pyt_x = torch.from_numpy(npx)
        pyt_y = torch.from_numpy(npy)
        
        name = test_name        
        usr_result = usr_x[k, l]
        pyt_result = pyt_x[k, l]
        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        
        name = test_name        
        usr_result = usr_y[l, k]
        pyt_result = pyt_y[l, k]
        cmp2 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
            
        name = test_name        
        usr_result = usr_x[k, :]
        pyt_result = pyt_x[k, :]
        cmp3 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        name = test_name        
        usr_result = usr_y[l, :]
        pyt_result = pyt_y[l, :]
        cmp4 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        name = test_name        
        usr_result = usr_x[:k, :l]
        pyt_result = pyt_x[:k, :l]
        cmp5 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        name = test_name        
        usr_result = usr_y[:l, :k]
        pyt_result = pyt_y[:l, :k]
        cmp6 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        

        if not (cmp1 and cmp2 and cmp3 and cmp4 and cmp5 and cmp6):
            print("failed!")
            return False
        else:
            print("passed!")

    return True


def test_tensor_slice_backward_(test_name):
    for i in range(3, 6):
        j = np.random.randint(2, 6)

        k = np.random.randint(0, i)
        l = np.random.randint(0, j)

        print(f"** test{i-2}:", end=' ')

        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        npy = np.random.uniform(0.1, 1.0, size=(j,i))

        #user
        usr_x = PuruTorch.Tensor.tensor(npx, requires_grad=True)
        usr_y = PuruTorch.Tensor.tensor(npy, requires_grad=True)

        #pytorch
        pyt_x = torch.from_numpy(npx).requires_grad_()
        pyt_y = torch.from_numpy(npy).requires_grad_()
        
        name = test_name        
        usr_result = usr_x[:k, :l] @ usr_y[:l, :k]
        usr_result.backward(grad=PuruTorch.Tensor.ones_like(usr_result))

        pyt_result = pyt_x[:k, :l] @ pyt_y[:l, :k]
        pyt_result.backward(gradient=torch.ones_like(pyt_result))

        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+': x_grad'))
        
        cmp2 = (cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'closeness', name+': x_grad'))
        

        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")

    return True


def test_tensor_sq_usq_forward_(test_name):
    for i in range(2, 5):
        j = np.random.randint(2, 6)

        print(f"** test{i-1}:", end=' ')

        #np
        if test_name == "squeeze":
            npx = np.random.uniform(0.1, 1.0, size=(i,1,j))
            npy = np.random.uniform(0.1, 1.0, size=(j,i,1))
        if test_name == "unsqueeze":
            npx = np.random.uniform(0.1, 1.0, size=(i,))
            npy = np.random.uniform(0.1, 1.0, size=(j,))


        #user
        usr_x = PuruTorch.Tensor.tensor(npx)
        usr_y = PuruTorch.Tensor.tensor(npy)

        #pytorch
        pyt_x = torch.from_numpy(npx)
        pyt_y = torch.from_numpy(npy)
        
        name = test_name
        if test_name == "squeeze":        
            usr_result = usr_x.squeeze(1)
            pyt_result = pyt_x.squeeze(1)
        if test_name == "unsqueeze":
            usr_result = usr_x.unsqueeze(0)
            pyt_result = pyt_x.unsqueeze(0)

        cmp1 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
        
        
        name = test_name
        if test_name == "squeeze":        
            usr_result = usr_y.squeeze(2)
            pyt_result = pyt_y.squeeze(2)
        if test_name == "unsqueeze":
            usr_result = usr_y.unsqueeze(1)
            pyt_result = pyt_y.unsqueeze(1)

        cmp2 = (cmp_usr_pyt_tensor(usr_result, pyt_result, 'type',      name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'shape',     name)
            and cmp_usr_pyt_tensor(usr_result, pyt_result, 'closeness', name))
            
        
        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")

    return True


def test_tensor_sq_usq_backward_(test_name):
    for i in range(3, 6):
        j = np.random.randint(2, 6)

        print(f"** test{i-2}:", end=' ')

        #np
        if test_name == "squeeze": 
            npx = np.random.uniform(0.1, 1.0, size=(i,1,j))
            npy = np.random.uniform(0.1, 1.0, size=(j,i,1))
        if test_name == "unsqueeze":
            npx = np.random.uniform(0.1, 1.0, size=(i,))
            npy = np.random.uniform(0.1, 1.0, size=(j,)) 

        #user
        usr_x = PuruTorch.Tensor.tensor(npx, requires_grad=True)
        usr_y = PuruTorch.Tensor.tensor(npy, requires_grad=True)

        #pytorch
        pyt_x = torch.from_numpy(npx).requires_grad_()
        pyt_y = torch.from_numpy(npy).requires_grad_()
        
        name = test_name 
        if test_name == "squeeze":        
            usr_result = usr_x.squeeze(1) @ usr_y.squeeze(2)
        if test_name == "unsqueeze":
            usr_result = usr_x.unsqueeze(1) @ usr_y.unsqueeze(0)
        
        usr_result.backward(grad=PuruTorch.Tensor.ones_like(usr_result))

        if test_name == "squeeze": 
            pyt_result = pyt_x.squeeze(1) @ pyt_y.squeeze(2)
        if test_name == "unsqueeze":
            pyt_result = pyt_x.unsqueeze(1) @ pyt_y.unsqueeze(0)
        
        pyt_result.backward(gradient=torch.ones_like(pyt_result))

        cmp1 = (cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_x.grad, pyt_x.grad, 'closeness', name+': x_grad'))
        
        cmp2 = (cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'type',      name+': x_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'shape',     name+': x_grad')
            and cmp_usr_pyt_tensor(usr_y.grad, pyt_y.grad, 'closeness', name+': x_grad'))
        

        if not (cmp1 and cmp2):
            print("failed!")
            return False
        else:
            print("passed!")

    return True

def test_tensor_max_sum_mean_backward_(test_name):
    for i in range(1, 4):
        j = np.random.randint(2, 6)

        print(f"** test{i}:", end=' ')
        #np
        npx = np.random.uniform(0.1, 1.0, size=(i,j))
        npa = np.random.uniform(0.1, 1.0, size=(i,1))
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
        usr_result = usr_b + usr_a + usr_x
        pyt_result = pyt_b + pyt_a + pyt_x
        
        if test_name == "max":
            usr_result.max().backward()
            pyt_result.max().backward()
        elif test_name == "sum":
            usr_result.sum().backward()
            pyt_result.sum().backward()
        elif test_name == "mean":
            usr_result.mean().backward()
            pyt_result.mean().backward()

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

def test_tensor_reshape_forward():
    return test_tensor_1op_forward("reshape", lambda a: a.reshape(1, -1))

def test_tensor_reshape_backward():
    return test_tensor_1op_backward("reshape", lambda a: a.reshape(1, -1))

def test_tensor_squeeze_forward():
    return test_tensor_sq_usq_forward_("squeeze")

def test_tensor_squeeze_backward():
    return test_tensor_sq_usq_backward_("squeeze")

def test_tensor_unsqueeze_forward():
    return test_tensor_sq_usq_forward_("unsqueeze")

def test_tensor_unsqueeze_backward():
    return test_tensor_sq_usq_backward_("unsqueeze")

def test_tensor_matmul_forward():
    return test_tensor_2op_forward("matmul", lambda a,b: a @ b)

def test_tensor_matmul_backward():
    return test_tensor_2op_backward("matmul", lambda a,b: a @ b)

def test_tensor_slice_forward():
    return test_tensor_slice_forward_("slice")

def test_tensor_slice_backward():
    return test_tensor_slice_backward_("slice")

def test_tensor_log_forward():
    return test_tensor_1op_forward("log", lambda a: a.log())

def test_tensor_log_backward():
    return test_tensor_1op_backward("log", lambda a: a.log())

def test_tensor_exp_forward():
    return test_tensor_1op_forward("exp", lambda a: a.exp())

def test_tensor_exp_backward():
    return test_tensor_1op_backward("exp", lambda a: a.exp())

def test_tensor_sum_forward():
    return test_tensor_1op_forward("sum", lambda a: a.sum(0, True))

def test_tensor_sum_backward():
    return test_tensor_max_sum_mean_backward_("sum")

def test_tensor_max_forward():
    return test_tensor_1op_forward("max", lambda a: a.max(0, True))

def test_tensor_max_backward():
    return test_tensor_max_sum_mean_backward_("max")

def test_tensor_mean_forward():
    return test_tensor_1op_forward("mean", lambda a: a.mean(0, True))

def test_tensor_mean_backward():
    return test_tensor_max_sum_mean_backward_("mean")

