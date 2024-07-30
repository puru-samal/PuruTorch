import  numpy as np
import  sys
sys.path.append("./")
from    pathlib import Path
from    helpers import *
from    test_tensor import *

import  PuruTorch

test_list = {
    'Tensor': [
        {
            'name':    'Tensor Add Forward',
            'autolab': 'Tensor Add Forward',
            'handler': test_tensor_add_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Add Backward',
            'autolab': 'Tensor Add Backward',
            'handler': test_tensor_add_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Neg Forward',
            'autolab': 'Tensor Neg Forward',
            'handler': test_tensor_neg_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Neg Backward',
            'autolab': 'Tensor Neg Backward',
            'handler': test_tensor_neg_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Sub Forward',
            'autolab': 'Tensor Sub Forward',
            'handler': test_tensor_sub_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Sub Backward',
            'autolab': 'Tensor Sub Backward',
            'handler': test_tensor_sub_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Mul Forward',
            'autolab': 'Tensor Mul Forward',
            'handler': test_tensor_mul_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Mul Backward',
            'autolab': 'Tensor Mul Backward',
            'handler': test_tensor_mul_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Div Forward',
            'autolab': 'Tensor Div Forward',
            'handler': test_tensor_div_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Div Backward',
            'autolab': 'Tensor Div Backward',
            'handler': test_tensor_div_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Pow Forward',
            'autolab': 'Tensor Pow Forward',
            'handler': test_tensor_pow_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Pow Backward',
            'autolab': 'Tensor Pow Backward',
            'handler': test_tensor_pow_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Transpose Forward',
            'autolab': 'Tensor Transpose Forward',
            'handler': test_tensor_transp_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Transpose Backward',
            'autolab': 'Tensor Transpose Backward',
            'handler': test_tensor_transp_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Matmul Forward',
            'autolab': 'Tensor Matmul Forward',
            'handler': test_tensor_matmul_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Matmul Backward',
            'autolab': 'Tensor Matmul Backward',
            'handler': test_tensor_matmul_backward,
            'value': 1,
        },
        {
            'name':    'Tensor Slice Forward',
            'autolab': 'Tensor Slice Forward',
            'handler': test_tensor_slice_forward,
            'value': 1,
        },
        {
            'name':    'Tensor Slice Backward',
            'autolab': 'Tensor Slice Backward',
            'handler': test_tensor_slice_backward,
            'value': 1,
        },
    ],
}


if __name__=='__main__':
    # # DO NOT EDIT
    if len(sys.argv) == 1:
        # run all tests
        tests = [test for sublist in test_list.values() for test in sublist]
        pass
    elif len(sys.argv) == 2:
        # run only tests for specified section
        test_type = sys.argv[1]
        if test_type in test_list:
           tests = test_list[test_type]
        else:
            sys.exit(f'Invalid test type option provided.\nEnter one of [{", ".join(test_list.keys())}].\nOr leave empty to run all tests.')
    else:
        sys.exit(f'Multiple test type options provided.\nEnter one of [{", ".join(test_list.keys())}].\nOr leave empty to run all tests.')

    # tests.reverse()
    run_tests(tests)


