import  numpy as np
import  sys
sys.path.append("./")
from    pathlib import Path
from    helpers import *
from    test_ops import *
from    test_linear import *
from    test_activations import *
from    test_mlp import *
from    test_loss import *
from    test_optim import *
from    test_batchnorm import *
from    test_rnn import *
from    test_gru import *
from    test_ctc_decoding import *
from    test_resampling import *
from    test_conv1d import *
from    test_conv2d import *
from    test_convtransp1d import *
from    test_convtransp2d import *
from    test_dropout import *
from    test_pool import *
from    test_attention import *

search_test = SearchTest()

test_list = {
    'ops': [
        {
            'name':    'Functional: Add Forward/Backward',
            'autolab': 'Functional: Add Forward/Backward',
            'handler': test_add_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Neg Forward/Backward',
            'autolab': 'Functional: Neg Forward/Backward',
            'handler': test_neg_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Sub Forward/Backward',
            'autolab': 'Functional: Sub Forward/Backward',
            'handler': test_sub_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Mul Forward/Backward',
            'autolab': 'Functional: Mul Forward/Backward',
            'handler': test_mul_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Div Forward/Backward',
            'autolab': 'Functional: Div Forward/Backward',
            'handler': test_div_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Pow Forward/Backward',
            'autolab': 'Functional: Pow Forward/Backward',
            'handler': test_pow_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Transpose Forward/Backward',
            'autolab': 'Functional: Transpose Forward/Backward',
            'handler': test_transp_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Reshape Forward/Backward',
            'autolab': 'Functional: Reshape Forward/Backward',
            'handler': test_reshape_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Squeeze Forward/Backward',
            'autolab': 'Functional: Squeeze Forward/Backward',
            'handler': test_squeeze_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Unsqueeze Forward/Backward',
            'autolab': 'Functional: Unsqueeze Forward/Backward',
            'handler': test_unsqueeze_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Matmul Forward/Backward',
            'autolab': 'Functional: Matmul Forward/Backward',
            'handler': test_matmul_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Slice Forward/Backward',
            'autolab': 'Functional: Slice Forward/Backward',
            'handler': test_slice_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Log Forward/Backward',
            'autolab': 'Functional: Log Forward/Backward',
            'handler': test_log_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Exp Forward/Backward',
            'autolab': 'Functional: Exp Forward/Backward',
            'handler': test_exp_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Sum Forward/Backward',
            'autolab': 'Functional: Sum Forward/Backward',
            'handler': test_sum_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Max Forward/Backward',
            'autolab': 'Functional: Mac Forward/Backward',
            'handler': test_max_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Mean Forward/Backward',
            'autolab': 'Functional: Mean Forward/Backward',
            'handler': test_mean_fwd_bwd,
            'value': 1,
        },
        {
            'name':    'Functional: Var Forward/Backward',
            'autolab': 'Functional: Var Forward/Backward',
            'handler': test_var_fwd_bwd,
            'value': 1,
        },
    ],
    'nn': [
        {
            'name':    'Activation: Identity Forward/Backward',
            'autolab': 'Activation: Identity Forward/Backward',
            'handler': test_identity,
            'value': 1,
        },
        {
            'name':    'Activation: Sigmoid Forward/Backward',
            'autolab': 'Activation: Sigmoid Forward/Backward',
            'handler': test_sigmoid,
            'value': 1,
        },
        {
            'name':    'Activation: ReLU Forward/Backward',
            'autolab': 'Activation: ReLU Forward/Backward',
            'handler': test_relu,
            'value': 1,
        },
        {
            'name':    'Activation: Tanh Activation Forward/Backward',
            'autolab': 'Activation: Tanh Forward/Backward',
            'handler': test_tanh,
            'value': 1,
        },
        {
            'name':    'Activation: Softmax Forward/Backward',
            'autolab': 'Activation: Softmax Forward/Backward',
            'handler': test_softmax,
            'value': 1,
        },
        {
            'name':    'Loss: MSELoss Forward/Backward',
            'autolab': 'Loss: MSELoss Forward/Backward',
            'handler': test_loss_mse,
            'value': 1,
        },
        {
            'name':    'Loss: CELoss Forward/Backward',
            'autolab': 'Loss: CELoss Forward/Backward',
            'handler': test_loss_ce,
            'value': 1,
        },
        {
            'name':    'Loss: CTCLoss Forward/Backward',
            'autolab': 'Loss: CTCLoss Forward/Backward',
            'handler': test_loss_ctc,
            'value': 1,
        },
        {
            'name':    'Reg: Batchnorm1d Forward/Backward',
            'autolab': 'Reg: Batchnorm1d Forward/Backward',
            'handler': test_batchnorm1d,
            'value': 1,
        },
        {
            'name':    'Reg: Batchnorm2d Forward/Backward',
            'autolab': 'Reg: Batchnorm2d Forward/Backward',
            'handler': test_batchnorm2d,
            'value': 1,
        },
        {
            'name':    'Reg: Dropout Forward/Backward',
            'autolab': 'Reg: Dropout Forward/Backward',
            'handler': test_dropout,
            'value': 1,
        },
        {
            'name':    'Reg: Dropout1D Forward/Backward',
            'autolab': 'Reg: Dropout1D Forward/Backward',
            'handler': test_dropout1d,
            'value': 1,
        },
        {
            'name':    'Reg: Dropout2D Forward/Backward',
            'autolab': 'Reg: Dropout2D Forward/Backward',
            'handler': test_dropout2d,
            'value': 1,
        },
        {
            'name':    'Resampling: Upsample1D Forward/Backward',
            'autolab': 'Resampling: Upsample1D Forward/Backward',
            'handler': test_upsampling1d,
            'value': 1,
        },
        {
            'name':    'Resampling: Downsample1D Forward/Backward',
            'autolab': 'Resampling: Downsample1D Forward/Backward',
            'handler': test_downsampling1d,
            'value': 1,
        },
        {
            'name':    'Resampling: Upsample2D Forward/Backward',
            'autolab': 'Resampling: Upsample2D Forward/Backward',
            'handler': test_upsampling2d,
            'value': 1,
        },
        {
            'name':    'Resampling: Downsample2D Forward/Backward',
            'autolab': 'Resampling: Downsample2D Forward/Backward',
            'handler': test_downsampling2d,
            'value': 1,
        },
        {
            'name':    'Pooling: MaxPool1D Forward/Backward',
            'autolab': 'Pooling: MaxPool1D Forward/Backward',
            'handler': test_maxpool1d,
            'value': 1,
        },
        {
            'name':    'Pooling: MeanPool1D Forward/Backward',
            'autolab': 'Pooling: MeanPool1D Forward/Backward',
            'handler': test_meanpool1d,
            'value': 1,
        },
        {
            'name':    'Pooling: MaxPool2D Forward/Backward',
            'autolab': 'Pooling: MaxPool2D Forward/Backward',
            'handler': test_maxpool2d,
            'value': 1,
        },
        {
            'name':    'Pooling: MeanPool2D Forward/Backward',
            'autolab': 'Pooling: MeanPool2D Forward/Backward',
            'handler': test_meanpool2d,
            'value': 1,
        },
        {
            'name':    'Layer: Linear Forward/Backward',
            'autolab': 'Layer: Linear Forward/Backward',
            'handler': test_linear,
            'value': 1,
        },
        {
            'name':    'Layer: Conv1D Forward/Backward',
            'autolab': 'Layer: Conv1D Forward/Backward',
            'handler': test_conv1d,
            'value': 1,
        },
        {
            'name':    'Layer: Conv2D Forward/Backward',
            'autolab': 'Layer: Conv2D Forward/Backward',
            'handler': test_conv2d,
            'value': 1,
        },
        {
            'name':    'Layer: ConvTranspose1D Forward/Backward',
            'autolab': 'Layer: ConvTranspose1D Forward/Backward',
            'handler': test_convtransp1d,
            'value': 1,
        },
        {
            'name':    'Layer: ConvTranspose2D Forward/Backward',
            'autolab': 'Layer: ConvTranspose2D Forward/Backward',
            'handler': test_convtransp2d,
            'value': 1,
        },
        {
            'name':    'Layer: RNNCell Forward/Backward',
            'autolab': 'Layer: RNNCell Forward/Backward',
            'handler': test_rnncell,
            'value': 1,
        },
        {
            'name':    'Layer: GRUCell Forward/Backward',
            'autolab': 'Layer: GRUCell Forward/Backward',
            'handler': test_grucell,
            'value': 1,
        },
    ],
    'optim': [
        {
            'name':    'Optim: SGD Step',
            'autolab': 'Optim: SGD Step',
            'handler': test_optim_sgd,
            'value': 1,
        },
        {
            'name':    'Optim: Adam Step',
            'autolab': 'Optim: Adam Step',
            'handler': test_optim_adam,
            'value': 1,
        },
        {
            'name':    'Optim: AdamW Step',
            'autolab': 'Optim: AdamW Step',
            'handler': test_optim_adamW,
            'value': 1,
        },
    ],
    'model': [
        {
            'name':    'Model: MLP Forward/Backward',
            'autolab': 'Model: MLP Forward/Backward',
            'handler': test_mlp,
            'value': 1,
        },
        {
            'name':    'Model: CNN Forward/Backward',
            'autolab': 'Model: CNN Forward/Backward',
            'handler': test_cnn,
            'value': 1,
        },
        {
            'name':    'Model: ResBlock Forward/Backward',
            'autolab': 'Model: ResBlock Forward/Backward',
            'handler': test_resblock,
            'value': 1,
        },
        {
            'name':    'Model: RNNClassifier Forward/Backward',
            'autolab': 'Model: RNNClassifier Forward/Backward',
            'handler': test_rnn_classifier,
            'value': 1,
        },
        {
            'name':    'Model: GRUClassifier Forward/Backward',
            'autolab': 'Model: GRUClassifier Forward/Backward',
            'handler': test_gru_classifier,
            'value': 1,
        },
    ],
    'search': [
        {
            'name':    'Decoding: Greedy Search (Numpy impl.)',
            'autolab': 'Decoding: Greedy Search (Numpy impl.)',
            'handler': search_test.test_greedy_search,
            'value': 1
        },
        {
            'name':    'Decoding: Beam Search (Numpy impl.)',
            'autolab': 'Decoding: Beam Search (Numpy impl.)',
            'handler': search_test.test_beam_search,
            'value': 1
        }
    ],
    'Attention': [
        {
            'name':    'Attention Forward/Backward (Pytorch impl.)',
            'autolab': 'Attention Forward/Backward (Pytorch impl.)',
            'handler': test_attention,
            'value': 1
        },
    ]
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


