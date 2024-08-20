import numpy as np
from ..tensor import Tensor
from ..functional import *
from ..utils import Function

# ------------------------------------------
#  Resampling Functionals
# ------------------------------------------

class Upsample1D(Function):
    """
    Functional handling forward/backward passed for
    1D upsampling operation.
    """
    def __call__(self, A:Tensor, factor:int) -> Tensor:
        return self.forward(A, factor)

    def forward(self, A:Tensor, factor:int) -> Tensor:
        # A : N * C_in * W_in
        # Z : N * C_in * W_out
        self.ctx.save_for_backward(A)
        N, C_in, W_in  = A.shape      # batch size x in_chans x in_width
        W_out = factor * (W_in - 1) + 1
        Z = np.zeros((N, C_in, W_out))
        
        self.factor = factor # save for backward

        for batch in range(N):
            for channel in range(C_in):
                i = 0
                for w in range(0, W_out, factor):
                    Z[batch, channel, w] = A.data[batch, channel, i]
                    i += 1
        
        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)



    def backward(self, grad_output:Tensor) -> List[Tensor]:
        # dLdZ : N * C_in * W_out
        # dLdA : N * C_in * W_in
        
        A = self.ctx.saved_tensors[0]
        N, C_in, W_out  = grad_output.shape   # batch size x in_chans x out_width
        _, _, W_in      = A.shape
        dLdA = np.zeros((N, C_in, W_in))

        for batch in range(N):
            for channel in range(C_in):
                i = 0
                for w in range(0, W_out, self.factor):
                    dLdA[batch, channel, i] = grad_output.data[batch, channel, w]
                    i += 1
        return [Tensor.tensor(dLdA)]


class Downsample1D(Function):
    """
    Functional handling forward/backward passed for
    1D downsampling operation.
    """
    def __call__(self, A:Tensor, factor:int) -> Tensor:
        return self.forward(A, factor)

    def forward(self, A:Tensor, factor:int) -> Tensor:
        # A : N * C_in  * W_in
        self.ctx.save_for_backward(A)
        N, C_in, W_in  = A.shape      # batch size x in_chans x in_width
        W_out = int((W_in - 1) / factor + 1)
        Z = np.zeros((N, C_in, W_out))

        self.factor = factor # save for backward

        for batch in range(N):
            for channel in range(C_in):
                i = 0
                for w in range(0, W_in, factor):
                    Z[batch, channel, i] = A.data[batch, channel, w]
                    i += 1
        
        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)


    def backward(self, grad_output:Tensor) -> List[Tensor]:
        A = self.ctx.saved_tensors[0]
        _, _, W_in      = A.shape
        N, C_in, W_out  = grad_output.shape      # batch size x in_chans x out_width
        dLdA = np.zeros((N, C_in, W_in))

        for batch in range(N):
            for channel in range(C_in):
                i = 0
                for w in range(0, W_in, self.factor):
                    dLdA[batch, channel, w] = grad_output.data[batch, channel, i]
                    i += 1
        return [Tensor.tensor(dLdA)]


class Upsample2D(Function):
    """
    Functional handling forward/backward passed for
    2D upsampling operation.
    """
    def __call__(self, A:Tensor, factor:int) -> Tensor:
        return self.forward(A, factor)

    def forward(self, A:Tensor, factor:int) -> Tensor:
         # A : N * C_in  H_in * W_in
        self.ctx.save_for_backward(A)
        N, C_in, H_in, W_in  = A.shape      # batch size x in_chans x in_height x in_width
        H_out = factor * (H_in - 1) + 1
        W_out = factor * (W_in - 1) + 1
        Z = np.zeros((N, C_in, H_out, W_out))
        
        self.factor = factor # save for backward

        for batch in range(N):
            for channel in range(C_in):
                j = 0
                for h in range(0, H_out, factor):
                    i = 0
                    for w in range(0, W_out, factor):
                        Z[batch, channel, h, w] = A.data[batch, channel, j, i]
                        i+=1
                    j+=1 
        
        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)


    def backward(self, grad_output:Tensor) -> List[Tensor]:
        A = self.ctx.saved_tensors[0]
        N, C_in, H_out, W_out  = grad_output.shape   # batch size x in_chans x out_width
        _, _, H_in, W_in      = A.shape
        dLdA = np.zeros((N, C_in, H_in, W_in))

        for batch in range(N):
            for channel in range(C_in):
                j = 0
                for h in range(0, H_out, self.factor):
                    i = 0
                    for w in range(0, W_out, self.factor):
                        dLdA[batch, channel, j, i] = grad_output.data[batch, channel, h, w]
                        i+=1
                    j+=1

        return [Tensor.tensor(dLdA)]


class Downsample2D(Function):
    """
    Functional handling forward/backward passed for
    2D downsampling operation.
    """
    def __call__(self, A:Tensor, factor:int) -> Tensor:
        return self.forward(A, factor)

    def forward(self, A:Tensor, factor:int) -> Tensor:
        # A : N * C_in  H_in * W_in
        self.ctx.save_for_backward(A)
        N, C_in, H_in, W_in  = A.shape      # batch size x in_chans x in_height x in_width
        H_out = int((H_in - 1) / factor + 1)
        W_out = int((W_in - 1) / factor + 1)
        Z = np.zeros((N, C_in, H_out, W_out))

        self.factor = factor # save for backward

        for batch in range(N):
            for channel in range(C_in):
                j = 0
                for h in range(0, H_in, factor):
                    i = 0
                    for w in range(0, W_in, factor):
                        Z[batch, channel, j, i] = A.data[batch, channel, h, w]
                        i += 1
                    j +=1
            
        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)
        

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        A = self.ctx.saved_tensors[0]
        _, _, H_in, W_in       = A.shape
        N, C_in, H_out, W_out  = grad_output.shape  # batch size x in_chans x out_width
        dLdA = np.zeros((N, C_in, H_in, W_in))

        for batch in range(N):
            for channel in range(C_in):
                j = 0
                for h in range(0, H_in, self.factor):
                    i = 0
                    for w in range(0, W_in, self.factor):
                        dLdA[batch, channel, h, w] = grad_output.data[batch, channel, j, i]
                        i += 1
                    j +=1

        return [Tensor.tensor(dLdA)]