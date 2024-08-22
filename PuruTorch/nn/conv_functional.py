import numpy as np
from ..tensor import Tensor
from ..functional import *
from ..utils import Function

# ------------------------------------------
#  Conv Functionals
# ------------------------------------------

class Pad1D(Function):
    """
    Functional to handle forward/backward passes of 
    padding a A 3D input for 1D convolution.
    """
    def __call__(self, A:Tensor, padding:int=0) -> Tensor:
        return self.forward(A, padding)
    
    def forward(self, A:Tensor, padding:int=0) -> Tensor:
        self.ctx.save_for_backward(A)
        self.ctx.padding = padding # for backward
        if padding > 0:
            padded_A = np.pad(A.data, pad_width=((0,), (0,), (padding,)), mode='constant')
        else:
            padded_A = A.data.copy()
        return Tensor(padded_A, A.requires_grad, self if A.requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.padding > 0:
            dLdA = grad_output.data[:, :, self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding)]
        else:
            dLdA = grad_output.data.copy()
        return [Tensor.tensor(dLdA)]
        

class Conv1D_stride1(Function):
    """
    Functional to handle forward/backward passes of 
    a 1D stride1 convolution 
    """
    def __call__(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        return self.forward(A, W, b)
    
    def forward(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        # A : N * C_in  * W_in
        # Z : N * C_out * W_out
        # W : C_out * C_in * K
        # b : C_out,

        self.ctx.save_for_backward(A, W, b)
        N, _, W_in  = A.shape      # batch size x in_chans x in_width
        C_out, _, K = W.shape      # out_chans x in_chans x kernel
        W_out       = W_in - K + 1 # out_width

        Z = np.zeros((N, C_out, W_out))

        for w in range(W_out):
            axs = ([1, 2], [1, 2])
            Z[:, :, w] += np.tensordot(A.data[:, :, w : w+K], W.data, axes=axs)
            Z[:, :, w] += b.data
        
        requires_grad = A.requires_grad or W.requires_grad or b.requires_grad
        return Tensor(Z, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        # dLdA : N * C_in  * W_in
        # dLdZ : N * C_out * W_out (grad_output)
        # dLdW : C_out * C_in * K
        # dLdb : C_out,

        A, W, _ = self.ctx.saved_tensors
        _, _, K     = W.shape
        _, _, W_out = grad_output.shape
        _, _, W_in  = A.shape

        dLdb = np.sum(grad_output.data, axis=(0, 2))

        dLdW = np.zeros_like(W.data)
        for k in range(K):
            axs = ([0, 2], [0, 2])
            dLdW[:, :,k] += np.tensordot(grad_output.data, A.data[:, :, k:k+W_out], axes=axs)

        dLdA = np.zeros_like(A.data)
        # Padding only on laxt axis
        pwidths = ((0,), (0,), (K-1,))
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')
        flipW = np.flip(W.data, axis=2)  # Flip only on last axis

        for w in range(W_in):
            axs = ([1, 2], [0, 2])
            dLdA[:, :, w] = np.tensordot(pdLdZ[:, :, w:w+K], flipW, axes=axs)

        return [Tensor.tensor(dLdA), Tensor.tensor(dLdW), Tensor.tensor(dLdb)]

class Pad2D(Function):
    """
    Functional to handle forward/backward passes of 
    padding a A 4D input for 2D convolution.
    """
    def __call__(self, A:Tensor, padding:int=0) -> Tensor:
        return self.forward(A, padding)
    
    def forward(self, A:Tensor, padding:int=0) -> Tensor:
        self.ctx.save_for_backward(A)
        self.ctx.padding = padding # for backward
        if  padding > 0:
            padded_A = np.pad(A.data, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant')
        else:
            padded_A = A.data.copy()
        return Tensor(padded_A, A.requires_grad, self if A.requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        if self.ctx.padding > 0:
            dLdA = grad_output.data[:, :, 
                                    self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding), 
                                    self.ctx.padding:(grad_output.shape[-1]-self.ctx.padding)]
        else:
            dLdA = grad_output.data.copy()
        return [Tensor.tensor(dLdA)]


class Conv2D_stride1(Function):
    """
    Functional to handle forward/backward passes of 
    a 2D stride1 convolution 
    """
    def __call__(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        return self.forward(A, W, b)
    
    def forward(self, A:Tensor, W:Tensor, b:Tensor) -> Tensor:
        # A : N * C_in  * H_in * W_in
        # Z : N * C_out * H_in * W_out
        # W : C_out * C_in * K * K
        # b : C_out,

        self.ctx.save_for_backward(A, W, b)
        N, _, H_in, W_in  = A.shape      # batch size x in_chans x in_width
        C_out, _, _, K    = W.shape      # out_chans x in_chans x kernel
        H_out, W_out      = H_in - K + 1, W_in - K + 1 # out_width

        Z = np.zeros((N, C_out, H_out, W_out))

        for h in range(H_out):
            for w in range(W_out):
                axs = ([1, 2, 3], [1, 2, 3])
                Z[:, :, h, w] += np.tensordot(A.data[:, :, h:h+K, w:w+K], W.data, axes=axs)
                Z[:, :, h, w] += b.data
        
        requires_grad = A.requires_grad or W.requires_grad or b.requires_grad
        return Tensor(Z, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output:Tensor) -> List[Tensor]:
        # dLdA : N * C_in  * W_in
        # dLdZ : N * C_out * W_out (grad_output)
        # dLdW : C_out * C_in * K
        # dLdb : C_out * 1

        A, W, _ = self.ctx.saved_tensors
        _, _, _, K         = W.shape
        _, _, H_out, W_out = grad_output.shape
        _, _, H_in,  W_in  = A.shape

        dLdb = np.sum(grad_output.data, axis=(0, 2, 3))

        dLdW = np.zeros_like(W.data)
        for kh in range(K):
            for kw in range(K):
                axs = ([0, 2, 3], [0, 2, 3])
                dLdW[:, :, kh, kw] += np.tensordot(grad_output.data, A.data[:, :, kh:kh+H_out, kw:kw+W_out], axes=axs)

        dLdA = np.zeros_like(A.data)
        # Padding only on laxt axis
        pwidths = ((0,), (0,), (K-1,), (K-1,))
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')
        flipW = np.flip(W.data, axis=(2,3))  # Flip only on last two axis

        for h in range(H_in):
            for w in range(W_in):
                axs = ([1, 2, 3], [0, 2, 3])
                dLdA[:, :, h, w] = np.tensordot(pdLdZ[:, :, h:h+K, w:w+K], flipW, axes=axs)

        return [Tensor.tensor(dLdA), Tensor.tensor(dLdW), Tensor.tensor(dLdb)]

# ------------------------------------------
#  Pool Functionals
# ------------------------------------------

class MaxPool1D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_width)
        """

        self.ctx.save_for_backward(A)
        N, C_in, W_in = A.shape
        C_out = C_in
        W_out = W_in - kernel + 1

        self.ctx.maxindex = np.empty((N, C_out, W_out), dtype=tuple)
        Z = np.zeros((N, C_out, W_out))

        for batch in range(N):
            for ch in range(C_out):
                for w in range(W_out):
                    scan = A.data[batch, ch, w:w+kernel]
                    Z[batch, ch, w] = np.max(scan)
                    self.ctx.maxindex[batch, ch, w] = np.unravel_index(np.argmax(scan), scan.shape)
                    self.ctx.maxindex[batch, ch, w] = tuple(np.add((w), self.ctx.maxindex[batch, ch, w]))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, W_out = grad_output.shape

        for batch in range(N):
            for ch in range(C_out):
                for w in range(W_out):
                    i1  = self.ctx.maxindex[batch, ch, w]
                    dLdA[batch, ch, i1] = grad_output.data[batch, ch, w]
        return [Tensor.tensor(dLdA)]


class MeanPool1D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_width)
        """
        self.ctx.save_for_backward(A)
        self.ctx.kernel = kernel
        N, C_in, W_in = A.shape
        C_out = C_in
        W_out = W_in - kernel + 1
        Z = np.zeros((N, C_out, W_out))

        for w in range(W_out):
            Z[:, :, w] = np.mean(A.data[:, :, w:w+kernel], axis=2)

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, W_out = grad_output.shape

        pwidths = ((0,), (0,), (self.ctx.kernel-1,))
        # Pad with zeroes to shape match
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')

        for w in range(W_out):
            dLdA[:, :, w] = np.mean(pdLdZ[:, :, w:w+self.ctx.kernel], axis=2)
                
        return [Tensor.tensor(dLdA)]
       

class MaxPool2D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_height,  input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_height, output_height)
        """

        self.ctx.save_for_backward(A)
        N, C_in, H_in, W_in = A.shape
        C_out = C_in
        H_out, W_out = H_in - kernel + 1, W_in - kernel + 1

        self.ctx.maxindex = np.empty((N, C_out, H_out, W_out), dtype=tuple)
        Z = np.zeros((N, C_out, H_out, W_out))

        for batch in range(N):
            for ch in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        scan = A.data[batch, ch, h:h+kernel, w:w+kernel]
                        Z[batch, ch, h, w] = np.max(scan)
                        self.ctx.maxindex[batch, ch, h, w] = np.unravel_index(np.argmax(scan), scan.shape)
                        self.ctx.maxindex[batch, ch, h, w] = tuple(np.add((h, w), self.ctx.maxindex[batch, ch, h, w]))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_height, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, H_out, W_out = grad_output.shape

        for batch in range(N):
            for ch in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        i1, i2 = self.ctx.maxindex[batch, ch, h, w]
                        dLdA[batch, ch, i1, i2] = grad_output.data[batch, ch, h, w]
        return [Tensor.tensor(dLdA)]


class MeanPool2D_stride1(Function):
    def __call__(self, A:Tensor, kernel:int) -> Tensor:
        return self.forward(A, kernel)

    def forward(self, A:Tensor, kernel:int) -> Tensor:
        """
        Argument:
            A (Tensor): (batch_size, in_channels, input_height,  input_width)
        Return:
            Z (Tensor): (batch_size, out_channels, output_height, output_width)
        """
        self.ctx.save_for_backward(A)
        self.ctx.kernel = kernel
        N, C_in, H_in, W_in = A.shape
        C_out = C_in
        H_out, W_out = H_in - kernel + 1, W_in - kernel + 1
        Z = np.zeros((N, C_out, H_out, W_out))

        for h in range(H_out):
            for w in range(W_out):
                Z[:, :, h, w] = np.mean(A.data[:, :, h:h+kernel, w:w+kernel], axis=(2, 3))

        return Tensor(Z, A.requires_grad, self if A.requires_grad else None)

    def backward(self, grad_output:Tensor) -> List[Tensor]:
        """
        Argument:
            grad_output (Tensor): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (Tensor): (batch_size, in_channels, input_height, input_width)
        """
        A = self.ctx.saved_tensors[0]
        dLdA = np.zeros_like(A.data)
        N, C_out, H_out, W_out = grad_output.shape

        pwidths = ((0,), (0,), (self.ctx.kernel-1,), (self.ctx.kernel-1,))
        # Pad with zeroes to shape match
        pdLdZ = np.pad(grad_output.data, pad_width=pwidths, mode='constant')

        for h in range(H_out):
            for w in range(W_out):
                dLdA[:, :, h, w] = np.mean(pdLdZ[:, :, h:h+self.ctx.kernel, w:w+self.ctx.kernel], axis=(2, 3))
                
        return [Tensor.tensor(dLdA)]
       

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