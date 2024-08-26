# PuruTorch

A deep learning library with reverse-mode automatic differentiation implemented from scratch in Python with Numpy.

# Motivation

As a student in CMU’s [11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/F24/index.html) course, I gained hands-on experience coding various deep learning primitives through explicit forward and backward passes. The following summer, I was hired as a TA for the course and was responsible for the optional Autograd 1/2 assignments, where students would implement automatic differentiation for a small subset of deep learning modules. Although, I initially opted out of this assignment due to a busy semester, my role as a lead TA exposed me to the elegance and efficiency of reverse-mode automatic differentiation for computing gradients. Fascinated by its simplicity and inspired by a desire to reinforce my learning, I decided to re-implement the deep learning models I had previously coded, this time using reverse-mode automatic differentiation for backpropagation.

This turned out to be a highly rewarding experience. Not only did I gain a deeper understanding of automatic differentiation and its core principles, but I also solidified my knowledge of computational graph construction, gradient flow, and optimization techniques. Revisiting these topics from the ground up gave me a renewed appreciation for how abstracted concepts in high-level frameworks are built from first principles.

# Requirements

- This library was built and tested with:
  `numpy==1.24.3`
- The tests were built with: `torch==2.2.2`
- `attention.py` is a PyTorch implementation of a single attention mechanism and was built with `torch==2.2.2`

# Project Structure

- `PuruTorch`
  - `utils.py`: Contains `ContextManager` class for managing state context during forward calls, `Functional` superclass which all diffrentiable `Tensor` operations inherit from and a helper function for unbroadcasting gradients.
  - `tensor.py`: Contains the `Tensor` class and `Parameter` sub-class.
  - `functional.py`: Contains diffrentiable primitive operations with bindings to the `Tensor` class.
  - `nn`
    - `module.py`: Contains `Module` class, a superclass to various nn primitives. Manages train/eval state and tracks learnable parameters.
    - `act_functional.py`: Contains some activation functionals.
    - `activation.py`: Contains `Identity`, `Sigmoid`, `ReLU`, `Tanh` and `Softmax` activation modules.
    - `loss_functional.py`: Contains some loss functionals.
    - `loss.py`: Contains `Loss` superclass and `MSELoss`, `CrossEntropyLoss` and `CTCLoss`.
    - `batchnorm1d.py`: Contains `BatchNorm1D` Module.
    - `batchnorm2d.py`: Contains `BatchNorm2D` Module.
    - `dropout_functional.py`: Contains some dropout functionals.
    - `dropout.py`: Contains `Dropout`, `Dropout1D` and `Dropout2D` modules.
    - `linear.py`: Contains `Linear` module.
    - `conv_functional.py`: Contains some conv functionals.
    - `pool.py`: Contains `MaxPool1D`, `MeanPool1D`, `MaxPool2D`, `MeanPool2D` modules.
    - `conv1d.py`: Contains `Conv1D` module.
    - `conv2d.py`: Contains `Conv2D` module.
    - `convtranspose1d.py`: Contains `ConvTranspose1D` module.
    - `convtranspose2d.py`: Contains `ConvTranspose2D` module.
    - `rnncell.py`: Contains `RNNCell` module.
    - `grucell.py`: Contains `GRUCell` module.
    - `CTCDecoding.py`: Contains `GreedySearchDecoder` and `BeamSearchDecoder`, numpy implementations of greedy search and beam search algorithms to decode the predicted sequences from a model.
    - `attention.py`: Contains `Attention`, a PyTorch implementation of a single attention mechanism.
  - `models`
    - `MLP.py`: A simple `MLP` model built with this library.
    - `CNN.py`: A simple `CNN` model built with this library.
    - `ResBlock`: A `ResBlock` Module built with this library.
    - `RNNClassifier.py`: An `RNNClassifier` model built with this library.
    - `GRUClassifier.py`: A `GRUClassifier` model built with this library.
  - `optim`
    - `optimizer.py`: Contains `Optimizer` superclass responsible for updating model parameters after gradient computation and zero-ing out gradients.
    - `sgd.py`: Contains `SGD` Optimizer.
    - `adam.py`: Contains `Adam` Optimizer.
    - `adamW.py`: Contains `AdamW` Optimizer.
- `tests`: Contains tests for correctness and can serve as examples of how to use the library. All tests can be run with `python tests/runner.py`. The following parts of the library have been sucessfully tested against equivalent PyTorch implementations for correctness:

```
Summary:
+--------------------------------------------+-------------+
|                    Test                    |    Score    |
+--------------------------------------------+-------------+
| Functional: Add Forward/Backward           |      1      |
| Functional: Neg Forward/Backward           |      1      |
| Functional: Sub Forward/Backward           |      1      |
| Functional: Mul Forward/Backward           |      1      |
| Functional: Div Forward/Backward           |      1      |
| Functional: Pow Forward/Backward           |      1      |
| Functional: Transpose Forward/Backward     |      1      |
| Functional: Reshape Forward/Backward       |      1      |
| Functional: Squeeze Forward/Backward       |      1      |
| Functional: Unsqueeze Forward/Backward     |      1      |
| Functional: Matmul Forward/Backward        |      1      |
| Functional: Slice Forward/Backward         |      1      |
| Functional: Log Forward/Backward           |      1      |
| Functional: Exp Forward/Backward           |      1      |
| Functional: Sum Forward/Backward           |      1      |
| Functional: Mac Forward/Backward           |      1      |
| Functional: Mean Forward/Backward          |      1      |
| Functional: Var Forward/Backward           |      1      |
| Activation: Identity Forward/Backward      |      1      |
| Activation: Sigmoid Forward/Backward       |      1      |
| Activation: ReLU Forward/Backward          |      1      |
| Activation: Tanh Forward/Backward          |      1      |
| Activation: Softmax Forward/Backward       |      1      |
| Loss: MSELoss Forward/Backward             |      1      |
| Loss: CELoss Forward/Backward              |      1      |
| Loss: CTCLoss Forward/Backward             |      1      |
| Reg: Batchnorm1d Forward/Backward          |      1      |
| Reg: Batchnorm2d Forward/Backward          |      1      |
| Reg: Dropout Forward/Backward              |      1      |
| Reg: Dropout1D Forward/Backward            |      1      |
| Reg: Dropout2D Forward/Backward            |      1      |
| Resampling: Upsample1D Forward/Backward    |      1      |
| Resampling: Downsample1D Forward/Backward  |      1      |
| Resampling: Upsample2D Forward/Backward    |      1      |
| Resampling: Downsample2D Forward/Backward  |      1      |
| Pooling: MaxPool1D Forward/Backward        |      1      |
| Pooling: MeanPool1D Forward/Backward       |      1      |
| Pooling: MaxPool2D Forward/Backward        |      1      |
| Pooling: MeanPool2D Forward/Backward       |      1      |
| Layer: Linear Forward/Backward             |      1      |
| Layer: Conv1D Forward/Backward             |      1      |
| Layer: Conv2D Forward/Backward             |      1      |
| Layer: ConvTranspose1D Forward/Backward    |      1      |
| Layer: ConvTranspose2D Forward/Backward    |      1      |
| Layer: RNNCell Forward/Backward            |      1      |
| Layer: GRUCell Forward/Backward            |      1      |
| Optim: SGD Step                            |      1      |
| Optim: Adam Step                           |      1      |
| Optim: AdamW Step                          |      1      |
| Model: MLP Forward/Backward                |      1      |
| Model: CNN Forward/Backward                |      1      |
| Model: ResBlock Forward/Backward           |      1      |
| Model: RNNClassifier Forward/Backward      |      1      |
| Model: GRUClassifier Forward/Backward      |      1      |
| Decoding: Greedy Search (Numpy impl.)      |      1      |
| Decoding: Beam Search (Numpy impl.)        |      1      |
| Attention Forward/Backward (Pytorch impl.) |      1      |
+--------------------------------------------+-------------+
|                   TOTAL                    |    57/57    |
+--------------------------------------------+-------------+
```

# Usage

Usage is similar to PyTorch! Here are some examples:

## Simple Example:

{% raw %}

```python
import PuruTorch as pt

# initialize 3 Tensors
x = pt.Tensor.random.uniform(0.0, 1.0, size=(5,6), requires_grad=True)
y = pt.Tensor.random.uniform(0.0, 1.0, size=(6,5), requires_grad=True)
z = pt.Tensor.random.uniform(0.0, 1.0, size=(5,),  requires_grad=True)

# calculations
out = x @ y
out += z

# initiate backprop
out.backward()

# get gradients
print(x.grad)
print(y.grad)
print(z.grad)
```

{% endraw %}

## Complex Example (building a ResBlock):

{% raw %}

```python
from PuruTorch import Tensor, Module
from PuruTorch.nn import Identity, ReLU, BatchNorm2D, Conv2D, CrossEntropyLoss
from PuruTorch.optim import AdamW
import numpy as np

# Create a ResBlock
class ConvBn2D(Module):
    def __init__(self, in_channels:int, out_channels:int,
                kernel_size:int, stride:int=1, padding:int=1) -> None:
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)
        self.bn   = BatchNorm2D(num_features=out_channels)

    def forward(self, x : Tensor) -> Tensor:
        out = self.bn(self.conv(x))
        return out


class ResBlock(Module):
    '''
    A Residual block built with PuruTorch library.
    '''
    def __init__(self, in_channels:int, out_channels:int,
                kernel_size:int, stride:int=3, padding:int=1):
        super().__init__()
        self.cbn1 = ConvBn2D(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding)
        self.cbn2 = ConvBn2D(out_channels, out_channels, kernel_size=kernel_size,
                            stride=1, padding=padding)

        if stride != 1 or in_channels != out_channels or kernel_size != 1 or padding != 0:
            self.residual =  ConvBn2D(in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=1)
        else:
            self.residual =  Identity()

        self.act = ReLU()

    def forward(self, x : Tensor) -> Tensor:
        identity = x

        out = self.act(self.cbn1(x))
        out = self.cbn2(out)
        out = self.residual(identity) + out

        out = self.act(out)
        return out


# create input/target Tensor's from numpy array
batch, in_c, in_width, out_c, out_width = 10, 24, 128, 3, 26
npx = np.random.randn(batch,  in_c, in_width,   in_width)
npy = np.random.randn(batch, out_c, out_width, out_width)
x = Tensor.tensor(npx, requires_grad=True)
y = Tensor.tensor(npy, requires_grad=True)

# init model
model = ResBlock(in_channels=in_c, out_channels=out_c,
                kernel_size=5, stride=5, padding=5//2)

#loss function and optimizer
criterion = CrossEntropyLoss(reduction='mean')
optim     = AdamW(model.parameters(), lr=0.001, betas=[0.9, 0.99], weight_decay=0.01)

# pass input through model
z = model(x)

# get loss
loss = criterion(z, y)

# backpropagate
loss.backward()

# step
optim.step()

# reset gradients to zero
optim.zero_grad()

```

{% endraw %}

## Printing the Computational Graph

You can also call the `Tensor.bfs` on any node `Tensor` which will kick start a breadth-first traversal of the computational graph and print the node `Tensor`'s shape, the `Functional` used to create it, and the input `Tensor`s to that `Functional`. For ex, calling `Tensor.bfs(loss)` in the above example would print:

{% raw %}

```
Depth: 0
____________________
| out_shape: ()
| op: <PuruTorch.nn.loss_functional.SoftmaxCrossEntropy object at 0x7f7f30a6e310>
| inp1_shape: (10, 3, 26, 26)
____________________
* Depth: 1
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.act_functional.ReLU object at 0x7f7f30a6e250>
| inp1_shape: (10, 3, 26, 26)
____________________
** Depth: 2
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Add object at 0x7f7f30a6e190>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (10, 3, 26, 26)
____________________
*** Depth: 3
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Add object at 0x7f7f30a6e0d0>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
*** Depth: 3
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Add object at 0x7f7f30a67fd0>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
**** Depth: 4
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Mul object at 0x7f7f30a6bf10>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
**** Depth: 4
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a6bfd0>
| inp1_shape: (3,)
____________________
**** Depth: 4
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Mul object at 0x7f7f30a67e50>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
**** Depth: 4
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a67f10>
| inp1_shape: (3,)
____________________
***** Depth: 5
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Div object at 0x7f7f30a6bd90>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
***** Depth: 5
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a6be50>
| inp1_shape: (3,)
____________________
***** Depth: 5
____________________
| out_shape: (3,)
| op: None
____________________
***** Depth: 5
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Div object at 0x7f7f30a67cd0>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
***** Depth: 5
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a67d90>
| inp1_shape: (3,)
____________________
***** Depth: 5
____________________
| out_shape: (3,)
| op: None
____________________
****** Depth: 6
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Sub object at 0x7f7f30a6ba60>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
****** Depth: 6
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Pow object at 0x7f7f30a6bca0>
| inp1_shape: (1, 3, 1, 1)
____________________
****** Depth: 6
____________________
| out_shape: (3,)
| op: None
____________________
****** Depth: 6
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Sub object at 0x7f7f30a679a0>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
****** Depth: 6
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Pow object at 0x7f7f30a67be0>
| inp1_shape: (1, 3, 1, 1)
____________________
****** Depth: 6
____________________
| out_shape: (3,)
| op: None
____________________
******* Depth: 7
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.conv_functional.Downsample2D object at 0x7f7f30a6b310>
| inp1_shape: (10, 3, 130, 130)
____________________
******* Depth: 7
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a53460>
| inp1_shape: (3,)
____________________
******* Depth: 7
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Add object at 0x7f7f30a6bbe0>
| inp1_shape: (1, 3, 1, 1)
| inp2_shape: (1,)
____________________
******* Depth: 7
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.conv_functional.Downsample2D object at 0x7f7f30a67250>
| inp1_shape: (10, 3, 26, 26)
____________________
******* Depth: 7
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a530d0>
| inp1_shape: (3,)
____________________
******* Depth: 7
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Add object at 0x7f7f30a67b20>
| inp1_shape: (1, 3, 1, 1)
| inp2_shape: (1,)
____________________
******** Depth: 8
____________________
| out_shape: (10, 3, 130, 130)
| op: <PuruTorch.nn.conv_functional.Conv2D_stride1 object at 0x7f7f30a6b280>
| inp1_shape: (10, 24, 130, 130)
| inp2_shape: (3, 24, 1, 1)
| inp3_shape: (3,)
____________________
******** Depth: 8
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Mean object at 0x7f7f30a6b430>
| inp1_shape: (10, 3, 26, 26)
____________________
******** Depth: 8
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a6bb20>
| inp1_shape: (3,)
____________________
******** Depth: 8
____________________
| out_shape: (1,)
| op: None
____________________
******** Depth: 8
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.conv_functional.Conv2D_stride1 object at 0x7f7f30a671c0>
| inp1_shape: (10, 3, 30, 30)
| inp2_shape: (3, 3, 5, 5)
| inp3_shape: (3,)
____________________
******** Depth: 8
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Mean object at 0x7f7f30a67370>
| inp1_shape: (10, 3, 26, 26)
____________________
******** Depth: 8
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a67a60>
| inp1_shape: (3,)
____________________
******** Depth: 8
____________________
| out_shape: (1,)
| op: None
____________________
********* Depth: 9
____________________
| out_shape: (10, 24, 130, 130)
| op: <PuruTorch.nn.conv_functional.Pad2D object at 0x7f7f30a6b0d0>
| inp1_shape: (10, 24, 128, 128)
____________________
********* Depth: 9
____________________
| out_shape: (3, 24, 1, 1)
| op: None
____________________
********* Depth: 9
____________________
| out_shape: (3,)
| op: None
____________________
********* Depth: 9
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Var object at 0x7f7f30a6b4c0>
| inp1_shape: (10, 3, 26, 26)
____________________
********* Depth: 9
____________________
| out_shape: (10, 3, 30, 30)
| op: <PuruTorch.nn.conv_functional.Pad2D object at 0x7f7f30a5afd0>
| inp1_shape: (10, 3, 26, 26)
____________________
********* Depth: 9
____________________
| out_shape: (3, 3, 5, 5)
| op: None
____________________
********* Depth: 9
____________________
| out_shape: (3,)
| op: None
____________________
********* Depth: 9
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Var object at 0x7f7f30a67400>
| inp1_shape: (10, 3, 26, 26)
____________________
********** Depth: 10
____________________
| out_shape: (10, 24, 128, 128)
| op: None
____________________
********** Depth: 10
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.act_functional.ReLU object at 0x7f7f30a5af10>
| inp1_shape: (10, 3, 26, 26)
____________________
*********** Depth: 11
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Add object at 0x7f7f30a5ae50>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
************ Depth: 12
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Mul object at 0x7f7f30a5acd0>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
************ Depth: 12
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a5ad90>
| inp1_shape: (3,)
____________________
************* Depth: 13
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Div object at 0x7f7f30a5ab50>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
************* Depth: 13
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a5ac10>
| inp1_shape: (3,)
____________________
************* Depth: 13
____________________
| out_shape: (3,)
| op: None
____________________
************** Depth: 14
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.functional.Sub object at 0x7f7f30a5a820>
| inp1_shape: (10, 3, 26, 26)
| inp2_shape: (1, 3, 1, 1)
____________________
************** Depth: 14
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Pow object at 0x7f7f30a5aa60>
| inp1_shape: (1, 3, 1, 1)
____________________
************** Depth: 14
____________________
| out_shape: (3,)
| op: None
____________________
*************** Depth: 15
____________________
| out_shape: (10, 3, 26, 26)
| op: <PuruTorch.nn.conv_functional.Downsample2D object at 0x7f7f30a5a070>
| inp1_shape: (10, 3, 128, 128)
____________________
*************** Depth: 15
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a49d00>
| inp1_shape: (3,)
____________________
*************** Depth: 15
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Add object at 0x7f7f30a5a9a0>
| inp1_shape: (1, 3, 1, 1)
| inp2_shape: (1,)
____________________
**************** Depth: 16
____________________
| out_shape: (10, 3, 128, 128)
| op: <PuruTorch.nn.conv_functional.Conv2D_stride1 object at 0x7f7f30a53f70>
| inp1_shape: (10, 24, 132, 132)
| inp2_shape: (3, 24, 5, 5)
| inp3_shape: (3,)
____________________
**************** Depth: 16
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Mean object at 0x7f7f30a5a1c0>
| inp1_shape: (10, 3, 26, 26)
____________________
**************** Depth: 16
____________________
| out_shape: (1, 3, 1, 1)
| op: <PuruTorch.functional.Reshape object at 0x7f7f30a5a8e0>
| inp1_shape: (3,)
____________________
**************** Depth: 16
____________________
| out_shape: (1,)
| op: None
____________________
***************** Depth: 17
____________________
| out_shape: (10, 24, 132, 132)
| op: <PuruTorch.nn.conv_functional.Pad2D object at 0x7f7f30a53520>
| inp1_shape: (10, 24, 128, 128)
____________________
***************** Depth: 17
____________________
| out_shape: (3, 24, 5, 5)
| op: None
____________________
***************** Depth: 17
____________________
| out_shape: (3,)
| op: None
____________________
***************** Depth: 17
____________________
| out_shape: (3,)
| op: <PuruTorch.functional.Var object at 0x7f7f30a5a250>
| inp1_shape: (10, 3, 26, 26)
____________________
```

{% endraw %}
