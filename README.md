# PuruTorch

A deep learning library with reverse-mode automatic differentiation implemented from scratch in Python with Numpy.

# Motivation

As a student in CMU’s ‘11-785: Introduction to Deep Learning’ course, I gained hands-on experience coding various deep learning primitives through explicit forward and backward passes. The following summer, I was hired as a TA for the course and was responsible for the optional Autograd 1/2 assignments, where students would implement automatic differentiation for a small subset of deep learning modules. Although I initially opted out of this assignment due to a busy semester, my role as a lead TA exposed me to the elegance and efficiency of reverse-mode automatic differentiation for computing gradients. Fascinated by its simplicity and inspired by a desire to reinforce my learning, I decided to re-implement the deep learning models I had previously coded, this time using reverse-mode automatic differentiation for backpropagation.

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
