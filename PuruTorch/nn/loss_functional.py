import numpy as np
from ..tensor import Tensor
from ..utils import Function
from typing import List, Union, Literal

# ------------------------------------------
#  Loss Functional
# ------------------------------------------

class SoftmaxCrossEntropy(Function):
    """
    An explicit SoftmaxCrossEntropy functional to reuse values computed in forward.
    """    

    def __call__(self, predictions: Tensor, targets: Tensor, 
                 reduction: Union[None, Literal['mean', 'sum']]="mean") -> Tensor:
        return self.forward(predictions, targets, reduction)

    def forward(self, predictions: Tensor, targets: Tensor, 
                reduction: Union[None, Literal['mean', 'sum']]="mean") -> Tensor:
        super().forward()
        softmax = np.exp(predictions.data) / np.sum(np.exp(predictions.data), axis=-1, keepdims=True)
        data = np.sum(-targets.data * np.log(softmax), axis=-1)
        if reduction is None:
            data = data
            N = 1
        elif reduction == "mean":
            data = np.mean(data)
            *N_dim, _ = predictions.shape
            N = np.prod(N_dim)
        elif reduction == "sum":
            data = np.sum(data)
            N = 1
        requires_grad = predictions.requires_grad
        if requires_grad:
            self.ctx.save_for_backward(predictions)
            self.ctx.reduction = reduction
            self.ctx.targets = targets.data 
            self.ctx.softmax = softmax
            self.ctx.N = N
        return Tensor(data, requires_grad, self if requires_grad else None)
    
    def backward(self, grad_output: Tensor) -> List[Tensor]:
        super().backward()
        a_grad = grad_output.data * (self.ctx.softmax - self.ctx.targets) / self.ctx.N
        return [Tensor.tensor(a_grad)]

class _CTC(object):
    '''
    Helper functions to compute CTCLoss
    '''
    def __init__(self, BLANK:int=0):
        """
        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        """
        self.BLANK = BLANK

    def extend_target_with_blank(self, target:np.ndarray) -> np.ndarray:
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = [self.BLANK]
        skip_connect = [0]
        for i, symbol in enumerate(target):
            extended_symbols.append(symbol)
            skip_connect.append(int(i > 0 and target[i] != target[i-1]))
            extended_symbols.append(self.BLANK)
            skip_connect.append(0)

        N = len(extended_symbols)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))
        return extended_symbols, skip_connect

    def get_forward_probs(self, logits:np.ndarray, extended_symbols:np.ndarray, 
                          skip_connect:np.ndarray) -> np.ndarray:
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextended_symbols[i]]

        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skip_connect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # Intialize alpha[0][0]
        # Intialize alpha[0][1]
        # Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        for t in range(1, T):
            alpha[t][0] = alpha[t - 1][0] * logits[t][extended_symbols[0]]
            for l in range(1, S):
                if bool(skip_connect[l]):
                    alpha[t][l] = alpha[t-1][l] + \
                        alpha[t-1][l-1] + alpha[t-1][l-2]
                else:
                    alpha[t][l] = alpha[t - 1][l] + alpha[t-1][l-1]
                alpha[t][l] *= logits[t][extended_symbols[l]]
        return alpha

    def get_backward_probs(self, logits:np.ndarray, extended_symbols:np.ndarray, 
                           skip_connect:np.ndarray) -> np.ndarray:
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extended_symbols[i]]

        extended_symbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        beta[-1, -1] = 1
        beta[-1, -2] = 1

        for t in range(T - 2, -1, -1):
            beta[t][-1] = beta[t + 1][-1] * logits[t + 1][extended_symbols[-1]]
            for i in range(S - 2, -1, -1):
                if i + 2 < S - 1 and bool(skip_connect[i + 2]):
                    beta[t][i] += beta[t + 1][i] * \
                        logits[t + 1][extended_symbols[i]]
                    beta[t][i] += beta[t + 1][i + 1] * \
                        logits[t + 1][extended_symbols[i + 1]]
                    beta[t][i] += beta[t + 1][i + 2] * \
                        logits[t + 1][extended_symbols[i + 2]]
                else:
                    beta[t][i] += beta[t + 1][i] * \
                        logits[t + 1][extended_symbols[i]]
                    beta[t][i] += beta[t + 1][i + 1] * \
                        logits[t + 1][extended_symbols[i + 1]]

        return beta

    def get_posterior_probs(self, alpha:np.ndarray, beta:np.ndarray) -> np.ndarray:
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        gamma += alpha * beta
        sumgamma += np.sum(gamma, axis=1)
        return gamma / sumgamma.reshape(-1, 1)


class CTCLoss(Function):
    '''
    The Function class used to compute CTCLoss. Called by CTCLoss in nn.Loss
    '''

    def __call__(self, logits:Tensor, target:Tensor, input_lengths:Tensor, target_lengths:Tensor, 
                 BLANK:int=0, reduction: Union[None, Literal['mean', 'sum']]='mean') -> Tensor:
        return self.forward(logits, target, input_lengths, target_lengths, BLANK, reduction)

    def forward(self, logits:Tensor, target:Tensor, input_lengths:Tensor, target_lengths:Tensor, 
                BLANK:int=0, reduction: Union[None, Literal['mean', 'sum']]='mean') -> Tensor:
        """
        CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [Tensor, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [Tensor, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [Tensor, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [Tensor, dim=(batch_size,)]:
            lengths of the target

        BLANK (int, optional): blank label index. Default 0.
        
        Returns
        -------
        loss [Tensor]:
            avg. divergence between the posterior probability and the target

        """
        super().forward()
        ctc = _CTC(BLANK=BLANK)
        # IMP:
        # Output losses should be the mean loss over the batch
        B, _ = target.shape
        total_loss = np.zeros(B)
        extended_symbols = []
        gammas = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target_t = target[batch_itr, :target_lengths[batch_itr].data].data
            logits_t = logits[:input_lengths[batch_itr].data, batch_itr].data
            extended_symbol, skip_connect = ctc.extend_target_with_blank(target_t)
            alpha = ctc.get_forward_probs(logits_t, extended_symbol, skip_connect)
            beta = ctc.get_backward_probs(logits_t, extended_symbol, skip_connect)
            gamma = ctc.get_posterior_probs(alpha, beta)
            for r in range(gamma.shape[1]):
                total_loss[batch_itr] -= np.sum(gamma[:, r] * np.log(logits_t[:, extended_symbol[r]]))
            gammas.append(gamma)
            extended_symbols.append(extended_symbol)

        requires_grad = logits.requires_grad
        if requires_grad:
            self.ctx.save_for_backward(logits)
            self.ctx.input_lengths = input_lengths.data
            self.ctx.target_lengths = target_lengths.data
            self.ctx.gammas = gammas
            self.ctx.extended_symbols = extended_symbols

        if reduction is None:
            return Tensor.tensor(total_loss, requires_grad, self if requires_grad else None)
        elif reduction == "sum":
            return Tensor.tensor(np.sum(total_loss), requires_grad, self if requires_grad else None)
        elif reduction == "mean":
            return Tensor(np.sum(total_loss) / B, requires_grad, self if requires_grad else None)
    

    def backward(self, grad_output:Tensor):
        """

        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """
        super().backward()
        # No need to modify
        logits = self.ctx.saved_tensors[0]
        T, B, C = logits.shape
        dY = np.full_like(logits.data, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            logits_t = logits[:self.ctx.input_lengths[batch_itr], batch_itr].data
            extended_symbols = self.ctx.extended_symbols[batch_itr]
            gamma = self.ctx.gammas[batch_itr]
            for r in range(gamma.shape[1]):
                dY[:self.ctx.input_lengths[batch_itr], batch_itr, extended_symbols[r]] -= gamma[:, r] / logits_t[:,extended_symbols[r]]

        return [grad_output * Tensor.tensor(dY)]
