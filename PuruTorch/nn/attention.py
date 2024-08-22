import torch

""" A PyTorch implementation of Attention. Implemented as a part of CMU's 11-785 course """

class Softmax:
    '''
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape
        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)
        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)
        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape
        dLdA = dLdA.reshape(self.N, self.C)
        dLdZ = torch.zeros((self.N, self.C))

        for i in range(self.N):
            J = torch.zeros((self.C, self.C))
            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]
            dLdZ[i, :] = dLdA[i, :] @ J
        return dLdZ.reshape(dLdA_original_shape)


class Attention:

    def __init__(self, weights_keys, weights_queries, weights_values):
        """
        Initialize instance variables.
        input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

        Argument(s)
        -----------
        weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys 
        weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries 
        weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values 
        """

        # Store the given weights as parameters of the class.
        self.W_k = weights_keys    # TODO
        self.W_q = weights_queries  # TODO
        self.W_v = weights_values  # TODO
        self.D_k = torch.Tensor([self.W_k.size()[-1]])

        self.softmax = Softmax()

    def forward(self, X):
        """
        Compute outputs of the self-attention layer.
        Stores keys, queries, values, raw and normalized attention weights.
        batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

        Input
        -----
        X (torch.tensor, dim = (B, T, D)): Input batch

        Return
        ------
        X_new (torch.tensor, dim = (B, T, D_v)): Output batch

        """

        self.X = X

        # Compute the values of Key, Query and Value

        self.Q = torch.tensordot(self.X, self.W_q, dims=1)  # (B, T, D_k)
        self.K = torch.tensordot(self.X, self.W_k, dims=1)  # (B, T, D_k)
        self.V = torch.tensordot(self.X, self.W_v, dims=1)  # (B, T, D_v)

        # Calculate unormalized Attention Scores (logits)

        self.A_w = torch.bmm(self.Q, self.K.permute(0, 2, 1))  # (B, T, T)

        # Create additive causal attention mask and apply mask
        # Hint: Look into torch.tril/torch.triu and account for batch dimension
        attn_mask = torch.triu(torch.ones_like(self.A_w), diagonal=1) * -1e9  # (B, T, T)
        self.A_w += attn_mask

        # Calculate/normalize Attention Scores
        self.A_sig = self.softmax.forward(self.A_w / torch.sqrt(self.D_k))  # (B, T, T)

        # Calculate Attention context
        X_new = torch.bmm(self.A_sig, self.V)  # (B, T, D_v)
        return X_new

    def backward(self, dLdXnew):
        """
        Backpropogate derivatives through the self-attention layer.
        Stores derivatives wrt keys, queries, values, and weight matrices.
        Refer to writeup for notation.
        batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

        Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
        You should permute only the required axes.

        Input
        -----
        dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

        Return
        ------
        dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs

        """

        # Derivatives wrt attention weights (raw and normalized)
        # (B, T, T)
        dLdA_sig = torch.bmm(dLdXnew, self.V.permute(0, 2, 1))
        # (B, T, T)
        dLdA_w = self.softmax.backward(dLdA_sig)/torch.sqrt(self.D_k)

        # Derivatives wrt keys, queries, and value

        # (B, T, D_v)
        self.dLdV = torch.bmm(self.A_sig.permute(0, 2, 1), dLdXnew)
        # (B, T, D_k)
        self.dLdK = torch.bmm(dLdA_w.permute(0, 2, 1), self.Q)
        # (B, T, D_k)
        self.dLdQ = torch.bmm(dLdA_w, self.K)

        # Dervatives wrt weight matrices
        # Remember that you need to sum the derivatives along the batch dimension.
        # (D_k, D)
        self.dLdWq = torch.tensordot(self.dLdQ, self.X, dims=([0, 1], [0, 1]))
        # (D_v, D)
        self.dLdWv = torch.tensordot(self.dLdV, self.X, dims=([0, 1], [0, 1]))
        # (D_k, D)
        self.dLdWk = torch.tensordot(self.dLdK, self.X, dims=([0, 1], [0, 1]))

        # Derivative wrt input
        # (B, T, D)
        dLdX = torch.tensordot(self.dLdV, self.W_v.permute(1, 0), dims=1)
        # (B, T, D)
        dLdX += torch.tensordot(self.dLdK, self.W_k.permute(1, 0), dims=1)
        # (B, T, D)
        dLdX += torch.tensordot(self.dLdQ, self.W_q.permute(1, 0), dims=1)
        return dLdX
