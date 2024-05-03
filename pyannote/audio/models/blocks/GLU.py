import torch.nn as nn


class GLU(nn.Module):
    """
    GLU(X) = (X ∗ W + b) ⊗ σ(X ∗ V + c)
    Gated linear Unit, Activation to use with convolution to create GCNN
    Need fixed time segments
    in_size: length of sequence
    fast: Use the fast version (only compute one matrix, slightly worse)
    """

    def __init__(self, in_size, fast=False):
        self.fast = fast
        self.in_size = in_size
        if self.fast:
            self.linear = nn.Linear(in_size, 2 * in_size)

        else:
            self.linear1 = nn.Linear(in_size, in_size)
            self.linear2 = nn.Linear(in_size, in_size)

    def forward(self, X):
        """
        X: (batch,frames,time)

        """
        if self.fast:
            projection = self.linear(X)
            gated = (
                projection[:, : self.in_size] * projection[:, self.in_size :].sigmoid()
            )
        else:
            projection1 = self.linear1(X)
            projection2 = self.linear2(X)
            gated = projection1 * projection2.sigmoid()
        return gated
