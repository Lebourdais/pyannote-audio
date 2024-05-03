import torch.nn as nn
from mamba_ssm import Mamba

from .RMSNorm import RMSNorm


class MambaNet(nn.Module):
    """Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self,
        input_size,
        dropout=0,
    ):
        super(MambaNet, self).__init__()
        parameters = {
            "d_model": input_size,
            "d_state": 8,  # 8 or 16
            "d_conv": 4,
            "expand": 2,
        }
        self.input_size = input_size

        self.dropout = nn.Dropout(p=dropout)
        self.norm = RMSNorm(d=parameters["d_model"], eps=1e-5)
        self.mamba = Mamba(
            d_model=parameters["d_model"],
            d_state=parameters["d_state"],
            d_conv=parameters["d_conv"],
            expand=parameters["expand"],
        )

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        output = inp
        rnn_output = self.norm(self.mamba(output))
        rnn_output = self.dropout(rnn_output)

        return rnn_output
