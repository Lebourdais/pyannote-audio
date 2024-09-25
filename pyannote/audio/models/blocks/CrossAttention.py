import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
    """

    def __init__(
        self,
        query_dim,
        cross_attention_dim=None,
    ):
        super().__init__()
        self.query_dim = query_dim
        if cross_attention_dim is None:
            cross_attention_dim = self.query_dim
        inner_dim = self.query_dim * 2
        self.to_q = nn.Linear(self.query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, self.query_dim)

    def forward(self, S1, S2):
        """
        softmax((Wq S2)(Wk S1).T) Wv S1
        """
        query = self.to_q(S2)
        key = self.to_k(S1)
        value = self.to_v(S1)
        mask = torch.softmax(query @ key.permute(0, 2, 1), dim=-1)
        masked = mask @ value
        out = self.to_out(masked)
        return out


if __name__ == "__main__":
    ca = CrossAttention(128)
    s1 = torch.rand(2, 100, 128)
    s2 = torch.rand(2, 100, 128)
    out = ca(s1, s2)
    print(f"{out.shape=}")
