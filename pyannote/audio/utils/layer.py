import torch


class Transpose(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class BatchNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gamma = torch.nn.parameter.Parameter(torch.ones(dim))
        self.beta = torch.nn.parameter.Parameter(torch.zeros(dim))
        # torch.nn.init.uniform_(self.gamma)
        # torch.nn.init.uniform_(self.beta)
        self.eps = 1e-7
        print("Custom BN used")

    def forward(self, x):
        # Input : (batch,channel,time)
        top = torch.sub(
            x.permute(2, 1, 0), x.mean(dim=1).mean(dim=1)
        )  # Normalize on time and channel, output: (time,channel,batch)
        bottom = torch.sqrt(
            x.var(dim=1).var(dim=1) + self.eps
        )  # One per batch output: (batch)
        result = ((top / bottom).permute(2, 0, 1) * self.gamma + self.beta).permute(
            0, 2, 1
        )
        # print(f"{result.shape=}")
        return result  # out: (batch,channel,time)
