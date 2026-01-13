import torch
from torch import nn, Tensor

class SinusoidalPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super(SinusoidalPE, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos = torch.arange(seq_len).unsqueeze(1).expand(-1, d_model).unsqueeze(0) # (1, N, d_model)
        self.div = (10000 ** (2 * torch.arange(end=seq_len, step=2).unsqueeze(0).expand(seq_len, -1) / d_model)).unsqueeze(0) # (1, N, d_model / 2)
        self.sin = torch.sin(self.pos[:, ::2] / self.div)
        self.cos = torch.cos(self.pos[:, 1::2] / self.div)

    def forward(self, X: Tensor) -> Tensor:
        X[:, :, ::2] += self.sin # Broadcasted to (B, N, d_model)
        X[:, :, 1::2] += self.cos # Broadcasted to (B, N, d_model)
        return X # (B, N, d_model)
    

class RoPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super(RoPE, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class DecoupledRoPE(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DecoupledRoPE, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class ALiBi(nn.Module):
    def __init__(self, seq_len, d_model):
        super(ALiBi, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass
