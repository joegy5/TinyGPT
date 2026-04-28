import torch
from torch import nn

class RMSNormalization(nn.Module):
    def __init__(self, seq_len, d_model):
        self.eps = 1e-8
        self.seq_len = seq_len
        self.d_model = d_model

        self.gamma = nn.Parameter(torch.randn(size=(1, self.seq_len, 1)))
        self.beta = nn.Parameter(torch.randn(size=(1, self.seq_len, 1)))

    def forward(self, X):
        X = X / torch.sqrt((1 / self.d_model) * torch.sum(X ** 2, dim=-1) + self.eps)
        return X

