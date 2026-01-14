import torch
from torch import nn, Tensor

class BatchNormalization(nn.Module):
    def __init__(self, d_model):
        super(BatchNormalization, self).__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(1, 1, self.d_model))
        self.beta = nn.Parameter(torch.ones(1, 1, self.d_model))

    def forward(self, X: Tensor):
        mean, std = torch.mean(X, dim=1), torch.std(X, dim=1)
        return self.gamma * (X - mean) / std + self.beta


class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super(LayerNormalization, self).__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(1, 1, self.d_model))
        self.beta = nn.Parameter(torch.ones(1, 1, self.d_model))

    def forward(self, X):
        mean, std = torch.mean(X, dim=-1), torch.std(X, dim=-1)
        return self.gamma * (X - mean) / std + self.beta


class RMSNormalization(nn.Module):
    def __init__(self, epsilon, d_model):
        super(RMSNormalization, self).__init__()
        self.epsilon = epsilon
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(1, 1, self.d_model))

    def forward(self, X):
        # X dim - (B, N, d_model)
        rms = torch.sqrt(torch.sum(X ** 2, dim=-1) + self.epsilon)
        return self.gamma * (X / rms)

 