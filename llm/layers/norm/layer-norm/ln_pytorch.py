import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LayerNormalization, self).__init__()
        self.eps = 1e-8
        self.seq_len = seq_len
        self.d_model = d_model

        self.gamma = nn.Parameter(torch.randn(size=(1, self.seq_len, 1)))
        self.beta = nn.Parameter(torch.randn(size=(1, self.seq_len, 1)))

    def forward(self, X):
        # calculate the mean and variance for each feature across the batch
        mean, var = torch.mean(X, dim=-1, keepdim=True), torch.var(X, dim=-1, keepdim=True) # (B, N, 1)
        X = (X - mean) / torch.sqrt(var + self.eps) # (B, N, d_model)
        X = X * self.gamma + self.beta # (B, N, d_model)
        return X # (B, N, d_model)

