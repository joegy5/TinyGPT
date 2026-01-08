import torch
from torch import nn, Tensor

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass