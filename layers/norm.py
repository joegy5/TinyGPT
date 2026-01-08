import torch
from torch import nn, Tensor

class BatchNormalization(nn.Module):
    def __init__(self):
        super(BatchNormalization, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class LayerNormalization(nn.Module):
    def __init__(self):
        super(LayerNormalization, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class RMSNormalization(nn.Module):
    def __init__(self):
        super(RMSNormalization, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass

 