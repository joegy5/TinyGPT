import torch
from torch import nn, Tensor

class TransformerCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TransformerCrossEntropyLoss, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor) -> Tensor:
        pass