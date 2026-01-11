import torch
from torch import nn, Tensor

class TransformerCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TransformerCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, targets: Tensor) -> Tensor:
        # pred dim - (B, N, d_vocab)
        # targets dim - (B, N)
        return self.loss_fn(pred, targets)