import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = nn.Linear(vocab_size, d_model)

    def forward(self, X: Tensor) -> Tensor:
        # X dim - (B, N)
        X = F.one_hot(X, num_classes = self.vocab_size)
        return self.emb(X)
        