import torch
from torch import nn
import argparse

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8, use_triton=False):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.use_triton = use_triton
        self.gamma = nn.Parameter(torch.ones((1, 1, d_model)))
        self.beta = nn.Parameter(torch.ones((1, 1, d_model)))

    def forward(self, X): 
        # X: (B, N+1, D)
        mean, var = X.mean(2, keepdim=True), X.var(2, keepdim=True) # (B, N+1, 1)
        X = (X - mean) / (torch.sqrt(var + self.eps)) # (B, N+1, D)
        X = self.gamma * X + self.beta # (B, N+1, D)
        return X
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()

    layer_norm = LayerNorm(
        d_model=args.d_dim
    )
    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        out = layer_norm(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}")

