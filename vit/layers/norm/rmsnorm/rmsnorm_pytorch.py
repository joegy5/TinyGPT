import torch
from torch import nn
import argparse

class RMSNorm(nn.Module):
    def __init__(self, num_patches, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.num_patches = num_patches
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.randn(size=(1, self.num_patches+1, 1)))

    def forward(self, X):
        # X: (B, N+1, D)
        denom = torch.sqrt((1.0 / self.d_model) * torch.sum(X ** 2, dim=2, keepdim=True) + self.eps) # (B, N, 1)
        X = X / denom # (B, N, D)
        X = self.gamma * X # (B, N, D)
        return X
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()

    rms_norm = RMSNorm(
        num_patches=args.n_dim, 
        d_model=args.d_dim, 
    )
    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        out = rms_norm(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}")