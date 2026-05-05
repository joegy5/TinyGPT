import torch
from torch import nn
import argparse
import math

class LearnedPE(nn.Module):
    def __init__(self, batch_size, num_patches, d_model):
        super(LearnedPE, self).__init__()
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn((1, num_patches+1, d_model)))
    
    def forward(self, X):
        # X: (B, N, D)
        return X + self.pe 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    learned_pe = LearnedPE(
        batch_size=args.b_dim,
        num_patches=args.n_dim, 
        d_model=args.d_dim, 
    )
    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        out = learned_pe(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}")





