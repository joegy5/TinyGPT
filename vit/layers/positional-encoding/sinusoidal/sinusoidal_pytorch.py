import torch
from torch import nn
import argparse
import math

class SinusoidalPE(nn.Module):
    def __init__(self, num_patches, d_model, denom_const=10000):
        super(SinusoidalPE, self).__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        self.denom_const = denom_const
        self.pe = self._init_pe()

    def _init_pe(self):
        assert self.d_model % 2 == 0, "d_model cannot be odd"

        pos = torch.arange(0, self.num_patches).unsqueeze(-1).expand(-1, self.d_model // 2) # (N, D / 2)
        # original formula might just become zero if denominator gets too large -> apply log to turn into subtraction of terms, then apply exp after
        inter = torch.exp(torch.log(pos) - math.log(self.denom_const) * (2. / self.d_model) * torch.arange(0, self.d_model, 2))
        pe = torch.zeros(size=(self.num_patches, self.d_model))
        pe[:, ::2] = torch.sin(inter)
        pe[:, 1::2] = torch.cos(inter)

        print(f"pe: {pe}")
        return pe # (N, D)
    
    
    def forward(self, X):
        # X: (B, N, D)
        return X + self.pe # self.pe gets broadcasted to (B, N, D) -> final shape is (B, N, D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    sinusoidal_pe = SinusoidalPE(
        num_patches=args.n_dim, 
        d_model=args.d_dim, 
        denom_const=args.denom_const
    )
    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        out = sinusoidal_pe(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}")





