import torch
from torch import nn
import argparse
import math

class RotaryPE(nn.Module):
    def __init__(self, batch_size, num_heads, num_patches, d_k, denom_const=10000):
        super(RotaryPE, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.d_k = d_k
        self.denom_const = denom_const
        self.rot_mat = self._init_pe()

    def _init_pe(self):
        pos = torch.arange(0, self.num_patches).unsqueeze(-1).expand(-1, self.d_k // 2)
        angles = torch.exp(torch.log(pos) - math.log(self.denom_const) * (2. / self.d_k) * torch.arange(0, self.d_k // 2))
        rot_sin, rot_cos = torch.sin(angles).unsqueeze(1), torch.cos(angles).unsqueeze(1)

        rot_mat = torch.cat((rot_cos, -rot_sin, rot_sin, rot_cos), dim=1)
        rot_mat = rot_mat.permute(0, 2, 1).reshape(1, 1, self.num_patches, self.d_k // 2, 2, 2)
        return rot_mat

    def _apply_rotation(self, rot_mat: torch.Tensor, X: torch.Tensor):
        X = X.reshape(self.batch_size, self.num_heads, self.num_patches, self.d_k // 2, 2, 1)
        return (rot_mat @ X).squeeze(-1).reshape(self.batch_size, self.num_heads, self.num_patches, -1)

    def forward(self, Q: torch.Tensor, K: torch.Tensor):
        # Q, K: (B, H, N, d_k)
        # apply the individual R_m and R_n transformations to the q and k vectors so that 
        # downstream MHA automatically applies relative rotation transformation
        return self._apply_rotation(self.rot_mat, Q), self._apply_rotation(self.rot_mat, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--h_dim", type=int, default=2, help="head dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    rope = RotaryPE(
        batch_size=args.b_dim,
        num_heads=args.h_dim,
        num_patches=args.n_dim, 
        d_k=args.d_dim, 
        denom_const=args.denom_const
    )
    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        q_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        k_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        out_q, out_k = rope(q_in, k_in)

        print(f"TEST #{test_no} ------------")
        print(f"OUTPUT_Q:\n{out_q}")
        print(f"OUTPUT_K:\n{out_k}")


