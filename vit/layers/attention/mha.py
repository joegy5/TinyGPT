import torch
from torch import nn
import argparse
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size, num_patches, num_heads, d_model, d_k):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, X):
        assert self.num_heads * self.d_k == self.d_model, "d_model should be equal to num_heads * d_k"

        # X: (B, N+1, D)
        Q = self.W_Q(X).reshape(self.batch_size, self.num_patches + 1, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, H, N+1, d_k)
        K = self.W_K(X).reshape(self.batch_size, self.num_patches + 1, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, H, N+1, d_k)
        V = self.W_V(X).reshape(self.batch_size, self.num_patches + 1, self.num_heads, self.d_k).permute(0, 2, 1, 3) # (B, H, N+1, d_k)
        
        # no masks KV cache needed during inference for ViT
        attn = nn.functional.softmax((Q @ K.permute(0, 1, 3, 2)) / math.sqrt(self.d_k), dim=-1) # (B, H, N+1, N+1)
        out = attn @ V # (B, H, N+1, d_k)
        out = out.permute(0, 2, 1, 3).reshape(self.batch_size, self.num_patches + 1, self.d_model) # (B, N+1, D)
        
        return self.W_O(out) # (B, N+1, D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=12, help="embedding dimension of the test input")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads in the MHA implementation")
    parser.add_argument("--d_k", type=int, default=3, help="intermediate embedding dimension used in the MHA calculations")
    args = parser.parse_args()

    mha = MultiHeadAttention(
        batch_size=args.b_dim,
        num_patches=args.n_dim,
        num_heads=args.n_heads,
        d_model=args.d_dim,
        d_k=args.d_k

    )
    for test_no in range(args.n_tests):
        input = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim))
        out = mha(input)
        out_ref = mha(input, use_ref=True)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT: {input}")
        print(f"OUTPUT: {out}")
        print(f"OUTPUT_REF: {out_ref}")


        






