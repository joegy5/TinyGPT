import torch
from torch import nn
import triton
import triton.language as tl
import argparse
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def rope_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    denom_const: tl.constexpr
):
    pid = tl.program_id(0)
    block_id = 2 * pid
    block_start = x_ptr + block_id
    offsets = tl.arange(0, BLOCK_SIZE)
    block_ptrs, mask = block_start + offsets, offsets < n_cols
    block = tl.load(block_ptrs, mask)

    pos_row = tl.floor((1. * block_id) / n_cols).to(tl.int32)
    pos, dim = pos_row % n_rows, block_id % n_cols
    angle = tl.exp(tl.log(pos-0.0) - (1.0 / n_cols) * tl.log(denom_const-0.0) * dim)
    sin, cos = tl.full((1,1), tl.sin(angle), dtype=tl.float32), tl.full((1,1), tl.cos(angle), dtype=tl.float32)
    rot1, rot2, r_block = tl.cat(cos, -sin, dim=1), tl.cat(sin, cos, dim=1), tl.reshape(block, (1, 2))
    out1, out2 = tl.sum(rot1 * r_block, axis=1), tl.sum(rot2 * r_block, axis=1)
    out = tl.cat(out1, out2, dim=0)

    out_start = out_ptr + 2 * pid
    out_ptrs = out_start + offsets
    tl.store(out_ptrs, out, mask)

    
def rope(X: torch.Tensor, denom_const=10000):
    b_sz, n_heads, n_rows, n_cols = X.shape
    BLOCK_SIZE = 2 # each block is a pair of dimensions in each token embedding vector within the Q/K tensor
    grid_sz = b_sz * n_heads * n_rows * n_cols // 2
    grid = (grid_sz,)

    out = torch.empty_like(X)
    rope_kernel[grid](X, out, n_rows, n_cols, BLOCK_SIZE, denom_const)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--h_dim", type=int, default=1, help="head dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=1, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        q_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        k_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32)
        
        q, k = rope(q_in.to(DEVICE)), rope(k_in.to(DEVICE))

        print(f"TEST #{test_no} ------------")
        print(f"OUT_Q:\n{q}") 
        print(f"OUT_K:\n{k}") 

