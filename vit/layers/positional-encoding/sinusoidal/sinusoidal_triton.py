import argparse
import torch
from torch import nn
import triton
import triton.language as tl
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def sinpe_kernel(
    out_ptr, out_stride,
    n_rows: tl.constexpr, n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    den_const: tl.constexpr = 10000
):
    pid = tl.program_id(0)
    dims = tl.arange(0, BLOCK_SIZE // 2)
    pos = pid - n_rows * (pid // n_rows)
    inter = tl.exp(tl.log(pos-0.0) - tl.log(den_const - 0.0) * (2.0 / n_cols) * dims)
    out = tl.interleave(tl.sin(inter), tl.cos(inter))

    out_start = out_ptr + pid * out_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    out_ptrs = out_start + offsets
    tl.store(out_ptrs, out, offsets < n_cols)


def sinpe(X: torch.Tensor, den_cost=10000):
    X = X.to(DEVICE)
    b_sz, n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # each row is a block
    grid = (b_sz * n_rows,)

    out = torch.empty_like(X)
    sinpe_kernel[grid](out, out.stride(1), n_rows, n_cols, BLOCK_SIZE, den_cost)
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=2, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=4, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=8, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        out = input + sinpe(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}") 

