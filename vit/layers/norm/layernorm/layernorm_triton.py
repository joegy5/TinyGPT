import argparse
import torch
from torch import nn
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def layernorm_backward_kernel(

):
    pass


@triton.jit
def layernorm_kernel(
    x_ptr, out_ptr,
    x_stride, out_stride,
    gamma_ptr, beta_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    eps=1e-8
):
    # STEP 1: get memory addresses of all the elements in this block
    pid = tl.program_id(0)
    block_start = x_ptr + pid * x_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_ptrs = block_start + offsets # scalar + tensor becomes tensor
    gamma_ptrs = gamma_ptr + offsets
    beta_ptrs = beta_ptr + offsets

    # STEP 2: load elements from those memory addresses
    mask = offsets < n_cols
    block = tl.load(block_ptrs, mask, other=0.0)
    gamma = tl.load(gamma_ptrs, mask, other=0.0)
    beta = tl.load(beta_ptrs, mask, other=0.0)

    # STEP 3: operate on the loaded elements
    mean = tl.full((BLOCK_SIZE,), tl.sum(block) / n_cols, dtype=tl.float32)
    block_var = tl.load(block_ptrs, mask, other=mean)
    diff = block_var - mean
    var = tl.sum(diff * diff) / (n_cols - 1)
    out = gamma * (diff / tl.sqrt(var + eps)) + beta

    # STEP 4: write the results back to the correct output memory addresses
    out_start = out_ptr + pid * out_stride
    out_ptrs = out_start + offsets
    tl.store(out_ptrs, out, mask)


def layernorm(X: torch.Tensor):
    X = X.to(DEVICE)
    gamma, beta = torch.ones((1, 1, X.shape[2])).to(DEVICE), torch.zeros((1, 1, X.shape[2])).to(DEVICE)
    out = torch.empty_like(X)
    b_sz, n_rows, n_cols = X.shape

    # need to have 1 block for each row within each example in the batch
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # block size must be power of 2
    grid = (b_sz * n_rows,)
    
    # NOTE: no need to care about b_sz within the kernel; remember that X is just long array of elements in memory
    #   -> as long as we keep traversing rows, we'll even traverse into rows within the next example
    layernorm_kernel[grid](X, out, X.stride(1), out.stride(1), gamma, beta, n_cols, BLOCK_SIZE)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=1, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=5, help="embedding dimension of the test input")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        out = layernorm(input)
        print(f"TEST #{test_no} ------------")
        print(f"INPUT:\n{input}")
        print(f"OUTPUT:\n{out}")



