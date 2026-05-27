import torch
from torch import nn
import triton
import triton.language as tl
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def rmsnorm_backward_kernel(
    dZ_ptr, dX_ptr, dgamma_ptr, 
    X_ptr, gamma_ptr, denom_ptr,
    dZ_stride, dX_stride, dgamma_stride, 
    X_stride, gamma_stride, denom_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    # load current dZ, X, gamma blocks into memory
    pid = tl.program_id(0)
    
    dZ_block_start = dZ_ptr + pid * dZ_stride
    X_block_start = X_ptr + pid * X_stride

    gamma_block_start = gamma_ptr + pid * gamma_stride
    denom_block_start = denom_ptr + pid * denom_stride

    offsets = tl.arange(0, BLOCK_SIZE)

    dZ_block_ptrs = dZ_block_start + offsets
    X_block_ptrs = X_block_start + offsets

    mask = offsets < n_cols
    dZ_block = tl.load(dZ_block_ptrs, mask, other=0.0)
    X_block = tl.load(X_block_ptrs, mask, other=0.0)

    gamma_block = tl.load(gamma_block_start)
    denom_block = tl.load(denom_block_start)

    # Compute derivatives using chain rule
    dgamma_block = tl.sum(1. * X_block / denom_block, axis=0, keep_dims=True)
    dX_block = gamma_block * X_block * (2 * denom_block - (1. * X_block / n_cols) * (1. / denom_block)) / (denom_block * denom_block)


    # Write results back to memory
    dX_block_start = dX_ptr + pid * dX_stride
    dgamma_block_ptr = dgamma_ptr + pid * dgamma_stride + tl.arange(0, 1)
    dX_block_ptrs = dX_block_start + offsets

    tl.store(dX_block_ptrs, dX_block, mask)
    tl.store(dgamma_block_ptr, dgamma_block)


def rmsnorm_backward(dZ: torch.Tensor, X: torch.Tensor, gamma: torch.Tensor, denom: torch.Tensor):
    dZ, X, gamma, denom = dZ.to(DEVICE), X.to(DEVICE), gamma.to(DEVICE), denom.to(DEVICE)
    b_sz, n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    dX, dgamma = torch.empty_like(X).to(DEVICE), torch.empty_like(gamma).to(DEVICE)
    
    grid = (b_sz * n_rows,)
    rmsnorm_backward_kernel[grid](
            dZ, dX, dgamma, 
            X, gamma, denom,
            dZ.stride(1), dX.stride(1), dgamma.stride(1),
            X.stride(1), gamma.stride(1), denom.stride(1),
            n_cols, BLOCK_SIZE
    ) 
    
    return dX, dgamma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=1, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=4, help="embedding dimension of the test input")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        dZ = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        X = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        gamma = torch.randn(size=(1, args.n_dim+1, 1), dtype=torch.float32)
        denom = torch.randn(size=(args.b_dim, args.n_dim+1, 1), dtype=torch.float32)

        dX, dgamma = rmsnorm_backward(dZ, X, gamma, denom)
        
        print(f"TEST #{test_no} ------------")
        print(f"dX:\n{dX}")
        print(f"dgamma:\n{dgamma}")





