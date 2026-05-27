import argparse
import torch
from torch import nn
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, out_ptr, denom_ptr,
    x_stride, out_stride,
    gamma_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    eps=1e-8
):
    # STEP 1: get memory addresses of all the elements in this block
    pid = tl.program_id(0)
    x_start = x_ptr + pid * x_stride
    gamma_start = gamma_ptr

    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_start + offsets # scalar + tensor becomes tensor
    gamma_ptrs = gamma_start + offsets

    # STEP 2: load elements from those memory addresses
    mask = offsets < n_cols
    x = tl.load(x_ptrs, mask, other=0.0)
    gamma = tl.load(gamma_ptrs, mask, other=0.0)

    # STEP 3: operate on the loaded elements
    var = tl.sum(x * x) / n_cols
    denom = tl.sqrt(var + eps)
    out = gamma * (x / denom)
    
    # STEP 4: write the results back to the correct output memory addresses
    out_start = out_ptr + pid * out_stride
    out_ptrs = out_start + offsets
    
    tl.store(out_ptrs, out, mask)
    tl.store(denom_ptr + pid, denom)

@triton.jit
def rmsnorm_bwd_dgamma_kernel(
    dz_ptr, dgamma_ptr,
    dz_row_stride, dgamma_row_stride,
    dz_col_stride, dgamma_col_stride,
    x_ptr, gamma_ptr, denom_ptr,
    x_row_stride, gamma_row_stride, denom_row_stride,
    x_col_stride, gamma_col_stride, denom_col_stride,
    b_sz, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    dz_block_start = dz_ptr + pid * dz_col_stride
    x_block_start = x_ptr + pid * x_col_stride
    gamma_block_start = gamma_ptr + pid * gamma_col_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < b_sz * n_rows

    dz_block_ptrs = dz_block_start + offsets * dz_row_stride
    x_block_ptrs = x_block_start + offsets * x_row_stride
    denom_block_ptrs = denom_ptr + offsets

    dz_block = tl.load(dz_block_ptrs, mask, other=0.0)
    x_block = tl.load(x_block_ptrs, mask, other=0.0)
    gamma_block = tl.load(gamma_block_start)
    denom_block = tl.load(denom_block_ptrs, mask, other=0.0)

    dgamma_block = tl.sum(dz_block * (x_block / denom_block))
    
    dgamma_block_start = dgamma_ptr + pid * dgamma_col_stride
    tl.store(dgamma_block_start, dgamma_block)

@triton.jit
def rmsnorm_bwd_dx_kernel(
    dz_ptr, dgamma_ptr, dx_ptr,
    dz_row_stride, dgamma_row_stride, dx_row_stride,
    dz_col_stride, dgamma_col_stride,
    x_ptr, gamma_ptr, denom_ptr,
    x_row_stride, gamma_row_stride, denom_row_stride,
    x_col_stride, gamma_col_stride, denom_col_stride,
    b_sz, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    dz_block_start = dz_ptr + pid * dz_row_stride
    x_block_start = x_ptr + pid * x_row_stride
    denom_block_start = denom_ptr + pid * denom_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    dz_block_ptrs = dz_block_start + offsets
    x_block_ptrs = x_block_start + offsets
    gamma_block_ptrs = gamma_ptr + offsets

    dz_block = tl.load(dz_block_ptrs, mask, other=0.0)
    x_block = tl.load(x_block_ptrs, mask, other=0.0)
    gamma_block = tl.load(gamma_block_ptrs, mask, other=0.0)
    denom_block = tl.load(denom_block_start)

    transition = dz_block * gamma_block
    dx_block = transition * denom_block - (x_block / n_cols / denom_block) * tl.sum(transition * x_block)
    dx_block = dx_block / denom_block / denom_block

    dx_block_start = dx_ptr + pid * dx_row_stride
    dx_block_ptrs = dx_block_start + offsets
    tl.store(dx_block_ptrs, dx_block, mask)


def rmsnorm_bwd(dZ: torch.Tensor, X: torch.Tensor, gamma: torch.Tensor, denom: torch.Tensor):
    dZ, X, gamma, denom = dZ.to(DEVICE), X.to(DEVICE), gamma.to(DEVICE), denom.to(DEVICE)
    b_sz, n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(b_sz * n_rows)
    dX, dgamma = torch.empty_like(X).to(DEVICE), torch.empty_like(gamma).to(DEVICE)

    grid = (n_cols,)
    rmsnorm_bwd_dgamma_kernel[grid](
        dZ, dgamma,
        dZ.stride(1), dgamma.stride(1),
        dZ.stride(2), dgamma.stride(2),
        X, gamma, denom,
        X.stride(1), gamma.stride(1), denom.stride(1),
        X.stride(2), gamma.stride(2), denom.stride(2),
        b_sz, n_rows, n_cols, BLOCK_SIZE
    )

    grid = (b_sz * n_rows,)
    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    rmsnorm_bwd_dx_kernel[grid](
        dZ, dgamma, dX,
        dZ.stride(1), dgamma.stride(1), dX.stride(1),
        dZ.stride(2), dgamma.stride(2),
        X, gamma, denom,
        X.stride(1), gamma.stride(1), denom.stride(1),
        X.stride(2), gamma.stride(2), denom.stride(2),
        b_sz, n_rows, n_cols, BLOCK_SIZE
    )

    return dX, dgamma


def rmsnorm_fwd(X: torch.Tensor, gamma: torch.Tensor):
    X, gamma = X.to(DEVICE), gamma.to(DEVICE)
    out = torch.empty_like(X).to(DEVICE)
    b_sz, n_rows, n_cols = X.shape
    denom = torch.empty((b_sz, n_rows, 1)).to(DEVICE)

    # need to have 1 block for each row within each example in the batch
    BLOCK_SIZE = triton.next_power_of_2(b_sz * n_cols) # block size must be power of 2
    grid = (b_sz * n_rows,)
    
    # NOTE: no need to care about b_sz within the kernel; remember that X is just long array of elements in memory
    #   -> as long as we keep traversing rows, we'll even traverse into rows within the next example
    rmsnorm_fwd_kernel[grid](X, out, denom, X.stride(1), out.stride(1), gamma, n_rows, n_cols, BLOCK_SIZE)
    return out, denom


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=0, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()
    

    for test_no in range(args.n_tests):
        print(f"TEST #{test_no} -------------")
        
        X = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        gamma = torch.randn(size=(1, 1, args.d_dim), dtype=torch.float32)
        out_fwd, denom = rmsnorm_fwd(X, gamma)

        print(f"FORWARD PASS OUTPUT: {out_fwd}")
        print(f"FORWARD PASS DENOMINATOR: {denom}")
        
        dZ = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        dX, dgamma = rmsnorm_bwd(dZ, X, gamma, denom)
       
        print(f"BACKWARD PASS dX: {dX}")
        print(f"BACKWARD PASS dgamma: {dgamma}")











