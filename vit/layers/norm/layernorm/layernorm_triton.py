import argparse
import torch
from torch import nn
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def layernorm_fwd_kernel(
    x_ptr,
    x_row_stride,
    out_ptr,
    out_row_stride,
    mean_ptr,
    mean_row_stride,
    rstd_ptr,
    rstd_row_stride,
    gamma_ptr,
    beta_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    eps=1e-8
):
    pid = tl.program_id(0)

    row_start = x_ptr + pid * x_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(row_start + offsets, mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask, other=0.0)

    mean_block = tl.sum(row) / n_cols
    diff = row - mean_block

    diff_std = tl.where(mask, diff, 0.0)
    var = tl.sum(diff_std * diff_std) / n_cols
    rstd_block = tl.rsqrt(var + eps)

    out = gamma * (diff * rstd_block) + beta

    out_start = out_ptr + pid * out_row_stride
    tl.store(out_start + offsets, out, mask)

    mean_start = mean_ptr + pid * mean_row_stride
    tl.store(mean_start, mean_block)

    rstd_start = rstd_ptr + pid * rstd_row_stride
    tl.store(rstd_start, rstd_block)


def layernorm_fwd(X: torch.Tensor, out: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
    b_sz, n_rows, n_cols = X.shape

    # need to have 1 block for each row within each example in the batch
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # block size must be power of 2
    grid = (b_sz * n_rows,)

    # NOTE: no need to care about b_sz within the kernel; remember that X is just long array of elements in memory
    #   -> as long as we keep traversing rows, we'll even traverse into rows within the next example
    layernorm_fwd_kernel[grid](
        X,
        X.stride(1),
        out,
        out.stride(1),
        mean,
        mean.stride(1),
        rstd,
        rstd.stride(1),
        gamma,
        beta,
        n_cols,
        BLOCK_SIZE
    )
    return out, mean, rstd


@triton.jit
def layernorm_bwd_kernel(
    dz_ptr, 
    dz_row_stride,
    dx_ptr,
    dx_row_stride, 
    dgamma_ptr,
    dgamma_row_stride,
    x_ptr, 
    x_row_stride,
    gamma_ptr,
    mean_ptr, 
    mean_row_stride, 
    rstd_ptr,   
    rstd_row_stride,
    n_cols, 
    BLOCK_SIZE: tl.constexpr, 
    eps=1e-8
):
    pid = tl.program_id(0)
   
    dz_block_start = dz_ptr + pid * dz_row_stride
    x_block_start = x_ptr + pid * x_row_stride
    mean_block_start = mean_ptr + pid * mean_row_stride
    rstd_block_start = rstd_ptr + pid * rstd_row_stride
   
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    dz_block = tl.load(dz_block_start + offsets, mask, other=0.0)
    mean_block = tl.load(mean_block_start)
    rstd_block = tl.load(rstd_block_start)
    x_block = tl.load(x_block_start + offsets, mask, other=mean_block)
    gamma_block = tl.load(gamma_ptr + offsets, mask, other=0.0)

    rd = 1. / n_cols
    xhat_block = rstd_block * (x_block - mean_block)
    dxhat_block = dz_block * gamma_block
    
    dx_block = rstd_block * (dxhat_block - rd * tl.sum(dxhat_block) - rd * xhat_block * tl.sum(dxhat_block * xhat_block))
    dgamma_block = dz_block * xhat_block

    dx_block_start = dx_ptr + pid * dx_row_stride
    tl.store(dx_block_start + offsets, dx_block, mask)

    dgamma_block_start = dgamma_ptr + pid * dgamma_row_stride
    tl.store(dgamma_block_start + offsets, dgamma_block, mask)


def layernorm_bwd(dZ: torch.Tensor, dX: torch.Tensor, X: torch.Tensor, gamma: torch.Tensor, dgamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor):
    b_sz, n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (b_sz * n_rows,)
    layernorm_bwd_kernel[grid](
        dZ,
        dZ.stride(1),
        dX,
        dX.stride(1),
        dgamma,
        dgamma.stride(1),
        X,
        X.stride(1),
        gamma,
        mean,
        mean.stride(1),
        rstd,
        rstd.stride(1),
        n_cols,
        BLOCK_SIZE
    )

    dgamma = torch.sum(dgamma, dim=(0,1))
    db = torch.sum(dZ, dim=(0,1))

    return dX, dgamma, db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=0, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        X = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32).to(DEVICE)
        gamma, beta = torch.ones((1, 1, args.d_dim)).to(DEVICE), torch.zeros((1, 1, args.d_dim)).to(DEVICE)
        dX = torch.empty_like(X).to(DEVICE)
        dgamma = torch.empty_like(X).to(DEVICE)

        out, mean = torch.empty_like(X).to(DEVICE), torch.empty((args.b_dim, args.n_dim + 1, 1)).to(DEVICE)
        rstd = torch.empty_like(mean).to(DEVICE)
        out, mean, rstd = layernorm_fwd(X, out, mean, rstd, gamma, beta)

        print(f"TEST #{test_no} ------------")
        print(f"OUTPUT:\n{out}")
        print(f"MEAN:\n{mean}")

        dZ = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32).to(DEVICE)
        dX, dgamma, db = layernorm_bwd(dZ, dX, X, gamma, dgamma, mean, rstd) 

        print(f"BACKWARD PASS db: {db}")
        print(f"BACKWARD PASS dgamma: {dgamma}")
        print(f"BACKWARD PASS dX: {dX}")

