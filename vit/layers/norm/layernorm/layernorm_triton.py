import argparse
import torch
from torch import nn
import triton
import triton.language as tl

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def layernorm_bwd_db_kernel(
    dz_ptr, db_ptr, 
    dz_col_stride, dz_row_stride,
    db_col_stride,
    b_sz, n_rows,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    dz_block_start = dz_ptr + pid * dz_col_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < b_sz * n_rows
    
    dz_block = tl.load(dz_block_start + offsets * dz_row_stride , mask, other=0.0)
    db_block = tl.sum(dz_block)
    
    db_block_start = db_ptr + pid * db_col_stride
    tl.store(db_block_start, db_block)


def layernorm_bwd_db(dZ: torch.Tensor):
    b_sz, n_rows, n_cols = dZ.shape
    BLOCK_SIZE = triton.next_power_of_2(b_sz * n_rows)
    grid_sz = n_cols
    
    db = torch.empty((1, 1, n_cols)).to(DEVICE)
    dZ = dZ.to(DEVICE)

    grid = (grid_sz,)
    layernorm_bwd_db_kernel[grid](
        dZ, db, 
        dZ.stride(2), dZ.stride(1),
        db.stride(2),
        b_sz, n_rows, BLOCK_SIZE
    )

    return db 

@triton.jit
def layernorm_bwd_dgamma_kernel(
    dz_ptr, x_ptr, dgamma_ptr, mean_ptr, var_ptr,
    dz_col_stride, dz_row_stride,
    x_col_stride, x_row_stride,
    mean_col_stride, mean_row_stride,
    var_col_stride, var_row_stride,
    dgamma_col_stride,
    b_sz, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    dz_block_start = dz_ptr + pid * dz_col_stride
    x_block_start = x_ptr + pid * x_col_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < b_sz * n_rows

    mean_block = tl.load(mean_ptr + offsets * mean_row_stride, mask, other=0.0)
    var_block = tl.load(var_ptr + offsets * var_row_stride, mask, other=0.0)
    dz_block = tl.load(dz_block_start + offsets * dz_row_stride, mask, other=0.0)
    x_block = tl.load(x_block_start + offsets * x_row_stride, mask, other=0.0)

    x_hat_block = (x_block - mean_block) / var_block
    dgamma_block = tl.sum(dz_block * x_hat_block)

    dgamma_block_start = dgamma_ptr + pid * dgamma_col_stride
    tl.store(dgamma_block_start, dgamma_block)

def layernorm_bwd_dgamma(dZ: torch.Tensor, X: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
    b_sz, n_rows, n_cols = dZ.shape
    BLOCK_SIZE = triton.next_power_of_2(b_sz * n_rows)
    grid_sz = n_cols

    dgamma = torch.empty((1, 1, n_cols)).to(DEVICE)
    X, dZ, mean, var = X.to(DEVICE), dZ.to(DEVICE), mean.to(DEVICE), var.to(DEVICE)

    grid = (grid_sz,)
    layernorm_bwd_dgamma_kernel[grid](
        dZ, X, dgamma, mean, var,
        dZ.stride(2), dZ.stride(1),
        X.stride(2), X.stride(1),
        mean.stride(2), mean.stride(1),
        var.stride(2), var.stride(1),
        dgamma.stride(2),
        b_sz, n_rows, n_cols,
        BLOCK_SIZE
    )
    return dgamma
    

@triton.jit
def layernorm_fwd_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr,
    x_stride, out_stride, mean_stride, var_stride,
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
    mean = tl.sum(block) / n_cols
    block_var = tl.load(block_ptrs, mask, other=mean)
    diff = block_var - mean
    var = tl.sqrt((tl.sum(diff * diff) / n_cols) + eps)

    out = gamma * (diff / var) + beta

    # STEP 4: write the results back to the correct output memory addresses
    out_start = out_ptr + pid * out_stride
    out_ptrs = out_start + offsets
    tl.store(out_ptrs, out, mask)

    mean_start = mean_ptr + pid * mean_stride
    tl.store(mean_start, mean)

    var_start = var_ptr + pid * var_stride
    tl.store(var_start, var)


def layernorm_fwd(X: torch.Tensor):
    X = X.to(DEVICE)
    gamma, beta = torch.ones((1, 1, X.shape[2])).to(DEVICE), torch.zeros((1, 1, X.shape[2])).to(DEVICE)
    b_sz, n_rows, n_cols = X.shape
    out, mean, var = torch.empty_like(X).to(DEVICE), torch.empty((b_sz, n_rows, 1)).to(DEVICE), torch.empty((b_sz, n_rows, 1)).to(DEVICE)

    # need to have 1 block for each row within each example in the batch
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # block size must be power of 2
    grid = (b_sz * n_rows,)
    
    # NOTE: no need to care about b_sz within the kernel; remember that X is just long array of elements in memory
    #   -> as long as we keep traversing rows, we'll even traverse into rows within the next example
    layernorm_fwd_kernel[grid](X, out, mean, var, X.stride(1), out.stride(1), mean.stride(1), var.stride(1), gamma, beta, n_cols, BLOCK_SIZE)
    return out, mean, var


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=0, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        input = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        out, mean, var = layernorm_fwd(input)
        
        print(f"TEST #{test_no} ------------")
        print(f"OUTPUT:\n{out}")
        print(f"MEAN:\n{mean}")

        dZ = torch.randn(size=(args.b_dim, args.n_dim+1, args.d_dim), dtype=torch.float32)
        
        db = layernorm_bwd_db(dZ)
        dgamma = layernorm_bwd_dgamma(dZ, input, mean, var)

        print(f"BACKWARD PASS db: {db}")
        print(f"BACKWARD PASS dgamma: {dgamma}") 
        






