import torch
import triton
import triton.language as tl

@triton.jit
def rms_kernel(
    in_ptr,
    out_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Step 1: get the memory addresses of the relevant data to load into memory
    pid = tl.program_id(axis=0)
    row_start = in_ptr + pid * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = row_start + col_offsets

    # Step 2: use those addresses to load the relevant data into memory
    mask = col_offsets < n_cols
    row = tl.load(in_ptrs, mask=mask, other=0.)

    # Step 3: operate on the loaded data
    res = row / tl.sqrt((1. / n_cols) * tl.sum(row ** 2, axis=0) + eps)

    # Step 4: load results back into proper memory addresses
    row_out_start = out_ptr + pid * output_row_stride
    out_ptrs = row_out_start + col_offsets
    tl.store(out_ptrs, value=res, mask=mask)

    
