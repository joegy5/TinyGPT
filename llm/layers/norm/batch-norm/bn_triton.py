import torch
import triton
import triton.language as tl

@triton.jit
def bn_kernel_forward(
    in_ptr,
    out_ptr,
    n_rows,
    input_row_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Step 1: get the memory addresses of the relevant data to load into memory
    pid = tl.program_id(axis=0)
    col_start = in_ptr + pid # NOT pid * input_row_stride, just need in_ptr + pid to determine which column current program should operate on
    row_offsets = input_row_stride * tl.arange(0, BLOCK_SIZE) # get each column instead of row
    in_ptrs = col_start + row_offsets

    # Step 2: use those addresses to load the relevant data into memory
    mask = row_offsets < n_rows
    row = tl.load(in_ptrs, mask=mask, other=0.)

    # Step 3: operate on the loaded data
    mean = (1. / n_rows) * tl.sum(row)
    diff = row - mean
    res = diff / tl.sqrt((1. / (n_rows - 1)) * tl.sum(diff ** 2, axis=0) + eps)

    # Step 4: load results back into proper memory addresses
    col_out_start = out_ptr + pid
    out_ptrs = col_out_start + row_offsets
    tl.store(out_ptrs, value=res, mask=mask)


@triton.jit
def bn_kernel_backward():
    pass





class LayerNormTriton(torch.autograd.Function):
    def __init__(self):
        pass

    def forward(self):
        pass

    
