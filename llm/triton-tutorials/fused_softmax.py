import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit 
# NOTE: we use block-level programming model for kernels - each kernel instance is designed to process only it's own corresponding block
# In this case, each block represents a row in the matrix; since each row's operations are independent, we make a kernel for each row to parallelize across them
def fused_softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # ============== STEP 1: GET ADDRESSES OF DATA TO LOAD INTO MEMORY ==============
    row_idx  = tl.program_id(axis=0)
    # input_row_stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # now that we have the pointer for the start of the row, we need to get the pointers for the rest of the 
    # elements in that row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets # BLOCK_SIZE is the next power of two >= n_cols (ensures all element in row loaded into SRAM)
        
    # ================= STEP 2: LOAD DATA INTO MEMORY ===============
    # set other to negative infinity, so exp of that becomes zero -> doesn't affect softmax result
    # don't just ignore other (not setting it to anything), because it will become zero instead for those extra values, which WILL affect the softmax result
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf')) 

    # ================= STEP 3: OPERATE ON DATA ==================
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_res = numerator / denominator
    softmax_res = softmax_res * softmax_res

    # ================= STEP 4: STORE DATA BACK IN MEMORY ===============
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, value=softmax_res, mask=mask)


def naive_softmax(x: torch.Tensor):
    # read BN elements; write B elements
    x_max = x.max(dim=1)[0] # (B, 1) NOTE: subtract by max to prevent overflows
    # read BN + B elements; write BN elements
    z = x - x_max # (B, N)
    # read BN elements; write BN elements
    numerator = torch.exp(z) # (B, N)
    # read BN elements, write B elements
    denominator = torch.sum(numerator, dim=1) # (B, 1)
    # read BN + B elements, write BN elements
    ret = numerator / denominator # (B, N)
    # TOTAL: read 5BN + 2B elements; wrote 3BN + 2B elements - lots of wasted reads and writes
    return ret


def fused_softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # make sure ALL elements in each row are covered in a block
    
    # NOTE: a warp is group of 32 GPU threads executing same instruction simultaneously
    # BLOCK_SIZE and num_warps can be autotuned for maximal performance
    num_warps = 4
    if BLOCK_SIZE > 4096:
        num_warps = 16
    elif BLOCK_SIZE > 2047:
        num_warps = 8
    
    out = torch.empty_like(x)
    grid = (n_rows,)
    fused_softmax_kernel[grid](x, out, x.stride(dim=0), out.stride(0), n_rows, n_cols, BLOCK_SIZE)
    return out

x = torch.tensor([[1., 2., 3., 4.]]).to(DEVICE)
print(torch.softmax(x, dim=1))
print(fused_softmax(x))
    
