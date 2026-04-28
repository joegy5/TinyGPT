import torch
import triton
import triton.language as tl
from math import ceil

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def add_kernel(
    x_ptr, # despite entire tensor being passed in, only memory address of first element x[0] actually becomes x_ptr
    y_ptr, # same for y_ptr
    out_ptr,
    n_elem,
    BLOCK_SIZE: tl.constexpr        
):
    # goal: get all the elements within the specific block, and store them back in the correct memory addresses of output
    
    # step 1: get all the data addresses of the current block from x and y, as tensors
    pid = tl.program_id(axis=0) # id of current program executing this kernel; axis=0 means we take id along the first (and only) axis
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # NOTE: can't do tl.arange(block_start, block_start + BLOCK_SIZE) - will cause error since block_start not of type tl.constexpr (required by tl.arange)
    mask = offsets < n_elem # prevent out of bounds errors in case n_elem not exactly divisible by BLOCK_SIZE and causing remainder elements left over in the final block
    
    # step 2: load all the data corresponding to the current block
    x = tl.load(x_ptr + offsets, mask=mask) # x_ptr + offsets becomes tensor of memory addr -> tl.load() loads tensor of corresponding elements
    y = tl.load(y_ptr + offsets, mask=mask)

    # step 3: add the correspoding elements from each tensor
    out = x + y
    tl.device_print("pid", pid, x)


    # step 4: load output tensor back into memory address of original out variable
    tl.store(out_ptr + offsets, out, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    n_elem = x.numel()
    BLOCK_SIZE = 2 # number of elements to process per block
    
    grid_sz = ceil(n_elem / BLOCK_SIZE) # number of blocks to work on
    grid = (grid_sz,)

    add_kernel[grid](x, y, out, n_elem, BLOCK_SIZE=BLOCK_SIZE) # process each block in the grid
    return out


# Works for arrays of any shape; grid is a tuple representing GPU dimensions, NOT tensor dimensions; multi-dim tensors are stored as continous arrays in memory
x = torch.tensor([[1., 2., 3., 4.]]).to(DEVICE)
y = torch.tensor([[2., 3., 4., 5.]]).to(DEVICE)
print(vector_add(x, y))

x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).to(DEVICE)
y = torch.tensor([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]]).to(DEVICE)
print(vector_add(x, y))

x = torch.tensor([[[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]]]).to(DEVICE)
y = torch.tensor([[[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]]]).to(DEVICE)
print(vector_add(x, y))