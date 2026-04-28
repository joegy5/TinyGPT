import torch
import triton
import triton.language as tl
from math import ceil

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def matmul_naive(X: torch.Tensor, Y: torch.Tensor):
    M, K = X.shape
    K2, N = Y.shape
    assert K == K2

    Z = torch.tensor([[0.0 for _ in range(N)] for _ in range(M)])

    # Each element in Y loaded M times -> highly wasteful, should just load it once
    # This constant memory access (reading from HBM) dominates the cheaper FLOPs
    for i in range(M):
        for j in range(N):
            acc = 0.0 
            for k in range(K):
                acc += X[i, k] * Y[k, j]
            Z[i, j] = acc
    
    return Z


def tiled_matmul_naive(X: torch.Tensor, Y: torch.Tensor, BLOCK_SIZE_M: int, BLOCK_SIZE_N: int, BLOCK_SIZE_K: int):
    M, K = X.shape
    K2, N = Y.shape
    assert K == K2

    Z = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    # computing as blocks instead of as individual rows and columns reduces the number of reads from DRAM, 
    # since we can reuse columns from Y (and rows from X)
    for m in range(0, M, BLOCK_SIZE_M):
        for n in range(0, N, BLOCK_SIZE_N):
            acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), device=DEVICE, dtype=torch.float16)
            for k in range(0, K, BLOCK_SIZE_K):
                X_block = X[m : m + BLOCK_SIZE_M, k : k + BLOCK_SIZE_K]
                Y_block = Y[k : k + BLOCK_SIZE_K, n : n + BLOCK_SIZE_N]
                acc += torch.matmul(X_block, Y_block)
            Z[m : m + BLOCK_SIZE_M, n : n + BLOCK_SIZE_N] = acc
    
    return Z


@triton.jit
def matmul_kernel(
    X_ptr, Y_ptr, Z_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    X_row_stride, X_col_stride,
    Y_row_stride, Y_col_stride,
    Z_row_stride, Z_col_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # num blocks along row axis that we need
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # num blocks along col axis that we need
    
    # goal: map pid -> (pid_m, pid_n) -> each Z block has a unique id -> each program instances handles a single block
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # add % operator so that addresses beyond M will just be wrapped around to beginning block values
    # this doesn't matter anyways because at the very end, the masks involved with writing final output to Z block filter out these values anyways
    X_row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M # [pm*BM, pm*BM+1, pm*BM+2, ..., pm*BM+BM-1]
    Y_col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N# [pn*BN, pn*BN+1, pn*BN+2, ..., pn*BN+BN-1]
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # X_row_offsets[:, None] equivalent to X_row_offsets.unsqueeze(-1); it turns each element into a singleton row vector
    # Multiplying by X_row_stride advances to next row
    # offsets_k[None, :] is equivalent to offsets_k.unsqueeze(0); offsets_k becomes a row vector
    # Multiplying by X_col_stride makes sure to actually advance to the next column value within a row
    # Result of adding row vector of shape (BLOCK_SIZE_M, 1) to column vector of shape (1, BLOCK_SIZE_K) is 2D matrix of shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # (achieved via broadcasting)
    # This 2D matrix contains the memory addresses of all the elements in the current block of X
    X_ptrs = X_ptr + (X_row_offsets[:, None] * X_row_stride + offsets_k[None, :] * X_col_stride)
    # Result of adding column vector of shape (1, BLOCK_SIZE_K) to row vector of shape (BLOCK_SIZE_N, 1) is 2D matrix fo shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    # This 2D matrix contains the memory addresses of all the elements in the current block of Y
    Y_ptrs = Y_ptr + (offsets_k[:, None] * Y_row_stride + Y_col_offsets[None, :] * Y_col_stride)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # whole K dimension rarely fits on-chip, which is why we use portions of K (BLOCK_SIZE_K), explaining 
    # why we don't use entire K in the block-based matrix multiplication algorithm
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # mask is boolean 1D tensor that is broadcasted to match X_ptrs/Y_ptrs shape 
        # < K - k * BLOCK_SIZE_K ensures that as k increases, initial values 0 to BLOCK_SIZE_K - 1 will become false if out of bounds of K dimension
        #   this is cleaner than having to manually determine the actual K dimension end and check where X_ptrs/Y_ptrs itself goes out of bounds
        X_block = tl.load(X_ptrs, mask=offsets_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        Y_block = tl.load(Y_ptrs, mask=offsets_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # perform matrix multiplication (with accumulation) between X_block and Y_block
        acc = tl.dot(X_block, Y_block, acc)

        # move X and Y pointers to next block for next iteration along K dimension
        X_ptrs += BLOCK_SIZE_K * X_col_stride
        Y_ptrs += BLOCK_SIZE_K * Y_row_stride
    
    Z = acc.to(tl.float16)

    # write Z block back to output matrix Z
    Z_row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    Z_col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Z_ptrs = Z_ptr + (Z_row_offsets[:, None] * Z_row_stride + Z_col_offsets[None, :] * Z_col_stride)
    # each 1D mask broadcasted to shape (BLOCK_SIZE_M, BLOCK_SIZE_N); & operator only True wherever both corresponding elements in same position in each broadcasted
    #   array are True
    tl.store(Z_ptrs, value=Z, mask=(Z_row_offsets[:, None] < M & Z_col_offsets[None, :] < N)) 



def matmul(X: torch.Tensor, Y: torch.Tensor):
    M, K = X.shape
    K2, N = Y.shape
    assert K == K2

    # should be tuned with triton's autotuner
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4

    # create output matrix
    Z = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    # determine number of kernel program instances to launch - each instance handles a specific block
    # need enough instances to cover all blocks of the two matrices -> use ceil() with mask
    grid = (ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N),)

    # launch the kernel
    matmul_kernel[grid](
        X, Y, Z, 
        M, N, K,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        Z.stride(0), Z.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return Z

