import torch
from torch import nn
import triton
import triton.language as tl
import math
from math import ceil
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def flash_attn_2_fwd_kernel(
    q_ptr, k_ptr, v_ptr, l_ptr, out_ptr,
    n_patches: tl.constexpr,
    in_stride: tl.constexpr,
    BLOCK_SIZE_Q_ROW: tl.constexpr,
    BLOCK_SIZE_K_ROW: tl.constexpr,
    q_block_sz: tl.constexpr,
    k_block_sz: tl.constexpr,
    norm: tl.constexpr
):
    pid = tl.program_id(0)
    head_amnt = n_patches * in_stride

    # get memory addresses of elements in current Q block of size BLOCK_SIZE_Q_ROW x d_k
    # NOTE: this only works if n_patches is a multiple (like 256) of BLOCK_SIZE_Q_ROW (likely 64)
    #   - Note that to fix this in the future, you also have to change the "other" value of the masks when loading the Q/K/V blocks
    q_block_start_offset = pid * q_block_sz
    q_block_start = q_ptr + q_block_start_offset
    q_row_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW)
    q_col_offsets = tl.arange(0, in_stride)
    q_block_offsets = q_row_offsets[:, None] * in_stride + q_col_offsets[None, :]
    q_block_ptrs = q_block_start + q_block_offsets

    # load elements from current Q block
    q_block = tl.load(q_block_ptrs)#, mask, other=0.0)

    # initialize o, l, and m blocks
    prev_out_block = tl.full((BLOCK_SIZE_Q_ROW, in_stride), value=0., dtype=tl.float32)
    prev_denom_block = tl.full((BLOCK_SIZE_Q_ROW, 1), value=0., dtype=tl.float32)
    prev_max_block = tl.full((BLOCK_SIZE_Q_ROW, 1), value=-float('inf'), dtype=tl.float32)

    # loop over the K blocks that will be multiplied to this Q block
    k_row_offsets = tl.arange(0, BLOCK_SIZE_K_ROW)
    k_block_offsets = k_row_offsets[:, None] * in_stride + q_col_offsets[None, :]
    k_block_start_offset_initial = head_amnt * (pid * BLOCK_SIZE_K_ROW // n_patches)

    for j in range(0, tl.ceil(n_patches / BLOCK_SIZE_K_ROW).to(tl.int32)):
        # get memory addresses of elements in current K block of size BLOCK_SIZE_K_ROW x d_k
        k_block_start_offset = k_block_start_offset_initial + j * k_block_sz
        k_block_start = k_ptr + k_block_start_offset
        k_block_ptrs = k_block_start + k_block_offsets

        # load elements from current K block
        k_block = tl.load(k_block_ptrs)

        # get memory addresses of elements in current V block of size BLOCK_SIZE_K_ROW x d_k
        v_block_start = v_ptr + k_block_start_offset
        v_block_ptrs = v_block_start + k_block_offsets

        # load elements from current V block
        v_block = tl.load(v_block_ptrs)

        # operate on loaded data
        s_block = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") # BLOCK_SIZE_Q_ROW x BLOCK_SIZE_K_ROW
        s_block = s_block / norm
        
        curr_max_block = tl.max(s_block, axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1
        max_block = tl.max(tl.cat(prev_max_block, curr_max_block, dim=1), axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1

        p_block = tl.exp(s_block - curr_max_block) # BLOCK_SIZE_Q_ROW x BLOCK_SIZE_K_ROW

        prev_max_diff, curr_max_diff = prev_max_block - max_block, curr_max_block - max_block # BLOCK_SIZE_Q_ROW x 1

        curr_denom_block = tl.sum(p_block, axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1
        new_denom_block = tl.exp(prev_max_diff) * prev_denom_block + tl.exp(curr_max_diff) * curr_denom_block

        prev_out_block = tl.exp(prev_max_diff) * prev_out_block + tl.exp(curr_max_diff) * tl.dot(p_block, v_block, input_precision="ieee")
        prev_max_block = max_block
        prev_denom_block = new_denom_block

    # get memory addresses of elements in curr L block
    l_block_start_offset = pid * BLOCK_SIZE_Q_ROW
    l_block_start = l_ptr + l_block_start_offset
    l_block_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW)
    l_block_ptrs = l_block_start + l_block_offsets
    l_block = tl.reshape(prev_max_block + tl.log(prev_denom_block), (BLOCK_SIZE_Q_ROW,))
    
    # write elements to current L block
    tl.store(l_block_ptrs, l_block)

    # get memory addresses for current O block
    out_block_start_offset = pid * q_block_sz
    out_block_start = out_ptr + out_block_start_offset
    out_block_ptrs = out_block_start + q_block_offsets

    # write elements to current O block
    tl.store(out_block_ptrs, prev_out_block / prev_denom_block)


def flash_attn_2_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW):
    b_sz, n_heads, n_patches, in_stride = Q.shape # same as K.shape and V.shape
    assert n_patches % BLOCK_SIZE_K_ROW == 0 and n_patches % BLOCK_SIZE_Q_ROW == 0, "num patches must be multiple of Q and K block sizes"

    grid_sz = b_sz * n_heads * ceil(1. * n_patches / BLOCK_SIZE_Q_ROW)
    grid = (grid_sz,)

    L = torch.empty((b_sz, n_heads, n_patches, 1)).to(DEVICE) # logsumexp used for backward pass
    out = torch.empty_like(Q).to(DEVICE)

    assert Q.stride(2) == K.stride(2) and Q.stride(2) == V.stride(2), "Q, K, V must all have same latent dimension"

    flash_attn_2_fwd_kernel[grid](
        Q, K, V, L, out, n_patches, in_stride,
        BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW,
        BLOCK_SIZE_Q_ROW * in_stride, BLOCK_SIZE_K_ROW * in_stride,
        math.sqrt(in_stride)
    )
    return out, L


@triton.jit
def flash_attn_2_bwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, d_ptr, L_ptr,
    dq_ptr, dk_ptr, dv_ptr, do_ptr,
    BLOCK_SIZE_Q_ROW: tl.constexpr,
    BLOCK_SIZE_K_ROW: tl.constexpr,
    n_patches: tl.constexpr,
    in_stride: tl.constexpr,
    d_k: tl.constexpr
):
    pid = tl.program_id(0)

    # load K and V blocks, both of size (BLOCK_SIZE_K_ROW x in_stride)
    k_block_start = k_ptr + pid * BLOCK_SIZE_K_ROW * in_stride
    v_block_start = v_ptr + pid * BLOCK_SIZE_K_ROW * in_stride

    k_block_row_offsets = tl.arange(0, BLOCK_SIZE_K_ROW)
    k_block_col_offsets = tl.arange(0, in_stride)
    k_block_offsets = k_block_row_offsets[:, None] * in_stride + k_block_col_offsets[None, :]
  
    k_block_ptrs = k_block_start + k_block_offsets
    v_block_ptrs = v_block_start + k_block_offsets

    k_block = tl.load(k_block_ptrs)
    v_block = tl.load(v_block_ptrs)

    k_block_transpose = tl.trans(k_block, (1, 0))
    v_block_transpose = tl.trans(v_block, (1, 0))

    # initialize dK and dV blocks as zeros
    dk_block = tl.zeros((BLOCK_SIZE_K_ROW, in_stride), dtype=tl.float32)
    dv_block = tl.zeros((BLOCK_SIZE_K_ROW, in_stride), dtype=tl.float32)

    for i in range(0, tl.ceil(1.0 * n_patches / BLOCK_SIZE_Q_ROW).to(tl.int32)):
        # load Q, O, dQ, dO, L, and D blocks
        block_start = (pid * BLOCK_SIZE_K_ROW // n_patches) * n_patches + i * BLOCK_SIZE_Q_ROW
        block_row_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW)
        block_col_offsets = tl.arange(0, in_stride)
        block_offsets = block_row_offsets[:, None] * in_stride + block_col_offsets[None, :] 

        q_block_start = q_ptr + block_start * in_stride
        o_block_start = o_ptr + block_start * in_stride
        dq_block_start = dq_ptr + block_start * in_stride
        do_block_start = do_ptr + block_start * in_stride
        L_block_start = L_ptr + block_start
        d_block_start = d_ptr + block_start

        dq_block_ptrs = dq_block_start + block_offsets
        
        q_block = tl.load(q_block_start + block_offsets)
        o_block = tl.load(o_block_start + block_offsets)
        dq_block = tl.load(dq_block_ptrs)
        do_block = tl.load(do_block_start + block_offsets)
        L_block = tl.load(L_block_start + block_row_offsets[:, None])
        d_block = tl.load(d_block_start + block_row_offsets[:, None])


        # perform computation steps
        s_block = tl.dot(q_block, k_block_transpose) / d_k # (BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW)
        p_block = tl.exp(s_block - L_block) # (BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW)
        
        dv_block = dv_block + tl.dot(tl.trans(p_block, (1, 0)), do_block) # (BLOCK_SIZE_K_ROW, in_stride)
        
        dp_block = tl.dot(do_block, v_block_transpose) # (BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW)
        ds_block = p_block * (dp_block - d_block) # (BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW)
        
        # tl.device_print("ds_block", ds_block)

        dq_block_add = tl.dot(ds_block, k_block) / d_k # (BLOCK_SIZE_Q_ROW, in_stride)
        dk_block = dk_block + tl.dot(tl.trans(ds_block, (1, 0)), q_block) # (BLOCK_SIZE_K_ROW, in_stride)

        # Write dq_block back to HBM (need atomic update)
        tl.atomic_add(dq_block_ptrs, dq_block_add)

    # Write dK and dV blocks back to HBM
    dk_block_start = dk_ptr + pid * BLOCK_SIZE_K_ROW * in_stride
    dv_block_start = dv_ptr + pid * BLOCK_SIZE_K_ROW * in_stride

    dk_block_ptrs = dk_block_start + k_block_offsets
    dv_block_ptrs = dv_block_start + k_block_offsets

    tl.store(dk_block_ptrs, dk_block / d_k)
    tl.store(dv_block_ptrs, dv_block)

def flash_attn_2_bwd(Q, K, V, O, dO, L, BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW):
    Q, K, V, O, dO, L = Q.to(DEVICE), K.to(DEVICE), V.to(DEVICE), O.to(DEVICE), dO.to(DEVICE), L.to(DEVICE)
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    
    b_sz, n_heads, n_patches, in_stride = Q.shape
    grid_sz = b_sz * n_heads * ceil(1. * n_patches / BLOCK_SIZE_K_ROW)
    grid = (grid_sz,)

    dQ, dK, dV = torch.zeros_like(Q).to(DEVICE), torch.zeros_like(K).to(DEVICE), torch.zeros_like(V).to(DEVICE)

    flash_attn_2_bwd_kernel[grid](
        Q, K, V, O, D, L,
        dQ, dK, dV, dO,
        BLOCK_SIZE_Q_ROW, 
        BLOCK_SIZE_K_ROW,
        n_patches, in_stride,
        math.sqrt(in_stride)
    )
    
    return dQ, dK, dV


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--h_dim", type=int, default=1, help="head dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=16, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=16, help="embedding dimension of the test input")

    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        q_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        k_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        v_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)

        print(f"TEST #{test_no} ------------")
        
        out, lse = flash_attn_2_fwd(q_in, k_in, v_in, 16, 16)
        dout = torch.randn_like(out).to(DEVICE)
        dq, dk, dv = flash_attn_2_bwd(q_in, k_in, v_in, out, dout, lse, 16, 16)

        print(f"OUT:\n{out}")
        print(f"LSE:\n{lse}")
        print(f"dQ:\n{dq}")
        print(f"dK:\n{dk}")
        print(f"dV:\n{dv}")








