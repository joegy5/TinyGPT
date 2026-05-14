import torch
from torch import nn
import triton
import triton.language as tl
from math import ceil
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, l_ptr, out_ptr,
    n_patches: tl.constexpr,
    q_stride: tl.constexpr, 
    k_stride: tl.constexpr, 
    v_stride: tl.constexpr,
    l_stride: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_SIZE_Q_ROW: tl.constexpr,
    BLOCK_SIZE_K_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # get memory addresses of elements in current Q block of size BLOCK_SIZE_Q_ROW x d_k
    # NOTE: this only works if n_patches is a multiple (like 256) of BLOCK_SIZE_Q_ROW (likely 64)
    #   - Note that to fix this in the future, you also have to change the "other" value of the masks when loading the Q/K/V blocks
    q_block_start_offset = pid * BLOCK_SIZE_Q_ROW * q_stride
    q_block_start = q_ptr + q_block_start_offset
    q_block_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW * q_stride)
    q_block_ptrs = q_block_start + q_block_offsets

    # load elements from current Q block
    q_head_amnt = n_patches * q_stride
    mask = q_block_start_offset + q_block_offsets <= n_patches * q_stride * tl.floor((1. * q_block_start_offset) / q_head_amnt).to(tl.int32) + q_head_amnt
    q_block = tl.load(q_block_ptrs, mask, other=0.0)
    q_block = tl.reshape(q_block, (BLOCK_SIZE_Q_ROW, q_stride))

    # initialize o, l, and m blocks
    prev_out_block = tl.full((BLOCK_SIZE_K_ROW, q_stride), value=0., dtype=tl.float32)
    prev_denom_block = tl.full((BLOCK_SIZE_Q_ROW, 1), value=0., dtype=tl.float32)
    prev_max_block = tl.full((BLOCK_SIZE_Q_ROW, 1), value=-float('inf'), dtype=tl.float32)

    # loop over the K blocks that will be multiplied to this Q block
    for j in range(0, tl.ceil(n_patches / BLOCK_SIZE_K_ROW).to(tl.int32)):
        # get memory addresses of elements in current K block of size BLOCK_SIZE_K_ROW x d_k
        k_block_sz = BLOCK_SIZE_K_ROW * k_stride
        k_block_start_offset = n_patches * k_stride * tl.floor(1. * pid * BLOCK_SIZE_K_ROW / n_patches).to(tl.int32) + j * k_block_sz
        k_block_start = k_ptr + k_block_start_offset
        k_block_offsets = tl.arange(0, BLOCK_SIZE_K_ROW * k_stride)
        k_block_ptrs = k_block_start + k_block_offsets

        # load elements from current K block
        k_head_amnt = n_patches * k_stride
        mask = k_block_start_offset + k_block_offsets <= n_patches * k_stride * tl.floor((1. * k_block_start_offset) / k_head_amnt) + k_head_amnt
        k_block = tl.load(k_block_ptrs, mask, other=0.0)

        # get memory addresses of elements in current V block of size BLOCK_SIZE_V_ROW x d_k
        v_block_sz = BLOCK_SIZE_K_ROW * v_stride
        v_block_start_offset = n_patches * k_stride * tl.floor(1. * pid * BLOCK_SIZE_K_ROW / n_patches).to(tl.int32) + j * v_block_sz
        v_block_start = v_ptr + v_block_start_offset
        v_block_offsets = tl.arange(0, BLOCK_SIZE_K_ROW * v_stride)
        v_block_ptrs = v_block_start + v_block_offsets

        # load elements from current V block
        v_head_amnt = n_patches * v_stride
        mask = v_block_start_offset + v_block_offsets <= n_patches * v_stride * tl.floor((1. * v_block_start_offset) / v_head_amnt) + v_head_amnt
        v_block = tl.load(v_block_ptrs, mask, other=0.0)

        # operate on loaded data
        k_block = tl.reshape(k_block, (BLOCK_SIZE_K_ROW, k_stride))
        v_block = tl.reshape(v_block, (BLOCK_SIZE_K_ROW, v_stride))

        s_block = tl.dot(q_block, tl.trans(k_block, (1, 0)), input_precision="ieee") # BLOCK_SIZE_Q_ROW x BLOCK_SIZE_K_ROW
        s_block = s_block / tl.sqrt(q_stride-0.0)
        curr_max_block = tl.max(s_block, axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1
        max_block = tl.max(tl.cat(prev_max_block, curr_max_block, dim=1), axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1

        p_block = tl.exp(s_block - curr_max_block) # BLOCK_SIZE_Q_ROW x BLOCK_SIZE_K_ROW

        prev_max_diff, curr_max_diff = prev_max_block - max_block, curr_max_block - max_block # BLOCK_SIZE_Q_ROW x 1

        curr_denom_block = tl.sum(p_block, axis=1, keep_dims=True) # BLOCK_SIZE_Q_ROW x 1
        prev_denom_block = tl.exp(prev_max_diff) * prev_denom_block + tl.exp(curr_max_diff) * curr_denom_block

        prev_out_block = tl.exp(prev_max_diff) * prev_out_block + tl.exp(curr_max_diff) * tl.dot(p_block, v_block, input_precision="ieee")
        prev_max_block = max_block
    
    # get memory addresses of elements in curr L block
    l_block_start_offset = pid * BLOCK_SIZE_Q_ROW * l_stride
    l_block_start = l_ptr + l_block_start_offset
    l_block_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW)
    l_block_ptrs = l_block_start + l_block_offsets
    l_block = tl.reshape(prev_max_block + tl.log(prev_denom_block-0.0), (BLOCK_SIZE_Q_ROW,))
    
    # write elements to current L block
    mask = (l_block_start_offset + l_block_offsets) <= n_patches * (tl.floor(1. * l_block_start_offset / n_patches) + n_patches * l_stride)
    tl.store(l_block_ptrs, l_block, mask)

    # get memory addresses for current O block
    out_block_start_offset = pid * BLOCK_SIZE_Q_ROW * out_stride
    out_block_start = out_ptr + out_block_start_offset
    out_block_offsets = tl.arange(0, BLOCK_SIZE_Q_ROW * out_stride)
    out_block_ptrs = out_block_start + out_block_offsets

    # write elements to current O block
    out_head_amnt = n_patches * out_stride
    mask = out_block_start_offset + out_block_offsets <= out_head_amnt * tl.floor((1. * out_block_start_offset) / out_head_amnt) + out_head_amnt
    out_block = tl.reshape(prev_out_block / prev_denom_block, (BLOCK_SIZE_Q_ROW * out_stride,))

    tl.store(out_block_ptrs, out_block, mask)


def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW):
    b_sz, n_heads, n_patches, _ = Q.shape # same as K.shape and V.shape
    assert n_patches % BLOCK_SIZE_K_ROW == 0 and n_patches % BLOCK_SIZE_Q_ROW == 0, "num patches must be multiple of Q and K block sizes"

    grid_sz = b_sz * n_heads * ceil(1. * n_patches / BLOCK_SIZE_Q_ROW)
    grid = (grid_sz,)

    L = torch.empty((b_sz, n_heads, n_patches, 1)).to(DEVICE) # logsumexp used for backward pass
    out = torch.empty_like(Q).to(DEVICE)

    assert Q.stride(2) == K.stride(2) and Q.stride(2) == V.stride(2), "Q, K, V must all have same latent dimension"

    flash_attn_kernel[grid](
        Q, K, V, L, out, n_patches,  
        Q.stride(2), K.stride(2), V.stride(2), L.stride(2), out.stride(2), 
        BLOCK_SIZE_Q_ROW, BLOCK_SIZE_K_ROW
    )
    return out

import math
import torch.nn.functional as F
def attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-head attention without causal masking or output projection (W_O).
 
    Args:
        Q: Query tensor of shape (B, H, N, D)
        K: Key tensor of shape (B, H, N, D)
        V: Value tensor of shape (B, H, N, D)
 
    Returns:
        Output tensor of shape (B, H, N, D)
        out = softmax(Q @ K^T / sqrt(D)) @ V
    """
    D = Q.shape[-1]
 
    # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
    attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
 
    attn_weights = F.softmax(attn_scores, dim=-1)
 
    # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
    out = attn_weights @ V
 
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=10, help="batch dimension of the test input")
    parser.add_argument("--h_dim", type=int, default=10, help="head dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=128, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=128, help="embedding dimension of the test input")
    parser.add_argument("--denom_const", type=int, default=10000, help="denominator constant used in the Sinusoidal PE calculations")
    args = parser.parse_args()

    for test_no in range(args.n_tests):
        # Sample from Standard Normal Distribution
        q_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        k_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        v_in = torch.randn(size=(args.b_dim, args.h_dim, args.n_dim, args.d_dim), dtype=torch.float32).to(DEVICE)
        
        out = flash_attn(q_in, k_in, v_in, 16, 16)
        out_ref = attn(q_in, k_in, v_in)

        # print(f"TEST #{test_no} ------------")
        # print(f"OUT:\n{out}") 





