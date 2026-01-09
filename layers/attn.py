import torch
from torch import nn, Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size, seq_len, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_model / num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.K_cache = None
        self.V_cache = None
        self.softmax = nn.Softmax(dim=-1)

    def apply_mask(self, raw_attn: Tensor, padding_mask: Tensor, use_causal_mask: bool = False):
        # padding_mask applied along second dimension (N)
        # padding_mask originally has shape (B, N) where each value is either 0 (pad token) or 1 (actual token)
        mask = 1. - padding_mask.unsqueeze(-1).expand(-1, -1, self.seq_len).unsqueeze(1) # (B, 1, N, N)
        mask[mask == 1.] = torch.inf
        if use_causal_mask:
            # upper triangular mask, diagonal should NOT be retained (tokens can attend to themselves)
            causal_mask = torch.triu(torch.full((self.batch_size, 1, self.seq_len, self.seq_len), torch.inf), diagonal=1)
            mask += causal_mask # torch.inf + torch.inf = torch.inf
        return raw_attn + mask
    
    def expand_attn_tensor(self, attn_tensor):
        return attn_tensor.reshape(self.batch_size, self.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def forward(self, X: Tensor, enc_out: Tensor = None, use_kv_cache: bool = False) -> Tensor:
        if not use_kv_cache: # training
            kv_in = enc_out if enc_out is not None else X
            Q = self.expand_attn_tensor(self.W_Q(X)) # (B, H, N, d_k)
            K = self.expand_attn_tensor(self.W_K(kv_in)) # (B, H, N, d_k)
            V = self.expand_attn_tensor(self.W_V(kv_in)) # (B, H, N, d_k)
        else: # inference
            # X.shape: (B, 1, d_model), containing just the query vector for the newest token
            Q = self.expand_attn_tensor(self.W_Q(X)) # (B, H, 1, d_model)
            if enc_out is not None: # cross-attention layer in enc-dec architecture - just compute K & V once since input doesn't change during each token's generation
                if self.K_cache is None:
                    self.K_cache = self.expand_attn_tensor(self.W_K(enc_out))
                    self.V_cache = self.expand_attn_tensor(self.W_V(enc_out))
                K = self.K_cache
                V = self.V_cache
            else:
                new_k_vector = self.expand_attn_tensor(self.W_K(X))
                new_v_vector = self.expand_attn_tensor(self.W_V(X))
                if self.K_cache is None:
                    self.K_cache = new_k_vector # (B, H, 1, d_model)
                    self.V_cache = new_v_vector # (B, H, 1, d_model)
                else:
                    self.K_cache = torch.cat((self.K_cache, new_k_vector), dim=3) # (B, H, t, d_model), where t is the current number of tokens (input + generated)
                    self.V_cache = torch.cat((self.V_cache, new_v_vector), dim=3) # (B, H, t, d_model)
                
        raw_attn = (Q @ K.transpose(-1, -2)) / torch.sqrt(self.d_k) # (B, H, N, N)
        masked_softmax_attn = self.softmax(self.apply_mask(raw_attn)) # (B, H, N, N)
        X = masked_softmax_attn @ V # (B, H, N, d_k)
        X = X.permute(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.num_heads * self.d_k) # (B, N, d_model)
        return self.W_O(X) # (B, N, d_model)


class MultiQueryAttention(nn.Module):
    def __init__(self):
        super(MultiQueryAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super(GroupedQueryAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class SlidingWindowAttention(nn.Module):
    def __init__(self):
        super(SlidingWindowAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class SparseAttention(nn.Module):
    def __init__(self):
        super(SparseAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class GlobalAttention(nn.Module):
    def __init__(self):
        super(GlobalAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class LinearAttention(nn.Module):
    def __init__(self):
        super(LinearAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class MultiLatentAttention(nn.Module):
    def __init__(self):
        super(MultiLatentAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


class TensorProductAttention(nn.Module):
    def __init__(self):
        super(TensorProductAttention, self).__init__()

    def forward(self, X: Tensor) -> Tensor:
        pass


