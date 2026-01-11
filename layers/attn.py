import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    '''
    Original attention mechanism in the "Attention is All You Need" paper
    '''
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

    def forward(self, X, enc_out=None, use_kv_cache=False):
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
            else:
                new_k_vector = self.expand_attn_tensor(self.W_K(X))
                new_v_vector = self.expand_attn_tensor(self.W_V(X))
                if self.K_cache is None:
                    self.K_cache = new_k_vector # (B, H, 1, d_k)
                    self.V_cache = new_v_vector # (B, H, 1, d_k)
                else:
                    self.K_cache = torch.cat((self.K_cache, new_k_vector), dim=3) # (B, H, t, d_k), where t is the current number of tokens (input + generated)
                    self.V_cache = torch.cat((self.V_cache, new_v_vector), dim=3) # (B, H, t, d_k)
            K = self.K_cache
            V = self.V_cache

        raw_attn = (Q @ K.transpose(-1, -2)) / torch.sqrt(self.d_k) # (B, H, N, N)
        masked_softmax_attn = self.softmax(self.apply_mask(raw_attn)) # (B, H, N, N)
        X = masked_softmax_attn @ V # (B, H, N, d_k)
        X = X.permute(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.num_heads * self.d_k) # (B, N, d_model)
        return self.W_O(X) # (B, N, d_model)


class GroupedQueryAttention(nn.Module):
    '''
    Each head has a separte Q matrix, but every (num_heads / k) heads share the same K and V matrices
    '''
    def __init__(self, batch_size, seq_len, d_model, num_heads, k):
        super(GroupedQueryAttention, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_k = d_model / num_heads
        self.num_heads = num_heads
        self.k = k
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, k * self.d_k)
        self.W_V = nn.Linear(d_model, k * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)
        self.K_cache = None
        self.V_cache = None
        self.softmax = nn.Softmax(dim=-1)

    def expand_attn_tensor(self, attn_tensor, is_Q=False):
        heads = self.num_heads if is_Q else self.k
        attn_tensor = attn_tensor.reshape(self.batch_size, attn_tensor.shape[1], heads, self.d_k).permute(0, 2, 1, 3)
        if is_Q: attn_tensor = attn_tensor.reshape(self.batch_size, self.k, self.num_heads / self.k, attn_tensor.shape[2], self.d_k)
        else: attn_tensor.unsqueeze(2)

        return attn_tensor

    def forward(self, X, enc_out=None, use_kv_cache=False):
        if not use_kv_cache: # training
            Q = self.expand_attn_tensor(self.W_Q(X), True) # (B, k, g, N, d_k), where g = H / k is the size of each "group"
            K = self.expand_attn_tensor(self.W_K(X)) # (B, k, 1, N, d_k)
            V = self.expand_attn_tensor(self.W_V(X)) # (B, k, 1, N, d_k)
        else: # inference
            # X.shape() - (B, 1, d_model)
            Q = self.expand_attn_tensor(self.W_Q(X), True) # (B, k, g, 1, d_k)
            if enc_out is not None: # cross-attention layer in enc-dec architecture - just compute K & V once since input doesn't change during each token's generation
                if self.K_cache is None:
                    # enc_out - (B, N, d_model)
                    self.K_cache = self.expand_attn_tensor(self.W_K(enc_out)) # (B, k, 1, N, d_k)
                    self.V_cache = self.expand_attn_tensor(self.W_V(enc_out)) # (B, k, 1, N, d_k)
            else:
                new_k_vector = self.expand_attn_tensor(self.W_K(X)) # (B, 1, k * d_k) -> (B, k, 1, 1, d_k)            
                new_v_vector = self.expand_attn_tensor(self.W_V(X)) # (B, 1, k * d_k) -> (B, k, 1, 1, d_k)
                if self.K_cache is None:
                    self.K_cache = new_k_vector # (B, k, 1, 1, d_k)
                    self.V_cache = new_v_vector # (B, k, 1, 1, d_k)
                else:
                    self.K_cache = torch.cat((self.K_cache, new_k_vector), dim=3) # (B, k, 1, t, d_k)
                    self.V_cache = torch.cat((self.V_cache, new_v_vector), dim=3) # (B, k, 1, t, d_k)
            K = self.K_cache # (B, k, 1, t, d_k)
            V = self.V_cache # (B, k, 1, t, d_k)
            pass
        
        # K and V will be broadcasted along group dimension to work with Q.
        # This is more efficient than scaling up K and V yourself, because broadcasting under the hood is actually performing 
        #   parallel operations repeatedly, not increasing the matrix size and thus consuming more memory
        raw_attn = (Q @ K.transpose(-1, -2)) / torch.sqrt(self.d_k) # (B, k, g, N, N) if not using kv cache, (B, k, g, 1, t) otherwise
        masked_softmax_attn = self.softmax(self.apply_mask(raw_attn)) # (B, k, g, N, N) if not using kv cache, (B, k, g, 1, t) otherwise
        X = masked_softmax_attn @ V # (B, k, g, N, d_k) if not using kv cache, (B, k, g, 1, d_k) otherwise
        X = X.reshape(self.batch_size, self.num_heads, X.shape[-2], self.d_k) # (B, H, N, d_k) if not using kv cache, (B, H, 1, d_k) otherwise
        X = X.permute(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.num_heads * self.d_k) # (B, N, d_model) if not using kv cache, (B, 1, d_model) otherwise
        return self.W_O(X) # (B, N, d_model)          


class MultiQueryAttention(GroupedQueryAttention):
    '''
    Each head has a separate Q matrix, but all heads share the same K and V matrices
    MQA can be thought of as a special case of GQA
    '''
    def __init__(self, batch_size, seq_len, d_model, num_heads):
        super(MultiQueryAttention, self).__init__(batch_size, seq_len, d_model, num_heads, k=1)

    def forward(self, X: Tensor) -> Tensor:
        return super().forward(X)


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


