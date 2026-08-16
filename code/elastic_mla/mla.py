"""
Minimal DeepSeek-V2 style Multi-head Latent Attention (MLA) implementation.

Reference: DeepSeek-V2 (arXiv:2405.04434).
- KV is compressed into a shared low-rank latent c_KV (dim d_c).
- Q is also low-rank compressed (dim d_c' ) for training memory efficiency (optional here).
- RoPE is "decoupled": a small extra per-head dimension carries positional info
  and is NOT compressed, so latent c_KV stays purely content-based.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D_rope)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t, device, dtype):
        freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MultiHeadLatentAttention(nn.Module):
    """
    d_model:   model hidden size
    n_heads:   number of attention heads
    d_head:    per-head dim for the "content" part (non-RoPE)
    d_rope:    per-head dim carrying RoPE (decoupled, shared K rope across heads)
    d_c:       KV latent compression dim   (this is the thing we want to study/vary)
    d_c_q:     Query latent compression dim (optional, can equal d_c)
    """

    def __init__(self, d_model, n_heads, d_head=64, d_rope=32, d_c=256, d_c_q=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rope = d_rope
        self.d_c = d_c
        d_c_q = d_c_q or d_c

        # --- KV latent compression / decompression ---
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)              # down-proj to latent
        self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False)      # up-proj -> K content
        self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False)      # up-proj -> V
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)            # decoupled rope key (shared across heads)

        # --- Query latent compression / decompression ---
        self.W_DQ = nn.Linear(d_model, d_c_q, bias=False)
        self.W_UQ = nn.Linear(d_c_q, n_heads * d_head, bias=False)
        self.W_QR = nn.Linear(d_c_q, n_heads * d_rope, bias=False)    # per-head rope query

        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.rope = RotaryEmbedding(d_rope)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

    def forward(self, x, attn_mask=None, return_latent=False, rank_mask=None):
        """
        x: (B, T, d_model)
        rank_mask: optional (d_c,) boolean/float mask to zero out latent dims
                   -> lets us simulate "using only rank r" at inference time.
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        c_kv = self.W_DKV(x)  # (B, T, d_c)  <-- THE latent we care about
        if rank_mask is not None:
            c_kv = c_kv * rank_mask

        k_content = self.W_UK(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        v = self.W_UV(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)          # (B,H,T,Dh)
        k_rope = self.W_KR(x).view(B, T, 1, self.d_rope).transpose(1, 2)                   # (B,1,T,Dr)

        c_q = self.W_DQ(x)
        q_content = self.W_UQ(c_q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q_rope = self.W_QR(c_q).view(B, T, self.n_heads, self.d_rope).transpose(1, 2)

        pos = torch.arange(T, device=device)
        cos, sin = self.rope(pos, device, dtype)  # (T, Dr)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        q = torch.cat([q_content, q_rope], dim=-1)  # (B,H,T,Dh+Dr)
        k = torch.cat([k_content, k_rope], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.W_O(out)

        if return_latent:
            return out, c_kv
        return out
