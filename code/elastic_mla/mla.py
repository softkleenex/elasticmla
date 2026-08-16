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


    def forward_cached(self, x, cache=None, rank_mask=None):
        """Incremental MLA attention with a compressed latent cache.

        This inference-only path stores, per past token, only the shared KV latent
        ``c_kv`` and the shared decoupled-RoPE key.  Reconstructed per-head content
        keys/values are deliberately *not* cached.  They are recomputed from the
        latent cache on each call, trading compute for the MLA memory saving.

        Args:
            x: New hidden states of shape ``(B, T_new, d_model)``.  ``T_new`` may
                be one (decode) or larger (prefill/chunked prefill).
            cache: ``None`` or a dict returned by a previous call, containing
                ``c_kv: (B, T_past, d_c)`` and
                ``k_rope: (B, 1, T_past, d_rope)``.  ``k_rope`` is stored after
                applying RoPE at its absolute position.
            rank_mask: Optional mask broadcastable to ``(B, T_new, d_c)``.  It
                applies only to newly appended tokens; cached tokens retain the
                representation with which they were originally stored.

        Returns:
            ``(output, new_cache)`` where output has shape ``(B, T_new, d_model)``.

        Note:
            The returned cache tensors are detached because this API is intended
            for autoregressive inference, not training through cached history.
        """
        B, T_new, _ = x.shape
        device, dtype = x.device, x.dtype

        if cache is None:
            past_len = 0
            past_c_kv = None
            past_k_rope = None
        else:
            if set(cache) != {"c_kv", "k_rope"}:
                raise ValueError("cache must contain exactly {'c_kv', 'k_rope'}")
            past_c_kv = cache["c_kv"]
            past_k_rope = cache["k_rope"]
            if past_c_kv.ndim != 3 or past_k_rope.ndim != 4:
                raise ValueError("invalid MLA cache tensor rank")
            if past_c_kv.shape[0] != B or past_k_rope.shape[0] != B:
                raise ValueError("cache batch size does not match x")
            if past_c_kv.shape[1] != past_k_rope.shape[2]:
                raise ValueError("c_kv and k_rope cache lengths differ")
            if (
                past_c_kv.shape[2] != self.d_c
                or past_k_rope.shape[1] != 1
                or past_k_rope.shape[-1] != self.d_rope
            ):
                raise ValueError("cache feature dimensions do not match this MLA layer")
            if past_c_kv.device != device or past_k_rope.device != device:
                raise ValueError("cache and x must be on the same device")
            # c_kv and k_rope need not share a dtype under autocast: RoPE
            # arithmetic may promote k_rope while linear projections remain fp16/bf16.
            # Each tensor is checked against its corresponding *new* projection below.
            past_len = past_c_kv.shape[1]

        c_kv_new = self.W_DKV(x)
        if rank_mask is not None:
            try:
                c_kv_new = c_kv_new * rank_mask
            except RuntimeError as exc:
                raise ValueError(
                    f"rank_mask shape {tuple(rank_mask.shape)} is not broadcastable "
                    f"to new latent shape {tuple(c_kv_new.shape)}"
                ) from exc

        k_rope_new = self.W_KR(x).view(B, T_new, 1, self.d_rope).transpose(1, 2)
        if past_len and past_c_kv.dtype != c_kv_new.dtype:
            raise ValueError(
                "c_kv cache dtype is incompatible with the current projection output: "
                f"cache={past_c_kv.dtype}, new={c_kv_new.dtype}"
            )

        c_q = self.W_DQ(x)
        q_content = self.W_UQ(c_q).view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)
        q_rope = self.W_QR(c_q).view(B, T_new, self.n_heads, self.d_rope).transpose(1, 2)

        positions = torch.arange(past_len, past_len + T_new, device=device)
        cos, sin = self.rope(positions, device, dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope_new = apply_rope(k_rope_new, cos, sin)
        if past_len and past_k_rope.dtype != k_rope_new.dtype:
            raise ValueError(
                "k_rope cache dtype is incompatible with the current rotated key: "
                f"cache={past_k_rope.dtype}, new={k_rope_new.dtype}"
            )

        if past_len:
            c_kv_all = torch.cat((past_c_kv, c_kv_new), dim=1)
            k_rope_all = torch.cat((past_k_rope, k_rope_new), dim=2)
        else:
            c_kv_all = c_kv_new
            k_rope_all = k_rope_new

        total_len = past_len + T_new
        k_content = self.W_UK(c_kv_all).view(
            B, total_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        v = self.W_UV(c_kv_all).view(
            B, total_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        k_rope_heads = k_rope_all.expand(-1, self.n_heads, -1, -1)

        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat((k_content, k_rope_heads), dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Query j in the new chunk may attend through absolute key position
        # past_len + j, but not to later keys in that same chunk.
        q_abs = past_len + torch.arange(T_new, device=device)[:, None]
        k_abs = torch.arange(total_len, device=device)[None, :]
        causal = k_abs > q_abs
        attn_scores = attn_scores.masked_fill(causal[None, None, :, :], float("-inf"))

        attn = self.dropout(F.softmax(attn_scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.n_heads * self.d_head)
        out = self.W_O(out)

        new_cache = {
            "c_kv": c_kv_all.detach(),
            "k_rope": k_rope_all.detach(),
        }
        return out, new_cache

    def cache_num_bytes(self, cache):
        """Return the actual allocated bytes of a cache produced by forward_cached."""
        if cache is None:
            return 0
        return sum(t.numel() * t.element_size() for t in cache.values())
