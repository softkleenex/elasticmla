"""
Small GPT-style LM using MultiHeadLatentAttention blocks.
Sized to run comfortably on Apple M4 Pro (MPS) or a single 4090.
"""
import torch
import torch.nn as nn
from .mla import MultiHeadLatentAttention


class MLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_rope, d_c, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_head, d_rope, d_c, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x, return_latent=False, rank_mask=None):
        h = self.ln1(x)
        if return_latent:
            a, c_kv = self.attn(h, return_latent=True, rank_mask=rank_mask)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, c_kv
        else:
            x = x + self.attn(h, rank_mask=rank_mask)
            x = x + self.mlp(self.ln2(x))
            return x


class MLAGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=8, n_heads=6,
                 d_head=64, d_rope=32, d_c=256, max_len=1024, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_head, d_rope, d_c, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.max_len = max_len
        self.d_c = d_c
        self.n_layers = n_layers
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, rank_mask=None, layer_idx_for_latent=None):
        x = self.drop(self.tok_emb(idx))
        latents = {}
        for i, blk in enumerate(self.blocks):
            want_latent = (layer_idx_for_latent is not None and i == layer_idx_for_latent)
            if want_latent:
                x, c_kv = blk(x, return_latent=True, rank_mask=rank_mask)
                latents[i] = c_kv
            else:
                # rank_mask (channel truncation) is independent of the latent-return
                # request: layer_idx_for_latent only controls which layer's c_kv is
                # returned to the caller, it must not change which layers get the
                # rank_mask applied. Previously this branch passed rank_mask=None to
                # every non-selected layer whenever layer_idx_for_latent was set,
                # silently disabling truncation on all-but-one layer (P1 bug).
                x = blk(x, rank_mask=rank_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if layer_idx_for_latent is not None:
            return logits, loss, latents.get(layer_idx_for_latent)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
