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


    def forward_cached(self, x, cache=None, rank_mask=None):
        """Run this block on new tokens and update its compressed MLA cache."""
        h = self.ln1(x)
        a, new_cache = self.attn.forward_cached(h, cache=cache, rank_mask=rank_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


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

    def forward_cached(self, idx, caches=None, rank_masks=None):
        """Incrementally process new token IDs using per-layer compressed MLA caches.

        Args:
            idx: New token IDs ``(B, T_new)``.
            caches: ``None`` for prefill, or one cache dict per transformer layer.
            rank_masks: Optional mask shared by every layer, or a list/tuple of
                length ``n_layers`` containing independent masks.  Each mask is
                passed to the corresponding layer and applies only to new tokens.

        Returns:
            ``(logits, new_caches)`` for the new positions only.
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, new_sequence_length)")
        if caches is None:
            caches = [None] * self.n_layers
        if len(caches) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layer caches, got {len(caches)}")

        present = [cache is not None for cache in caches]
        if any(present) and not all(present):
            raise ValueError("layer caches must be either all None or all populated")
        if all(present):
            lengths = {cache["c_kv"].shape[1] for cache in caches}
            batches = {cache["c_kv"].shape[0] for cache in caches}
            if len(lengths) != 1:
                raise ValueError("all layer caches must have the same past length")
            if batches != {idx.shape[0]}:
                raise ValueError("all layer cache batch sizes must match idx")

        if isinstance(rank_masks, (list, tuple)):
            if len(rank_masks) != self.n_layers:
                raise ValueError(
                    f"expected {self.n_layers} layer rank masks, got {len(rank_masks)}"
                )
            masks = list(rank_masks)
        else:
            masks = [rank_masks] * self.n_layers

        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for block, cache, mask in zip(self.blocks, caches, masks):
            x, new_cache = block.forward_cached(x, cache=cache, rank_mask=mask)
            new_caches.append(new_cache)
        logits = self.head(self.ln_f(x))
        return logits, new_caches

    def cache_num_bytes(self, caches):
        """Actual bytes allocated by all compressed per-layer MLA caches."""
        if caches is None:
            return 0
        if len(caches) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layer caches, got {len(caches)}")
        return sum(block.attn.cache_num_bytes(cache) for block, cache in zip(self.blocks, caches))

    def theoretical_mha_cache_num_bytes(self, batch_size, sequence_length, dtype):
        """Conservative standard-MHA K/V cache byte estimate.

        A standard RoPE MHA rotates coordinates *within* each head rather than
        appending a separately cached RoPE sub-vector.  The comparison therefore
        counts one ``d_head`` key and one ``d_head`` value per head.  This is more
        conservative than a shape-matched comparator that gives MHA this model's
        additional decoupled ``d_rope`` key coordinates.
        """
        element_size = torch.empty((), dtype=dtype).element_size()
        values_per_token_layer = 2 * self.blocks[0].attn.n_heads * self.blocks[0].attn.d_head
        return batch_size * sequence_length * self.n_layers * values_per_token_layer * element_size

    def theoretical_shape_matched_mha_cache_num_bytes(
        self, batch_size, sequence_length, dtype
    ):
        """MHA bytes when matching this model's ``d_head+d_rope`` key width.

        This secondary number is useful for architecture-shape accounting, but
        should not be presented as the primary standard-MHA baseline.
        """
        element_size = torch.empty((), dtype=dtype).element_size()
        attn = self.blocks[0].attn
        values_per_token_layer = attn.n_heads * (2 * attn.d_head + attn.d_rope)
        return batch_size * sequence_length * self.n_layers * values_per_token_layer * element_size

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation using the compressed MLA cache."""
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must be a non-empty (batch, sequence) tensor")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return idx

        logits, caches = self.forward_cached(idx)
        generated = idx
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(int(top_k), next_logits.shape[-1])
                values, _ = torch.topk(next_logits, k)
                next_logits = next_logits.masked_fill(
                    next_logits < values[:, [-1]], float("-inf")
                )
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_idx), dim=1)
            # No caller will consume logits after the final requested token.
            if step + 1 < max_new_tokens:
                logits, caches = self.forward_cached(next_idx, caches=caches)
        return generated

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
