"""Tiered rank routers and a packed-cache ElasticMLA wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TieredRankRouter(nn.Module):
    """Predict one discrete latent-width tier from a token hidden state."""

    def __init__(self, d_model, tiers=(16, 64, 160, 256), hidden_dim=None):
        super().__init__()
        tiers = tuple(int(t) for t in tiers)
        if not tiers or any(t <= 0 for t in tiers) or tuple(sorted(set(tiers))) != tiers:
            raise ValueError("tiers must be unique, positive, and strictly increasing")
        hidden_dim = hidden_dim or max(32, d_model // 4)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(tiers)),
        )
        self.register_buffer("tiers", torch.tensor(tiers, dtype=torch.long), persistent=True)

    def forward(self, hidden):
        return self.net(hidden)

    def select_ranks(self, logits):
        return self.tiers[logits.argmax(dim=-1)]

    def expected_rank(self, logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probs = F.softmax(logits / temperature, dim=-1)
        return (probs * self.tiers.to(logits.dtype)).sum(dim=-1)

    def targets_to_indices(self, target_ranks):
        matches = target_ranks[..., None] == self.tiers
        if not torch.all(matches.any(dim=-1)):
            raise ValueError("every target rank must be one of the configured tiers")
        return matches.to(torch.long).argmax(dim=-1)

    def supervised_loss(self, logits, target_ranks, class_weights=None):
        targets = self.targets_to_indices(target_ranks)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            weight=class_weights,
        )


class ElasticMLAGPT(nn.Module):
    """Attach per-layer tier routers to a pretrained ``MLAGPT`` base model.

    The base model owns all language-model parameters.  Routers observe the
    pre-attention normalized hidden state at each layer and choose how many
    saliency-ordered latent channels are persistently stored for each new token.
    """

    def __init__(self, base_model, channel_orders, tiers=(16, 64, 160, 256)):
        super().__init__()
        if len(channel_orders) != base_model.n_layers:
            raise ValueError("channel_orders must contain one permutation per layer")
        if max(tiers) > base_model.d_c:
            raise ValueError("router tier exceeds base model latent dimension")
        self.base = base_model
        self.routers = nn.ModuleList(
            TieredRankRouter(base_model.tok_emb.embedding_dim, tiers=tiers)
            for _ in range(base_model.n_layers)
        )
        for i, order in enumerate(channel_orders):
            order = torch.as_tensor(order, dtype=torch.long)
            if order.shape != (base_model.d_c,) or not torch.equal(
                torch.sort(order).values,
                torch.arange(base_model.d_c, device=order.device),
            ):
                raise ValueError(f"invalid channel order for layer {i}")
            self.register_buffer(f"channel_order_{i}", order, persistent=True)

    @property
    def channel_orders(self):
        return [getattr(self, f"channel_order_{i}") for i in range(self.base.n_layers)]

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        return self

    @torch.no_grad()
    def routing_features_full(self, idx):
        """Extract detached pre-attention features for offline router training."""
        x = self.base.drop(self.base.tok_emb(idx))
        features = []
        for block in self.base.blocks:
            h = block.ln1(x)
            features.append(h.detach())
            x = x + block.attn(h)
            x = x + block.mlp(block.ln2(x))
        return features

    def routing_logits_full(self, idx):
        """Return router logits on an ordinary full-attention teacher-forced pass."""
        x = self.base.drop(self.base.tok_emb(idx))
        router_logits = []
        for block, router in zip(self.base.blocks, self.routers):
            h = block.ln1(x)
            router_logits.append(router(h))
            x = x + block.attn(h)
            x = x + block.mlp(block.ln2(x))
        lm_logits = self.base.head(self.base.ln_f(x))
        return lm_logits, router_logits

    def forward_cached_packed(self, idx, caches=None, forced_ranks=None):
        """Packed incremental forward using predicted or externally supplied tiers."""
        if idx.ndim != 2:
            raise ValueError("idx must have shape (B, T_new)")
        if caches is None:
            caches = [None] * self.base.n_layers
        if len(caches) != self.base.n_layers:
            raise ValueError("wrong number of layer caches")
        present = [cache is not None for cache in caches]
        if any(present) and not all(present):
            raise ValueError("packed layer caches must be all None or all populated")
        if all(present):
            lengths = {cache["ranks"].shape[1] for cache in caches}
            batches = {cache["ranks"].shape[0] for cache in caches}
            if len(lengths) != 1:
                raise ValueError("all packed layer caches must have the same length")
            if batches != {idx.shape[0]}:
                raise ValueError("packed cache batch sizes must match idx")
        if forced_ranks is not None and len(forced_ranks) != self.base.n_layers:
            raise ValueError("forced_ranks must contain one tensor per layer")

        x = self.base.drop(self.base.tok_emb(idx))
        new_caches, chosen_ranks, all_router_logits = [], [], []
        for i, (block, router, cache, order) in enumerate(
            zip(self.base.blocks, self.routers, caches, self.channel_orders)
        ):
            h = block.ln1(x)
            route_logits = router(h)
            ranks = (
                forced_ranks[i].to(h.device)
                if forced_ranks is not None
                else router.select_ranks(route_logits)
            )
            if forced_ranks is not None:
                valid = (ranks[..., None] == router.tiers).any(dim=-1)
                if not torch.all(valid):
                    raise ValueError(
                        f"forced ranks for layer {i} must belong to {router.tiers.tolist()}"
                    )
            a, new_cache = block.attn.forward_cached_packed(
                h, cache=cache, ranks=ranks, channel_order=order
            )
            x = x + a
            x = x + block.mlp(block.ln2(x))
            new_caches.append(new_cache)
            chosen_ranks.append(ranks)
            all_router_logits.append(route_logits)
        return self.base.head(self.base.ln_f(x)), new_caches, chosen_ranks, all_router_logits

    def packed_cache_num_bytes(self, caches):
        return self.base.packed_cache_num_bytes(caches)


class GlobalElasticMLAGPT(nn.Module):
    """One token-difficulty router chooses a shared tier for every MLA layer.

    This matches an oracle obtained by intervening on all layers at the same rank.
    The router observes the first block's pre-attention normalized hidden state,
    which is unaffected by earlier compression and therefore avoids rollout feature
    shift at the routing decision.
    """

    def __init__(self, base_model, channel_orders, tiers=(16, 64, 160, 256)):
        super().__init__()
        self.base = base_model
        self.router = TieredRankRouter(base_model.tok_emb.embedding_dim, tiers=tiers)
        if len(channel_orders) != base_model.n_layers:
            raise ValueError("channel_orders must contain one order per layer")
        for i, order in enumerate(channel_orders):
            order = torch.as_tensor(order, dtype=torch.long)
            if order.shape != (base_model.d_c,) or not torch.equal(
                torch.sort(order).values,
                torch.arange(base_model.d_c, device=order.device),
            ):
                raise ValueError(f"invalid channel order for layer {i}")
            self.register_buffer(f"channel_order_{i}", order, persistent=True)

    @property
    def channel_orders(self):
        return [getattr(self, f"channel_order_{i}") for i in range(self.base.n_layers)]

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        return self

    @torch.no_grad()
    def routing_features(self, idx):
        x = self.base.drop(self.base.tok_emb(idx))
        return self.base.blocks[0].ln1(x).detach()

    def forward_cached_packed(self, idx, caches=None, forced_ranks=None):
        x0 = self.base.drop(self.base.tok_emb(idx))
        feature = self.base.blocks[0].ln1(x0)
        route_logits = self.router(feature)
        ranks = (
            forced_ranks.to(idx.device)
            if forced_ranks is not None
            else self.router.select_ranks(route_logits)
        )
        if forced_ranks is not None:
            valid = (ranks[..., None] == self.router.tiers).any(dim=-1)
            if not torch.all(valid):
                raise ValueError(f"forced ranks must belong to {self.router.tiers.tolist()}")
        logits, new_caches = self.base.forward_cached_packed(
            idx,
            ranks=ranks,
            channel_orders=self.channel_orders,
            caches=caches,
        )
        return logits, new_caches, ranks, route_logits

    def packed_cache_num_bytes(self, caches):
        return self.base.packed_cache_num_bytes(caches)
