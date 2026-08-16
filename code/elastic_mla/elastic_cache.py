"""Packed variable-width latent cache utilities for ElasticMLA.

The persistent representation stores only the selected latent channel values for
each token, plus compact offsets/rank metadata.  Dense ``(B,S,d_c)`` tensors are
created only as temporary reconstruction buffers for the correctness-first
attention implementation.
"""
from __future__ import annotations

import torch


PACKED_KEYS = {"values", "offsets", "ranks", "channel_order", "k_rope"}


def validate_channel_order(channel_order: torch.Tensor, d_c: int) -> torch.Tensor:
    order = torch.as_tensor(channel_order, dtype=torch.long)
    if order.ndim != 1 or order.numel() != d_c:
        raise ValueError(f"channel_order must have shape ({d_c},)")
    if not torch.equal(torch.sort(order.cpu()).values, torch.arange(d_c)):
        raise ValueError("channel_order must be a permutation of 0..d_c-1")
    return order


def validate_ranks(ranks: torch.Tensor, batch: int, length: int, d_c: int) -> torch.Tensor:
    ranks = torch.as_tensor(ranks)
    if ranks.shape != (batch, length):
        raise ValueError(f"ranks must have shape ({batch}, {length})")
    if ranks.dtype == torch.bool or ranks.is_floating_point():
        raise ValueError("ranks must use an integer dtype")
    ranks = ranks.to(dtype=torch.int64)
    if torch.any(ranks <= 0) or torch.any(ranks > d_c):
        raise ValueError(f"all ranks must be in [1, {d_c}]")
    return ranks


def pack_latents(c_kv: torch.Tensor, ranks: torch.Tensor, channel_order: torch.Tensor):
    """Pack ``c_kv`` values selected by token-specific prefix ranks.

    Token ordering in the flat buffer is row-major ``(batch, sequence)``.
    ``offsets`` has length ``B*S+1`` and int32 dtype; ``ranks`` is stored as
    int16 when possible to keep metadata small.
    """
    if c_kv.ndim != 3:
        raise ValueError("c_kv must have shape (B, S, d_c)")
    B, S, d_c = c_kv.shape
    order = validate_channel_order(channel_order, d_c).to(c_kv.device)
    ranks64 = validate_ranks(ranks, B, S, d_c).to(c_kv.device)

    flat = c_kv.reshape(B * S, d_c)
    flat_ranks = ranks64.reshape(-1)
    pieces = [flat[i, order[: int(rank)]] for i, rank in enumerate(flat_ranks.tolist())]
    values = torch.cat(pieces, dim=0) if pieces else c_kv.new_empty((0,))
    cumulative = torch.cat(
        (torch.zeros(1, device=c_kv.device, dtype=torch.int64), flat_ranks.cumsum(0))
    )
    if cumulative[-1].item() >= 2**31:
        raise OverflowError("packed cache offsets exceed int32 capacity")
    rank_dtype = torch.int16 if d_c <= torch.iinfo(torch.int16).max else torch.int32
    return {
        "values": values.detach(),
        "offsets": cumulative.to(torch.int32).detach(),
        "ranks": ranks64.to(rank_dtype).detach(),
        # Store the exact coordinate system used to interpret packed values.
        # int16 is sufficient whenever ranks themselves fit int16.
        "channel_order": order.to(rank_dtype).detach(),
    }


def unpack_latents(cache, d_c: int, channel_order: torch.Tensor) -> torch.Tensor:
    """Reconstruct a dense latent tensor from packed persistent storage."""
    for key in ("values", "offsets", "ranks", "channel_order"):
        if key not in cache:
            raise ValueError(f"packed cache missing {key!r}")
    values, offsets, ranks = cache["values"], cache["offsets"], cache["ranks"]
    if ranks.ndim != 2:
        raise ValueError("packed ranks must have shape (B, S)")
    B, S = ranks.shape
    order = validate_channel_order(channel_order, d_c).to(values.device)
    stored_order = cache["channel_order"].to(device=values.device, dtype=torch.long)
    if not torch.equal(stored_order, order):
        raise ValueError("channel_order does not match the order used to create this cache")
    flat_ranks = ranks.to(torch.int64).reshape(-1)
    if offsets.shape != (B * S + 1,):
        raise ValueError("packed offsets length is inconsistent with ranks")
    if offsets[0].item() != 0 or offsets[-1].item() != values.numel():
        raise ValueError("packed offsets do not span the values buffer")
    if not torch.equal(offsets[1:].to(torch.int64) - offsets[:-1].to(torch.int64), flat_ranks):
        raise ValueError("packed offsets and ranks disagree")

    dense = values.new_zeros((B * S, d_c))
    for i, rank in enumerate(flat_ranks.tolist()):
        start, end = int(offsets[i]), int(offsets[i + 1])
        dense[i, order[:rank]] = values[start:end]
    return dense.view(B, S, d_c)


def append_packed_latents(cache, c_new, ranks_new, channel_order):
    """Append new tokens while preserving row-major batch/sequence ordering."""
    if cache is None:
        return pack_latents(c_new, ranks_new, channel_order)
    old_dense = unpack_latents(cache, c_new.shape[-1], channel_order)
    if old_dense.shape[0] != c_new.shape[0]:
        raise ValueError("packed cache batch size does not match new latents")
    old_ranks = cache["ranks"].to(torch.int64)
    ranks_new = validate_ranks(
        ranks_new, c_new.shape[0], c_new.shape[1], c_new.shape[2]
    ).to(old_ranks.device)
    combined_dense = torch.cat((old_dense, c_new), dim=1)
    combined_ranks = torch.cat((old_ranks, ranks_new), dim=1)
    return pack_latents(combined_dense, combined_ranks, channel_order)


def packed_cache_num_bytes(cache) -> int:
    if cache is None:
        return 0
    return sum(t.numel() * t.element_size() for t in cache.values())
