"""Packed variable-width latent cache utilities for ElasticMLA.

The persistent representation stores only the selected latent channel values for
each token, plus compact offsets/rank metadata.  Dense ``(B,S,d_c)`` tensors are
created only as temporary reconstruction buffers for the correctness-first
attention implementation.

Tokens are laid out in the flat ``values``/``offsets`` buffers in **sequence-major**
``(s, b)`` order: flat token index ``k`` denotes sequence position ``k // B`` of
batch row ``k % B``.  This ordering is what makes incremental decoding cheap --
every token of a new step sorts after every token already stored, so appending is
a plain concatenation of the packed buffers instead of a full unpack/repack of the
whole history (see ``append_packed_latents``).  Row-major ``(b, s)`` ordering would
interleave the new tokens into the middle of the buffer and force a rebuild on
every decode step.  The choice is invisible to callers: ``ranks`` keeps its
``(B, S)`` shape, dense reconstruction is bit-identical either way, and the
persistent byte count is unchanged.
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


def _as_order(channel_order: torch.Tensor, d_c: int) -> torch.Tensor:
    """Shape/dtype-only conversion of an already-guaranteed channel order.

    Used on the incremental-decode path, where the order is checked for equality
    against the cache's own ``channel_order`` -- which ``pack_latents`` fully
    validated when the cache was created.  Equality with a validated permutation
    is itself a proof of permutation-ness, so re-running the sort-based check on
    every decode step is redundant work, and on an accelerator it costs a
    device-to-host copy and a synchronisation per call.
    """
    order = torch.as_tensor(channel_order, dtype=torch.long)
    if order.ndim != 1 or order.numel() != d_c:
        raise ValueError(f"channel_order must have shape ({d_c},)")
    return order


def validate_ranks(ranks: torch.Tensor, batch: int, length: int, d_c: int) -> torch.Tensor:
    ranks = torch.as_tensor(ranks)
    if ranks.shape != (batch, length):
        raise ValueError(f"ranks must have shape ({batch}, {length})")
    if ranks.dtype == torch.bool or ranks.is_floating_point():
        raise ValueError("ranks must use an integer dtype")
    ranks = ranks.to(dtype=torch.int64)
    # One fused comparison, so the bounds check costs a single synchronisation.
    if bool(torch.any((ranks <= 0) | (ranks > d_c))):
        raise ValueError(f"all ranks must be in [1, {d_c}]")
    return ranks


def _ragged_prefix_positions(flat_ranks: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized helper for ragged nested-prefix packing/unpacking.

    Given the per-token rank of each of ``N`` tokens, returns
    ``(token_index, prefix_position)`` of length ``sum(flat_ranks)``: for the
    ``k``-th value in the flat packed buffer, ``token_index[k]`` is which
    token it belongs to and ``prefix_position[k]`` is its 0-indexed position
    within that token's nested prefix (``0 <= prefix_position[k] < rank``).
    Replaces an equivalent per-token Python loop with ``repeat_interleave``.
    """
    token_index = torch.repeat_interleave(
        torch.arange(flat_ranks.shape[0], device=device), flat_ranks
    )
    starts = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int64), flat_ranks.cumsum(0)[:-1])
    )
    segment_starts = torch.repeat_interleave(starts, flat_ranks)
    # ``token_index`` already has the flat buffer's length, so read it from there
    # rather than paying a second reduction-plus-synchronisation on ``flat_ranks``.
    prefix_position = (
        torch.arange(token_index.shape[0], device=device) - segment_starts
    )
    return token_index, prefix_position


def _sequence_major_ranks(ranks64: torch.Tensor) -> torch.Tensor:
    """Flatten ``(B, S)`` ranks into the sequence-major ``(s, b)`` flat-buffer order."""
    return ranks64.transpose(0, 1).reshape(-1)


def _flat_token_coordinates(token_index: torch.Tensor, batch: int):
    """Split sequence-major flat token indices into ``(batch_index, position_index)``."""
    return token_index % batch, token_index // batch


def pack_latents(c_kv: torch.Tensor, ranks: torch.Tensor, channel_order: torch.Tensor):
    """Pack ``c_kv`` values selected by token-specific prefix ranks.

    Token ordering in the flat buffer is sequence-major ``(sequence, batch)``; see
    the module docstring for why.  ``offsets`` has length ``B*S+1`` and int32
    dtype; ``ranks`` is stored as int16 when possible to keep metadata small.
    Vectorized: no per-token Python loop, using ``repeat_interleave`` plus a
    single gather.
    """
    if c_kv.ndim != 3:
        raise ValueError("c_kv must have shape (B, S, d_c)")
    B, S, d_c = c_kv.shape
    order = validate_channel_order(channel_order, d_c).to(c_kv.device)
    packed = _pack_validated(c_kv, ranks, order)
    if int(packed["offsets"][-1]) >= 2**31:
        raise OverflowError("packed cache offsets exceed int32 capacity")
    return packed


def _pack_validated(c_kv: torch.Tensor, ranks: torch.Tensor, order: torch.Tensor):
    """Packing core for an already-validated channel order.

    Split out of ``pack_latents`` so the incremental-decode path can pack one new
    chunk without repeating the sort-based permutation check on an order it has
    already matched against the cache's stored one.
    """
    B, S, d_c = c_kv.shape
    order = order.to(c_kv.device)
    ranks64 = validate_ranks(ranks, B, S, d_c).to(c_kv.device)

    flat_ranks = _sequence_major_ranks(ranks64)
    cumulative = torch.cat(
        (torch.zeros(1, device=c_kv.device, dtype=torch.int64), flat_ranks.cumsum(0))
    )
    # ``validate_ranks`` guarantees every rank is at least 1, so an empty flat
    # buffer is exactly an empty token set -- no reduction-and-synchronisation
    # needed to detect it.  The int32 overflow guard lives in the callers, which
    # each already read the final offset once.
    if flat_ranks.numel() == 0:
        values = c_kv.new_empty((0,))
    else:
        token_index, prefix_position = _ragged_prefix_positions(flat_ranks, c_kv.device)
        batch_index, position_index = _flat_token_coordinates(token_index, B)
        values = c_kv[batch_index, position_index, order[prefix_position]]
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
    order = _as_order(channel_order, d_c).to(values.device)
    stored_order = cache["channel_order"].to(device=values.device, dtype=torch.long)
    if not torch.equal(stored_order, order):
        raise ValueError("channel_order does not match the order used to create this cache")
    flat_ranks = _sequence_major_ranks(ranks.to(torch.int64))
    if offsets.shape != (B * S + 1,):
        raise ValueError("packed offsets length is inconsistent with ranks")
    # Same two guarantees as before -- offsets start at 0, end at the buffer
    # length, and their differences are exactly the per-token ranks -- but folded
    # into one device-side comparison so the decode path pays a single
    # synchronisation instead of one per ``.item()``.
    offsets64 = offsets.to(torch.int64)
    expected = torch.cat(
        (torch.zeros(1, device=offsets64.device, dtype=torch.int64), flat_ranks.cumsum(0))
    )
    if not torch.equal(offsets64, expected):
        raise ValueError("packed offsets and ranks disagree")
    if int(offsets64[-1]) != values.numel():
        raise ValueError("packed offsets do not span the values buffer")

    dense = values.new_zeros((B, S, d_c))
    if values.numel() > 0:
        token_index, prefix_position = _ragged_prefix_positions(flat_ranks, values.device)
        batch_index, position_index = _flat_token_coordinates(token_index, B)
        dense[batch_index, position_index, order[prefix_position]] = values
    return dense


def append_packed_latents(cache, c_new, ranks_new, channel_order):
    """Append new tokens to a packed cache in place, without rebuilding the history.

    Cost is ``O(B * T_new)`` in the number of newly packed values plus a
    concatenation of the existing buffers, rather than the ``O(B * S)`` full
    unpack/repack round trip through a dense ``(B, S, d_c)`` buffer that this
    function used to perform on every decode step.  This is possible because the
    flat buffer is sequence-major (module docstring): every new token sorts after
    every stored token, so the new packed values and offsets simply extend the
    existing ones.  The result is bit-identical to packing the concatenated dense
    history from scratch.
    """
    if cache is None:
        return pack_latents(c_new, ranks_new, channel_order)
    if c_new.ndim != 3:
        raise ValueError("c_new must have shape (B, T_new, d_c)")
    B, _, d_c = c_new.shape
    order = _as_order(channel_order, d_c).to(c_new.device)
    stored_order = cache["channel_order"].to(device=c_new.device, dtype=torch.long)
    if not torch.equal(stored_order, order):
        raise ValueError("channel_order does not match the order used to create this cache")
    old_ranks = cache["ranks"].to(torch.int64)
    if old_ranks.ndim != 2:
        raise ValueError("packed ranks must have shape (B, S)")
    if old_ranks.shape[0] != B:
        raise ValueError("packed cache batch size does not match new latents")

    appended = _pack_validated(c_new, ranks_new, order)
    old_offsets = cache["offsets"].to(torch.int64)
    offsets = torch.cat(
        (old_offsets, old_offsets[-1] + appended["offsets"][1:].to(torch.int64))
    )
    # ``pack_latents`` already bounds the appended chunk; the running total is the
    # only thing that can newly overflow int32, and it is the last offset.
    if int(offsets[-1]) >= 2**31:
        raise OverflowError("packed cache offsets exceed int32 capacity")
    return {
        "values": torch.cat((cache["values"], appended["values"])),
        "offsets": offsets.to(torch.int32),
        "ranks": torch.cat(
            (cache["ranks"], appended["ranks"].to(cache["ranks"].device)), dim=1
        ),
        "channel_order": cache["channel_order"],
    }


def packed_cache_num_bytes(cache) -> int:
    if cache is None:
        return 0
    return sum(t.numel() * t.element_size() for t in cache.values())
