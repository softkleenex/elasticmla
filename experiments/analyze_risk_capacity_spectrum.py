"""Risk-capacity spectrum for layer-aligned MLA rank interventions.

This v5 analysis retains the corrected v4 alignment and nonoverlapping windows,
but measures upper-tail means of the future-loss vector. Tail count ``k`` is the
mean of the largest ``k`` losses in the exact horizon: k=H is the ordinary mean
and k=1 is the maximum. This produces a nested risk spectrum suitable for testing
risk-capacity monotonicity and the tail-capacity premium.

This remains a full-attention truncation simulation; it is not compressed-cache
autoregressive decoding.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from elastic_mla import MLAGPT  # noqa: E402


DATA_DIR = ROOT / "experiments" / "exp0_rank_variance" / "data"
CKPT_DIR = ROOT / "experiments" / "exp0_rank_variance" / "ckpt"
OUT_DIR = ROOT / "experiments" / "exp0_rank_variance" / "results"

DEFAULT_RANK_GRID = (16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384)
CALIBRATION_SEED_BASE = 12_341
EVALUATION_SEED = 23_452
BOOTSTRAP_SEED = 34_563



def file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def choose_device(requested: str = "auto") -> torch.device:
    """Select cuda > mps > cpu, or validate an explicit selection."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def empty_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def normalize_rank_grid(ranks: Iterable[int], d_c: int) -> list[int]:
    """Return a sorted valid grid that always includes exact full rank."""
    grid = sorted({int(rank) for rank in ranks if 0 < int(rank) <= d_c})
    if d_c not in grid:
        grid.append(d_c)
    if not grid:
        raise ValueError("rank grid contains no positive rank at or below d_c")
    return grid


def valid_probe_positions(sequence_length: int, horizon: int) -> np.ndarray:
    """Source positions whose next-token losses have the exact horizon.

    Masking the latent at source position ``pos`` changes ``logits[pos]`` itself;
    that logit predicts token ``pos + 1``.  Thus the loss window is
    ``[pos, pos + horizon)`` rather than starting at ``pos + 1``.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    count = sequence_length - horizon + 1
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(count, dtype=np.int64)


def future_loss_slice(pos: int, horizon: int, loss_length: int) -> slice:
    """Return the exact next-token loss window affected from source ``pos``."""
    start = int(pos)
    stop = start + int(horizon)
    if start < 0 or horizon <= 0 or stop > loss_length:
        raise ValueError("probe does not have the exact requested horizon")
    return slice(start, stop)


def suffix_all_satisfy_r_star(
    deltas: Sequence[float], ranks: Sequence[int], epsilon: float
) -> int:
    """Smallest rank whose value and every higher-rank value satisfy epsilon."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (len(ranks),):
        raise ValueError("deltas and ranks must have the same non-empty length")
    satisfies = np.isfinite(values) & (values <= epsilon)
    suffix_satisfies = np.logical_and.accumulate(satisfies[::-1])[::-1]
    indices = np.flatnonzero(suffix_satisfies)
    if not len(indices):
        raise ValueError("no tested rank has a suffix-safe distortion at this epsilon")
    return int(ranks[int(indices[0])])


def is_nonmonotonic(deltas: Sequence[float], tolerance: float = 1e-8) -> bool:
    """Whether loss effect rises at any adjacent increase in retained rank."""
    values = np.asarray(deltas, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return bool(len(finite) > 1 and np.any(np.diff(finite) > tolerance))


def normalize_tail_counts(counts: Iterable[int], horizon: int) -> list[int]:
    """Valid unique tail counts, ordered from mean-like (H) to max-like (1)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    values = {int(k) for k in counts if 1 <= int(k) <= horizon}
    values.update((horizon, 1))
    return sorted(values, reverse=True)


def upper_tail_mean(values: Sequence[float], tail_count: int) -> float:
    """Mean of the largest ``tail_count`` finite-horizon losses."""
    array = np.asarray(values)
    if array.ndim != 1 or not 1 <= tail_count <= len(array):
        raise ValueError("tail_count must be in [1, len(values)]")
    if tail_count == len(array):
        # Preserve v4's float32 reduction order exactly at the mean endpoint.
        return float(array.mean())
    if tail_count == 1:
        return float(array.max())
    partitioned = np.partition(array, len(array) - tail_count)
    return float(partitioned[-tail_count:].mean())


def sequence_cluster_bootstrap_mean_ci(
    values_by_sequence: dict[int, Sequence[float]],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI after resampling whole sequences with replacement."""
    if n_bootstrap < 1000:
        raise ValueError("headline sequence-cluster bootstrap requires >= 1000 draws")
    if not values_by_sequence:
        raise ValueError("cannot bootstrap an empty collection")
    clusters = [np.asarray(v, dtype=np.float64) for _, v in sorted(values_by_sequence.items())]
    if any(cluster.size == 0 for cluster in clusters):
        raise ValueError("every sequence cluster must contain at least one value")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n_clusters = len(clusters)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        total = sum(float(clusters[i].sum()) for i in sampled)
        count = sum(int(clusters[i].size) for i in sampled)
        means[draw] = total / count
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    batch, length, vocab = logits.shape
    losses = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    )
    return losses.view(batch, length).detach().cpu().numpy()


def forward_with_layer_masks(
    model: MLAGPT,
    idx: torch.Tensor,
    *,
    channel_masks: torch.Tensor | None = None,
    probe_positions: torch.Tensor | None = None,
    masked_layers: Sequence[int] | None = None,
) -> torch.Tensor:
    """Model forward with a different position-isolated channel mask per layer.

    ``channel_masks`` has shape (n_layers, batch, d_c).  For item b, its mask is
    applied only at ``probe_positions[b]``; all other positions remain full rank.
    Keeping this helper local avoids changing the training model's public API.
    """
    if (channel_masks is None) != (probe_positions is None):
        raise ValueError("channel_masks and probe_positions must be supplied together")
    if channel_masks is not None:
        expected = (model.n_layers, idx.shape[0], model.d_c)
        if tuple(channel_masks.shape) != expected:
            raise ValueError(f"channel_masks shape must be {expected}")
        if tuple(probe_positions.shape) != (idx.shape[0],):
            raise ValueError("probe_positions must have shape (batch,)")

    active_layers = set(range(model.n_layers)) if masked_layers is None else set(masked_layers)
    if any(layer < 0 or layer >= model.n_layers for layer in active_layers):
        raise ValueError("masked_layers contains an invalid layer index")
    hidden = model.drop(model.tok_emb(idx))
    batch_indices = torch.arange(idx.shape[0], device=idx.device)
    for layer_idx, block in enumerate(model.blocks):
        rank_mask = None
        if channel_masks is not None and layer_idx in active_layers:
            # Only one (source) position per batch item is intervened on.
            rank_mask = torch.ones(
                idx.shape[0], idx.shape[1], model.d_c,
                device=hidden.device, dtype=hidden.dtype,
            )
            rank_mask[batch_indices, probe_positions] = channel_masks[layer_idx].to(hidden.dtype)
        hidden = block(hidden, rank_mask=rank_mask)
    return model.head(model.ln_f(hidden))


def forward_with_all_latents(
    model: MLAGPT, idx: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return logits and every layer's latent from one calibration forward."""
    hidden = model.drop(model.tok_emb(idx))
    latents = []
    for block in model.blocks:
        hidden, latent = block(hidden, return_latent=True)
        latents.append(latent)
    return model.head(model.ln_f(hidden)), latents


def sample_starts(
    rng: np.random.Generator,
    low: int,
    high_exclusive: int,
    count: int,
    *,
    min_separation: int = 1,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Sample starts whose full token windows cannot overlap.

    ``min_separation`` should be the number of tokens read per sequence
    (``block_size + 1`` here).  ``excluded`` permits disjoint sampling across
    calibration repeats as well as within each repeat.
    """
    if count < 0 or min_separation <= 0 or high_exclusive <= low:
        raise ValueError("invalid sampling bounds/count/separation")
    accepted: list[int] = []
    forbidden = [int(value) for value in excluded]
    max_attempts = max(10_000, count * 10_000)
    for _ in range(max_attempts):
        if len(accepted) == count:
            return np.sort(np.asarray(accepted, dtype=np.int64))
        candidate = int(rng.integers(low, high_exclusive))
        if all(abs(candidate - other) >= min_separation for other in forbidden + accepted):
            accepted.append(candidate)
    raise ValueError("could not sample the requested number of nonoverlapping windows")


def sequence_batch(data: np.memmap, starts: Sequence[int], block_size: int) -> np.ndarray:
    return np.stack(
        [np.asarray(data[int(s): int(s) + block_size + 1], dtype=np.int64) for s in starts]
    )


def calibrate_layer_orders(
    model: MLAGPT,
    data: np.memmap,
    starts_by_repeat: Sequence[Sequence[int]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream calibration and return aggregate orders plus repeat saliencies."""
    repeats = len(starts_by_repeat)
    saliency_by_repeat = np.zeros((repeats, model.n_layers, model.d_c), dtype=np.float64)
    for repeat_idx, starts in enumerate(starts_by_repeat):
        for offset in range(0, len(starts), batch_size):
            tokens = sequence_batch(data, starts[offset: offset + batch_size], block_size)
            x = torch.from_numpy(tokens[:, :-1]).to(device)
            y = torch.from_numpy(tokens[:, 1:]).to(device)
            model.zero_grad(set_to_none=True)
            logits, latents = forward_with_all_latents(model, x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            grads = torch.autograd.grad(loss, latents, only_inputs=True)
            for layer_idx, (latent, grad) in enumerate(zip(latents, grads)):
                # loss is a token mean; restore batch-size weighting so a short
                # final microbatch does not count like a full microbatch.
                saliency_by_repeat[repeat_idx, layer_idx] += (
                    (grad * latent.detach()).abs().sum(dim=(0, 1)).cpu().double().numpy()
                    * latent.shape[0]
                )
            del tokens, x, y, logits, loss, latents, grads
            empty_device_cache(device)
        print(f"calibration repeat={repeat_idx + 1}/{repeats}", flush=True)
    aggregate = saliency_by_repeat.mean(axis=0)
    orders = np.argsort(-aggregate, axis=1)
    return orders, saliency_by_repeat


def make_channel_masks(
    layer_orders: np.ndarray, rank: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    masks = torch.zeros(
        layer_orders.shape[0], batch_size, layer_orders.shape[1],
        dtype=torch.float32, device=device,
    )
    for layer_idx, order in enumerate(layer_orders):
        keep = torch.as_tensor(order[:rank].copy(), dtype=torch.long, device=device)
        masks[layer_idx, :, keep] = 1.0
    return masks


def token_type(token_id: int, encoder) -> str:
    text = encoder.decode([int(token_id)])
    stripped = text.strip()
    if not stripped:
        return "space"
    if all(char in ".,!?;:'\"-()" for char in stripped):
        return "punct"
    if stripped[0].isupper():
        return "capitalized"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-horizon", type=int, default=32)
    parser.add_argument("--tail-counts", type=int, nargs="+", default=[32, 16, 8, 4, 2, 1],
                        help="top-k future losses to average; H and 1 are always included")
    parser.add_argument("--n-calib-sequences", type=int, default=16,
                        help="sequences per calibration-seed repeat")
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--n-eval-sequences", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--nonmonotonic-tolerance", type=float, default=0.0)
    parser.add_argument("--rank-grid", type=int, nargs="+", default=list(DEFAULT_RANK_GRID))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "latest.pt")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "val.bin")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--allow-large-output", action="store_true",
                        help="allow >10 million stored per-offset delta floats")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing canonical v5 result pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_repeats < 2:
        raise ValueError("calibration_repeats must be >= 2")
    if args.calibration_batch_size <= 0 or args.probe_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.positions_per_sequence <= 0:
        raise ValueError("positions_per_sequence must be positive")
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap_draws must be >= 1000")

    canonical_outputs = (
        args.output_dir / "risk_capacity_v5_summary.json",
        args.output_dir / "risk_capacity_v5_records.json",
    )
    if any(path.exists() for path in canonical_outputs) and not args.overwrite:
        raise FileExistsError("canonical v5 output exists; use a new directory or pass --overwrite")

    device = choose_device(args.device)
    torch.manual_seed(CALIBRATION_SEED_BASE)
    np.random.seed(CALIBRATION_SEED_BASE)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CALIBRATION_SEED_BASE)
    print(f"device: {device}", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MLAGPT(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    block_size = int(config["max_len"])
    d_c = int(config["d_c"])
    rank_grid = normalize_rank_grid(args.rank_grid, d_c)
    tail_counts = normalize_tail_counts(args.tail_counts, args.future_horizon)
    valid_positions = valid_probe_positions(block_size, args.future_horizon)
    if len(valid_positions) == 0:
        raise ValueError("future horizon leaves no valid probe positions")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_possible_starts = len(data) - block_size
    # A sequence-sized gap makes the calibration and evaluation token spans exact
    # and disjoint, not merely the sampled start indices.
    calibration_high = int(n_possible_starts * 0.55)
    evaluation_low = calibration_high + block_size
    if evaluation_low >= n_possible_starts:
        raise ValueError("validation stream is too short for disjoint regions")

    calibration_starts = []
    calibration_used: list[int] = []
    sequence_span = block_size + 1
    for repeat_idx in range(args.calibration_repeats):
        rng = np.random.default_rng(CALIBRATION_SEED_BASE + repeat_idx)
        starts = sample_starts(
            rng, 0, calibration_high, args.n_calib_sequences,
            min_separation=sequence_span, excluded=calibration_used,
        )
        calibration_starts.append(starts)
        calibration_used.extend(map(int, starts))
    eval_rng = np.random.default_rng(EVALUATION_SEED)
    eval_starts = sample_starts(
        eval_rng, evaluation_low, n_possible_starts, args.n_eval_sequences,
        min_separation=sequence_span,
    )

    layer_orders, saliency_repeats = calibrate_layer_orders(
        model, data, calibration_starts,
        block_size=block_size,
        batch_size=args.calibration_batch_size,
        device=device,
    )
    expected_channels = np.arange(d_c)
    if any(not np.array_equal(np.sort(order), expected_channels) for order in layer_orders):
        raise AssertionError("each layer channel order must be a permutation of 0..d_c-1")

    eval_tokens = sequence_batch(data, eval_starts, block_size)
    baseline_loss = np.empty((args.n_eval_sequences, block_size), dtype=np.float32)
    with torch.no_grad():
        for offset in range(0, args.n_eval_sequences, args.probe_batch_size):
            tokens = eval_tokens[offset: offset + args.probe_batch_size]
            x = torch.from_numpy(tokens[:, :-1]).to(device)
            y = torch.from_numpy(tokens[:, 1:]).to(device)
            logits = forward_with_layer_masks(model, x)
            baseline_loss[offset: offset + len(tokens)] = per_token_loss(logits, y)
            del x, y, logits
    empty_device_cache(device)

    positions_by_sequence: list[np.ndarray] = []
    for _ in range(args.n_eval_sequences):
        count = min(args.positions_per_sequence, len(valid_positions))
        positions_by_sequence.append(
            np.sort(eval_rng.choice(valid_positions, size=count, replace=False))
        )

    raw_float_count = sum(len(positions) for positions in positions_by_sequence) * len(rank_grid) * args.future_horizon
    if raw_float_count > 10_000_000 and not args.allow_large_output:
        raise ValueError(
            f"requested run would store {raw_float_count:,} raw delta floats; "
            "pass --allow-large-output after checking disk/RAM capacity"
        )

    # Curves are indexed [tail_count][sequence][position][rank].
    tail_curves = {
        tail_count: [
            np.empty((len(positions), len(rank_grid)), dtype=np.float32)
            for positions in positions_by_sequence
        ]
        for tail_count in tail_counts
    }
    raw_delta_curves = [
        np.empty((len(positions), len(rank_grid), args.future_horizon), dtype=np.float32)
        for positions in positions_by_sequence
    ]

    for rank_idx, rank in enumerate(rank_grid):
        for seq_idx, positions in enumerate(positions_by_sequence):
            for offset in range(0, len(positions), args.probe_batch_size):
                pos_batch = positions[offset: offset + args.probe_batch_size]
                batch_count = len(pos_batch)
                tokens = eval_tokens[seq_idx: seq_idx + 1]
                x = torch.from_numpy(np.repeat(tokens[:, :-1], batch_count, axis=0)).to(device)
                y = torch.from_numpy(np.repeat(tokens[:, 1:], batch_count, axis=0)).to(device)
                pos_tensor = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                channel_masks = make_channel_masks(layer_orders, rank, batch_count, device)
                with torch.no_grad():
                    logits = forward_with_layer_masks(
                        model, x, channel_masks=channel_masks, probe_positions=pos_tensor
                    )
                    losses = per_token_loss(logits, y)
                for local_idx, pos in enumerate(pos_batch):
                    window = future_loss_slice(int(pos), args.future_horizon, losses.shape[1])
                    delta = losses[local_idx, window] - baseline_loss[seq_idx, window]
                    if len(delta) != args.future_horizon:
                        raise AssertionError("probe did not have the exact requested horizon")
                    raw_delta_curves[seq_idx][offset + local_idx, rank_idx] = delta
                    for tail_count in tail_counts:
                        tail_curves[tail_count][seq_idx][offset + local_idx, rank_idx] = upper_tail_mean(
                            delta, tail_count
                        )
                del x, y, pos_tensor, channel_masks, logits, losses
        empty_device_cache(device)
        print(f"rank={rank} complete", flush=True)

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    records = []
    rstars_by_tail: dict[int, dict[int, list[int]]] = {
        tail_count: {seq: [] for seq in range(args.n_eval_sequences)}
        for tail_count in tail_counts
    }
    for seq_idx, positions in enumerate(positions_by_sequence):
        for pos_idx, pos_value in enumerate(positions):
            pos = int(pos_value)
            r_star_map = {}
            delta_map = {}
            nonmonotonic_map = {}
            for tail_count in tail_counts:
                values = tail_curves[tail_count][seq_idx][pos_idx].astype(float).tolist()
                r_star = suffix_all_satisfy_r_star(values, rank_grid, args.epsilon)
                rstars_by_tail[tail_count][seq_idx].append(r_star)
                r_star_map[str(tail_count)] = r_star
                delta_map[str(tail_count)] = dict(zip(map(str, rank_grid), values))
                nonmonotonic_map[str(tail_count)] = is_nonmonotonic(
                    values, args.nonmonotonic_tolerance
                )
            ordered_ranks = [r_star_map[str(k)] for k in tail_counts]
            # tail_counts decrease as risk aversion rises, so capacity cannot decrease.
            if any(right < left for left, right in zip(ordered_ranks, ordered_ranks[1:])):
                raise AssertionError("risk-capacity monotonicity violated")
            input_token_id = int(eval_tokens[seq_idx, pos])
            records.append({
                "seq": seq_idx,
                "sequence_start": int(eval_starts[seq_idx]),
                "pos": pos,
                "input_token_id": input_token_id,
                "input_token_type": token_type(input_token_id, encoder),
                "future_horizon": args.future_horizon,
                "r_star_by_tail_count": r_star_map,
                "future_delta_by_rank": {
                    str(rank): raw_delta_curves[seq_idx][pos_idx, rank_idx].astype(float).tolist()
                    for rank_idx, rank in enumerate(rank_grid)
                },
                "tail_mean_delta_by_tail_count_and_rank": delta_map,
                "rank_curve_nonmonotonic_by_tail_count": nonmonotonic_map,
            })

    spectrum = {}
    for spectrum_idx, tail_count in enumerate(tail_counts):
        values = np.asarray([
            record["r_star_by_tail_count"][str(tail_count)] for record in records
        ])
        bootstrap_seed = (
            BOOTSTRAP_SEED if tail_count == args.future_horizon
            else BOOTSTRAP_SEED + 1 if tail_count == 1
            else BOOTSTRAP_SEED + 10 + spectrum_idx
        )
        ci = sequence_cluster_bootstrap_mean_ci(
            rstars_by_tail[tail_count], n_bootstrap=args.bootstrap_draws,
            seed=bootstrap_seed,
        )
        spectrum[str(tail_count)] = {
            "upper_tail_fraction": tail_count / args.future_horizon,
            "upper_tail_alpha": 1.0 - tail_count / args.future_horizon,
            "bootstrap_seed": bootstrap_seed,
            "mean_r_star": float(values.mean()),
            "normalized_mean_r_star": float(values.mean() / d_c),
            "sequence_cluster_bootstrap_95pct_ci": list(ci),
            "histogram": {str(rank): int((values == rank).sum()) for rank in rank_grid},
            "nonmonotonic_frequency": float(np.mean([
                record["rank_curve_nonmonotonic_by_tail_count"][str(tail_count)]
                for record in records
            ])),
        }
    premium_by_sequence = {
        seq: np.asarray(rstars_by_tail[1][seq]) - np.asarray(rstars_by_tail[args.future_horizon][seq])
        for seq in range(args.n_eval_sequences)
    }
    premium_values = np.concatenate(list(premium_by_sequence.values()))
    premium_ci = sequence_cluster_bootstrap_mean_ci(
        premium_by_sequence, n_bootstrap=args.bootstrap_draws,
        seed=BOOTSTRAP_SEED + 2,
    )

    # Horizon law and positive-part diagnostics from the saved per-offset deltas.
    horizon_lengths = sorted({1, 2, 4, 8, 16, args.future_horizon})
    horizon_lengths = [h for h in horizon_lengths if h <= args.future_horizon]
    horizon_spectrum = {}
    horizon_rstars: dict[str, dict[int, dict[int, list[int]]]] = {
        metric: {h: {seq: [] for seq in range(args.n_eval_sequences)} for h in horizon_lengths}
        for metric in ("signed_mean", "positive_mean", "maximum")
    }
    mean_safe_diagnostics = []
    for record in records:
        seq = int(record["seq"])
        for horizon in horizon_lengths:
            curves = {metric: [] for metric in horizon_rstars}
            for rank in rank_grid:
                delta = np.asarray(
                    record["future_delta_by_rank"][str(rank)][:horizon], dtype=np.float32
                )
                curves["signed_mean"].append(float(delta.mean()))
                curves["positive_mean"].append(float(np.maximum(delta, 0).mean()))
                curves["maximum"].append(float(delta.max()))
            for metric, values in curves.items():
                horizon_rstars[metric][horizon][seq].append(
                    suffix_all_satisfy_r_star(values, rank_grid, args.epsilon)
                )
        mean_safe_rank = int(record["r_star_by_tail_count"][str(args.future_horizon)])
        delta = np.asarray(
            record["future_delta_by_rank"][str(mean_safe_rank)], dtype=np.float32
        )
        positive = np.maximum(delta, 0)
        mean_safe_diagnostics.append({
            "seq": seq,
            "mean_safe_rank": mean_safe_rank,
            "signed_mean": float(delta.mean()),
            "positive_part_mean": float(positive.mean()),
            "maximum": float(delta.max()),
            "offsets_above_epsilon": int(np.sum(delta > args.epsilon)),
            "positive_offset_fraction": float(np.mean(delta > 0)),
            "cancellation_gap": float(positive.mean() - delta.mean()),
        })
    for metric, by_horizon in horizon_rstars.items():
        horizon_spectrum[metric] = {}
        for idx, horizon in enumerate(horizon_lengths):
            flat = np.concatenate([np.asarray(v) for v in by_horizon[horizon].values()])
            horizon_seed = (
                BOOTSTRAP_SEED
                if metric == "signed_mean" and horizon == args.future_horizon
                else BOOTSTRAP_SEED + 1
                if metric == "maximum" and horizon == args.future_horizon
                else BOOTSTRAP_SEED + 100 + idx + 10 * list(horizon_rstars).index(metric)
            )
            ci = sequence_cluster_bootstrap_mean_ci(
                by_horizon[horizon], n_bootstrap=args.bootstrap_draws,
                seed=horizon_seed,
            )
            horizon_spectrum[metric][str(horizon)] = {
                "mean_r_star": float(flat.mean()),
                "normalized_mean_r_star": float(flat.mean() / d_c),
                "sequence_cluster_bootstrap_95pct_ci": list(ci),
                "bootstrap_seed": horizon_seed,
            }
    for seq in range(args.n_eval_sequences):
        if horizon_rstars["signed_mean"][args.future_horizon][seq] != rstars_by_tail[args.future_horizon][seq]:
            raise AssertionError("horizon-H signed-mean endpoint differs from top-H endpoint")
        if horizon_rstars["maximum"][args.future_horizon][seq] != rstars_by_tail[1][seq]:
            raise AssertionError("horizon-H maximum endpoint differs from top-1 endpoint")
    diagnostics_arrays = {
        key: np.asarray([row[key] for row in mean_safe_diagnostics], dtype=np.float64)
        for key in ("signed_mean", "positive_part_mean", "maximum", "offsets_above_epsilon",
                    "positive_offset_fraction", "cancellation_gap")
    }
    mean_safe_summary = {
        key: float(values.mean()) for key, values in diagnostics_arrays.items()
    }
    mean_safe_summary["fraction_with_any_offset_above_epsilon"] = float(np.mean(
        diagnostics_arrays["offsets_above_epsilon"] > 0
    ))

    repeat_top32_overlap = []
    top_k = min(32, d_c)
    for layer_idx in range(model.n_layers):
        repeat_orders = np.argsort(-saliency_repeats[:, layer_idx], axis=1)
        overlaps = []
        for left in range(args.calibration_repeats):
            for right in range(left + 1, args.calibration_repeats):
                overlaps.append(
                    len(set(repeat_orders[left, :top_k]) & set(repeat_orders[right, :top_k]))
                    / top_k
                )
        repeat_top32_overlap.append(float(np.mean(overlaps)))

    summary = {
        "status": "valid",
        "method_version": "v5 upper-tail risk-capacity spectrum on corrected v4 windows",
        "method": "all-layer one-position truncation; top-k loss-delta means; conservative suffix r*",
        "scope_limitation": "full-attention truncation simulation, not cache-aware decoding",
        "device": str(device),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data_sha256": file_sha256(args.data),
        "rank_grid": rank_grid,
        "epsilon_nats": args.epsilon,
        "future_horizon_exact": args.future_horizon,
        "tail_counts": tail_counts,
        "n_layers": model.n_layers,
        "d_c": d_c,
        "n_calibration_sequences_per_repeat": args.n_calib_sequences,
        "calibration_repeats": args.calibration_repeats,
        "n_evaluation_sequences": args.n_eval_sequences,
        "positions_per_sequence": args.positions_per_sequence,
        "n_positions_total": len(records),
        "seeds": {
            "calibration": [CALIBRATION_SEED_BASE + i for i in range(args.calibration_repeats)],
            "evaluation": EVALUATION_SEED,
            "bootstrap_mean_endpoint": BOOTSTRAP_SEED,
            "bootstrap_max_endpoint": BOOTSTRAP_SEED + 1,
            "bootstrap_tail_premium": BOOTSTRAP_SEED + 2,
            "bootstrap_intermediate_tail_rule": "BOOTSTRAP_SEED + 10 + spectrum index",
            "bootstrap_horizon_rule": "BOOTSTRAP_SEED + 100 + horizon index + 10 * metric index",
        },
        "bootstrap_draws": args.bootstrap_draws,
        "raw_delta_float_count": raw_float_count,
        "calibration_repeat_mean_pairwise_top32_overlap_by_layer": repeat_top32_overlap,
        "layer_channel_orders": layer_orders.tolist(),
        "risk_capacity_spectrum": spectrum,
        "tail_capacity_premium": {
            "definition": "r_star(top-1=max) - r_star(top-H=mean)",
            "mean_rank_premium": float(premium_values.mean()),
            "normalized_mean_premium": float(premium_values.mean() / d_c),
            "sequence_cluster_bootstrap_95pct_ci": list(premium_ci),
            "nonnegative_record_fraction": float(np.mean(premium_values >= 0)),
            "strictly_positive_record_fraction": float(np.mean(premium_values > 0)),
        },
        "risk_capacity_monotonicity_theorem_check": {
            "status": "passed",
            "note": "guaranteed by ordered top-k loss-delta means plus suffix-safe thresholding",
        },
        "horizon_capacity_spectrum": horizon_spectrum,
        "mean_safe_rate_positive_part_diagnostics": mean_safe_summary,
        "mean_safe_rate_diagnostics_by_record": mean_safe_diagnostics,
        "token_attribution": "x_eval[b, pos] (the intervened source token)",
        "r_star_rule": "smallest grid rank for which this and every higher rank are <= epsilon",
        "risk_definition": "empirical upper-tail mean of the largest k truncation-induced loss deltas",
        "horizon_population": "paired prefixes of probes eligible for the maximum requested horizon",
        "analysis_source_sha256": file_sha256(Path(__file__)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "risk_capacity_v5_summary.json"
    records_path = args.output_dir / "risk_capacity_v5_records.json"
    records_tmp = args.output_dir / ".risk_capacity_v5_records.json.tmp"
    summary_tmp = args.output_dir / ".risk_capacity_v5_summary.json.tmp"
    try:
        with records_tmp.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
            handle.flush(); os.fsync(handle.fileno())
        summary["records_sha256"] = file_sha256(records_tmp)
        with summary_tmp.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.flush(); os.fsync(handle.fileno())
        # The summary is the completion marker. Remove any stale marker before
        # installing records, then atomically install the new summary last.
        summary_path.unlink(missing_ok=True)
        os.replace(records_tmp, records_path)
        os.replace(summary_tmp, summary_path)
    finally:
        records_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {summary_path} and {records_path}", flush=True)


if __name__ == "__main__":
    main()
