"""Pair-interaction residuals for the additivity approximation (theory prediction 5).

Section 4 of `notes/theory_contextual_tail_rate.md` writes the explanatory special
case in which the joint distortion decomposes per token,

    D(r, Z) = sum_t d_t(r_t, Z),

and notes that under it the Lagrangian reduces to a per-token marginal-value rule.
The method does not assume this, but the oracle labels do: capacity is measured by
truncating one position at a time, while the deployed router truncates every
position at once. Prediction 5 says pairwise intervention residuals should be
nonzero if that isolated-label supervision is structurally misaligned with joint
routing.

The test is the second-order mixed difference. Writing ``L_S`` for the per-token
loss vector when the positions in ``S`` are truncated,

    residual = L_{t1,t2} - L_{t1} - L_{t2} + L_{}

which is identically zero for any additive distortion, whatever the individual
effects are. It is measured over the whole sequence rather than a horizon window,
so the two positions' windows need not be aligned, and it is reported against the
size of the individual effects so that "nonzero" means "not negligible" rather
than "not exactly zero in floating point".

Separation between the two positions is swept, because additivity is expected to
hold trivially once the positions are far enough apart that neither is reused
while the other still matters; the interesting regime is nearby positions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from elastic_mla import MLAGPT  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"
)
_RISK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RISK)

PAIR_SEED = 55_117


def sample_pairs(rng, valid_positions, separation, count):
    """Pairs ``(t, t + separation)`` with both ends probeable."""
    low, high = int(valid_positions[0]), int(valid_positions[-1])
    first_max = high - separation
    if first_max < low:
        return np.empty((0, 2), dtype=np.int64)
    count = min(count, first_max - low + 1)
    firsts = rng.choice(np.arange(low, first_max + 1), size=count, replace=False)
    return np.stack([np.sort(firsts), np.sort(firsts) + separation], axis=1)


def loss_vectors(model, tokens, positions, masks, device):
    """Per-token loss for one sequence with ``positions`` truncated (empty = baseline)."""
    x = torch.from_numpy(tokens[:, :-1].copy()).to(device)
    y = torch.from_numpy(tokens[:, 1:].copy()).to(device)
    with torch.no_grad():
        if len(positions) == 0:
            logits = _RISK.forward_with_layer_masks(model, x)
        else:
            pos = torch.as_tensor(np.asarray(positions)[None, :], dtype=torch.long, device=device)
            logits = _RISK.forward_with_layer_masks(
                model, x, channel_masks=masks, probe_positions=pos
            )
        return _RISK.per_token_loss(logits, y)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    p.add_argument("--separations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--pairs-per-separation", type=int, default=6)
    p.add_argument("--ranks", type=int, nargs="+", default=None,
                   help="truncation ranks to test; defaults to a low/mid/high triple of the grid")
    p.add_argument("--n-calib-sequences", type=int, default=16)
    p.add_argument("--calibration-repeats", type=int, default=3)
    p.add_argument("--n-eval-sequences", type=int, default=8)
    p.add_argument("--calibration-batch-size", type=int, default=1)
    p.add_argument("--future-horizon", type=int, default=32)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(_RISK.CALIBRATION_SEED_BASE)
    np.random.seed(_RISK.CALIBRATION_SEED_BASE)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MLAGPT(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    block_size = int(checkpoint["config"]["max_len"])
    d_c = int(checkpoint["config"]["d_c"])
    grid = _RISK.normalize_rank_grid(list(_RISK.DEFAULT_RANK_GRID), d_c)
    ranks = args.ranks or [grid[0], grid[len(grid) // 2], grid[-2]]

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_possible_starts = len(data) - block_size
    calibration_high = int(n_possible_starts * 0.55)
    evaluation_low = calibration_high + block_size
    sequence_span = block_size + 1

    # Same sampling protocol as the published spectrum, so the channel order and
    # the evaluation sequences are the ones every other result was measured on.
    calibration_starts, used = [], []
    for repeat_idx in range(args.calibration_repeats):
        rng = np.random.default_rng(_RISK.CALIBRATION_SEED_BASE + repeat_idx)
        starts = _RISK.sample_starts(
            rng, 0, calibration_high, args.n_calib_sequences,
            min_separation=sequence_span, excluded=used,
        )
        calibration_starts.append(starts)
        used.extend(map(int, starts))
    eval_rng = np.random.default_rng(_RISK.EVALUATION_SEED)
    eval_starts = _RISK.sample_starts(
        eval_rng, evaluation_low, n_possible_starts, args.n_eval_sequences,
        min_separation=sequence_span,
    )

    layer_orders, _ = _RISK.calibrate_layer_orders(
        model, data, calibration_starts, block_size=block_size,
        batch_size=args.calibration_batch_size, device=device,
    )
    eval_tokens = _RISK.sequence_batch(data, eval_starts, block_size)
    valid_positions = _RISK.valid_probe_positions(block_size, args.future_horizon)

    pair_rng = np.random.default_rng(PAIR_SEED)
    records = []
    for rank in ranks:
        masks = _RISK.make_channel_masks(layer_orders, rank, 1, device)
        for seq_idx in range(args.n_eval_sequences):
            tokens = eval_tokens[seq_idx: seq_idx + 1]
            base = loss_vectors(model, tokens, [], masks, device)
            for separation in args.separations:
                pairs = sample_pairs(pair_rng, valid_positions, separation,
                                     args.pairs_per_separation)
                for t1, t2 in pairs:
                    d1 = loss_vectors(model, tokens, [t1], masks, device) - base
                    d2 = loss_vectors(model, tokens, [t2], masks, device) - base
                    d12 = loss_vectors(model, tokens, [t1, t2], masks, device) - base
                    residual = d12 - d1 - d2
                    individual = np.abs(d1) + np.abs(d2)
                    records.append({
                        "rank": int(rank),
                        "seq": int(seq_idx),
                        "separation": int(separation),
                        "t1": int(t1), "t2": int(t2),
                        "residual_l1": float(np.abs(residual).sum()),
                        "residual_linf": float(np.abs(residual).max()),
                        "individual_l1": float(individual.sum()),
                        "relative_l1": float(np.abs(residual).sum() / max(individual.sum(), 1e-12)),
                        "joint_l1": float(np.abs(d12).sum()),
                        "signed_residual_sum": float(residual.sum()),
                    })
            _RISK.empty_device_cache(device)
        print(f"rank={rank} complete", flush=True)

    def summarize(rows):
        rel = np.array([r["relative_l1"] for r in rows])
        return {
            "n_pairs": len(rows),
            "mean_relative_l1": float(rel.mean()),
            "median_relative_l1": float(np.median(rel)),
            "p90_relative_l1": float(np.percentile(rel, 90)),
            "mean_residual_l1": float(np.mean([r["residual_l1"] for r in rows])),
            "mean_individual_l1": float(np.mean([r["individual_l1"] for r in rows])),
            "mean_signed_residual_sum": float(np.mean([r["signed_residual_sum"] for r in rows])),
        }

    by_separation = {
        str(s): summarize([r for r in records if r["separation"] == s])
        for s in args.separations if any(r["separation"] == s for r in records)
    }
    by_rank = {
        str(r_): summarize([r for r in records if r["rank"] == r_])
        for r_ in ranks if any(r["rank"] == r_ for r in records)
    }

    output = {
        "status": "complete",
        "prediction": ("5: pairwise intervention residuals should be nonzero if isolated-label "
                       "supervision is structurally misaligned with joint routing"),
        "residual_definition": "L_{t1,t2} - L_{t1} - L_{t2} + L_{} per token, summed as L1 "
                               "over the whole sequence; identically zero under additivity",
        "device": str(device),
        "checkpoint_sha256": _RISK.file_sha256(args.checkpoint),
        "data_sha256": _RISK.file_sha256(args.data),
        "analysis_source_sha256": _RISK.file_sha256(Path(__file__)),
        "d_c": d_c,
        "n_layers": int(checkpoint["config"]["n_layers"]),
        "ranks": [int(r) for r in ranks],
        "separations": args.separations,
        "n_evaluation_sequences": args.n_eval_sequences,
        "pairs_per_separation": args.pairs_per_separation,
        "n_pairs_total": len(records),
        "summary_by_separation": by_separation,
        "summary_by_rank": by_rank,
        "overall": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("PAIR_JSON_BEGIN")
    print(json.dumps({k: v for k, v in output.items() if k != "records"}))
    print("PAIR_JSON_END")


if __name__ == "__main__":
    main()
