"""Low-rank grid extension and rate-matched joint allocation (pre-registered analysis R10b).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R10b, committed in d154fa4
before this script was run).

1. Low ranks. The published grid starts at rank 16, and 81-87% of probes have their
   mean-safe r* there. Every probe is truncated (all layers, probe position only) at
   r in {0, 1, 2, 4, 8}; rank 16 is rerun as a reproduction check. A stored r*_k > 16
   cannot change; a stored r*_k = 16 becomes the smallest r in {0, 1, 2, 4, 8, 16} from
   which every rank up to 16 satisfies epsilon.
2. Rate-matched joint allocation (needs per-probe records). R5 truncated all probe
   positions of a sequence jointly at their own r*. Here the same total rate is placed
   (a) uniformly -- every probe position gets the sequence's mean rank, rounded -- and
   (b) shuffled -- the sequence's own ranks permuted across its probe positions.
3. Quantization flips. Fraction of positions whose greedy token changes under int8/int4.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOW_RANKS = (0, 1, 2, 4, 8)
CHECK_RANK = 16
MIN_REPRODUCTION_AGREEMENT = 0.99
SHUFFLE_SEED_BASE = 73_521
MATCH_TOLERANCE = 0.05


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quant = load_module("latent_quantization", "analyze_latent_quantization.py")
null_module = quant.null_module
risk = quant.risk


def extend_r_star(stored: int, low_curve: np.ndarray, low_ranks, epsilon: float) -> int:
    """Stored suffix-rule r* on a grid starting at CHECK_RANK, extended downwards.

    ``low_curve`` holds the tail values at ``low_ranks`` followed by CHECK_RANK.
    """
    if stored > CHECK_RANK:
        return stored
    ranks = [*low_ranks, CHECK_RANK]
    return int(risk.suffix_all_satisfy_r_star(low_curve, ranks, epsilon))


def shuffled_ranks(ranks: dict[int, int], seed: int) -> dict[int, int]:
    positions = sorted(ranks)
    values = np.array([ranks[p] for p in positions])
    permuted = np.random.default_rng(seed).permutation(values)
    return {p: int(v) for p, v in zip(positions, permuted)}


def uniform_ranks(ranks: dict[int, int]) -> dict[int, int]:
    level = int(round(float(np.mean(list(ranks.values())))))
    return {p: level for p in ranks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, default=None,
                        help="published per-probe records; enables all k and parts 2-3")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_summary.read_text())
    hashes = {"checkpoint": risk.file_sha256(args.checkpoint), "data": risk.file_sha256(args.data)}
    if hashes["checkpoint"] != reference["checkpoint_sha256"] or hashes["data"] != reference["data_sha256"]:
        raise SystemExit(f"input hashes do not match the reference summary: {hashes}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = risk.MLAGPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    block_size, d_c = int(checkpoint["config"]["max_len"]), int(checkpoint["config"]["d_c"])
    horizon = int(reference["future_horizon_exact"])
    epsilon = float(reference["epsilon_nats"])
    layer_orders = np.asarray(reference["layer_channel_orders"], dtype=np.int64)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    starts, positions = null_module.evaluation_windows(
        data, block_size, horizon, int(reference["n_evaluation_sequences"]),
        int(reference["positions_per_sequence"]))
    tokens = risk.sequence_batch(data, starts, block_size)
    keys = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
    sequence_ids = np.repeat(np.arange(len(starts)), [len(p) for p in positions])

    records = None
    if args.reference_records is not None:
        records = json.loads(args.reference_records.read_text())
        if [(int(r["sequence_start"]), int(r["pos"])) for r in records] != keys:
            raise SystemExit("regenerated windows/positions differ from the reference records")
        tail_counts = [int(k) for k in reference["tail_counts"]]
        stored_r = {k: np.array([int(r["r_star_by_tail_count"][str(k)]) for r in records]) for k in tail_counts}
        stored_check = np.array([r["future_delta_by_rank"][str(CHECK_RANK)] for r in records], dtype=np.float64)
    else:
        diagnostics = reference["mean_safe_rate_diagnostics_by_record"]
        if len(diagnostics) != len(keys) or any(int(d["seq"]) != s for d, s in zip(diagnostics, sequence_ids)):
            raise SystemExit("summary diagnostics do not line up with the regenerated probes")
        tail_counts = [horizon]
        stored_r = {horizon: np.array([int(d["mean_safe_rank"]) for d in diagnostics])}
        stored_check = None

    baseline = np.empty((len(starts), block_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(len(starts)):
            x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
            y = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
            baseline[s] = risk.per_token_loss(risk.forward_with_layer_masks(model, x), y)[0]

    # Part 1: deltas at the low ranks and the check rank.
    grid = [*LOW_RANKS, CHECK_RANK]
    deltas = np.empty((len(keys), len(grid), horizon), dtype=np.float32)
    probe = 0
    with torch.no_grad():
        for s in range(len(starts)):
            for offset in range(0, len(positions[s]), args.probe_batch_size):
                pos_batch = positions[s][offset:offset + args.probe_batch_size]
                count = len(pos_batch)
                x = torch.from_numpy(np.repeat(tokens[s:s + 1, :-1], count, axis=0)).to(device)
                y = torch.from_numpy(np.repeat(tokens[s:s + 1, 1:], count, axis=0)).to(device)
                pos_t = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                for g, rank in enumerate(grid):
                    masks = risk.make_channel_masks(layer_orders, rank, count, device)
                    losses = risk.per_token_loss(risk.forward_with_layer_masks(
                        model, x, channel_masks=masks, probe_positions=pos_t), y)
                    for local, pos in enumerate(pos_batch):
                        window = risk.future_loss_slice(int(pos), horizon, losses.shape[1])
                        deltas[probe + local, g] = losses[local, window] - baseline[s, window]
                probe += count
            print(f"low ranks: sequence {s + 1}/{len(starts)}", flush=True)

    agreement = {}
    if stored_check is not None:
        for k in tail_counts:
            ours = np.array([risk.upper_tail_mean(row, k) for row in deltas[:, -1]]) <= epsilon
            theirs = np.array([risk.upper_tail_mean(row, k) for row in stored_check]) <= epsilon
            agreement[str(k)] = float(np.mean(ours == theirs))
        check_max_abs_diff = float(np.abs(deltas[:, -1] - stored_check).max())
    else:
        # The summary stores the rank-16 signed mean only for probes whose r* is 16.
        floor = stored_r[horizon] == CHECK_RANK
        stored_mean = np.array([float(d["signed_mean"]) for d in reference["mean_safe_rate_diagnostics_by_record"]])
        ours_mean = deltas[:, -1].mean(axis=1)
        agreement[str(horizon)] = float(np.mean(ours_mean[floor] <= epsilon))
        check_max_abs_diff = float(np.abs(ours_mean[floor] - stored_mean[floor]).max())
    reproduced = all(v >= MIN_REPRODUCTION_AGREEMENT for v in agreement.values())

    extended = {}
    for j, k in enumerate(tail_counts):
        new_r = np.array([extend_r_star(int(stored_r[k][i]),
                                        np.array([risk.upper_tail_mean(row, k) for row in deltas[i]]),
                                        LOW_RANKS, epsilon) for i in range(len(keys))])
        groups = null_module.by_sequence(new_r / d_c, sequence_ids)
        ci = risk.sequence_cluster_bootstrap_mean_ci(groups, n_bootstrap=args.bootstrap_draws, seed=74_000 + j)
        extended[str(k)] = {"normalized_mean_r_star": float(np.mean(new_r) / d_c),
                            "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci],
                            "published_grid_normalized_mean_r_star": float(np.mean(stored_r[k]) / d_c),
                            "fraction_at_rank_0": float(np.mean(new_r == 0)),
                            "fraction_changed": float(np.mean(new_r != stored_r[k])),
                            "rank_histogram": {str(r): int(np.sum(new_r == r)) for r in sorted(set(new_r.tolist()))}}
    ratio = (extended["1"]["normalized_mean_r_star"] / extended[str(horizon)]["normalized_mean_r_star"]
             if "1" in extended and extended[str(horizon)]["normalized_mean_r_star"] > 0 else None)

    # Parts 2 and 3 need the per-probe records (250M).
    joint = flips = None
    if records is not None:
        by_seq = {}
        for i, rec in enumerate(records):
            by_seq.setdefault(int(rec["seq"]), {})[int(rec["pos"])] = i

        def run_allocation(rank_map: dict[int, int], s: int):
            masks = quant.joint_rank_masks(layer_orders, rank_map, block_size, device)
            x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
            y = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
            with torch.no_grad():
                delta = risk.per_token_loss(quant.forward_with_full_masks(model, x, masks), y)[0] - baseline[s]
            tail = delta[min(rank_map):]
            return tail.mean(), np.mean(tail > epsilon), tail.max()

        def clustered(values: np.ndarray, seed: int) -> dict:
            ci = risk.sequence_cluster_bootstrap_mean_ci({i: [float(v)] for i, v in enumerate(values)},
                                                         n_bootstrap=args.bootstrap_draws, seed=seed)
            return {"mean": float(values.mean()), "sequence_bootstrap_95pct_ci": [float(c) for c in ci]}

        joint = {}
        for a, (label, k) in enumerate((("mean_safe", horizon), ("max_safe", 1))):
            per = {name: {"mean": [], "frac": [], "max": [], "rate": []} for name in ("own", "uniform", "shuffled")}
            for s in range(len(starts)):
                own = {p: int(stored_r[k][i]) for p, i in by_seq[s].items()}
                for name, rank_map in (("own", own), ("uniform", uniform_ranks(own)),
                                       ("shuffled", shuffled_ranks(own, SHUFFLE_SEED_BASE + s))):
                    m, f, mx = run_allocation(rank_map, s)
                    per[name]["mean"].append(m)
                    per[name]["frac"].append(f)
                    per[name]["max"].append(mx)
                    per[name]["rate"].append(np.mean(list(rank_map.values())) / d_c)
            joint[label] = {}
            for b, (name, vals) in enumerate(per.items()):
                seed = 75_000 + 100 * a + 10 * b
                joint[label][name] = {
                    "mean_probe_rate_fraction_of_d_c": float(np.mean(vals["rate"])),
                    "sequence_mean_loss_change_nats": clustered(np.asarray(vals["mean"]), seed),
                    "fraction_of_positions_above_epsilon": clustered(np.asarray(vals["frac"]), seed + 1),
                    "max_position_loss_change_nats": clustered(np.asarray(vals["max"]), seed + 2),
                }
            for b, ref in enumerate(("uniform", "shuffled")):
                diff = np.asarray(per["own"]["frac"]) - np.asarray(per[ref]["frac"])
                summary = clustered(diff, 76_000 + 10 * a + b)
                gap = summary["mean"]
                summary["pre_registered_reading"] = (
                    f"within {MATCH_TOLERANCE:.2f} of {ref}: fraction reflects total rate, not the criterion"
                    if abs(gap) <= MATCH_TOLERANCE else
                    f"lower than {ref} by more than {MATCH_TOLERANCE:.2f}: criterion places rate better"
                    if gap < 0 else f"higher than {ref} by more than {MATCH_TOLERANCE:.2f}: worse than {ref}")
                joint[label][f"own_minus_{ref}_fraction_above_epsilon"] = summary
            print(label, json.dumps(joint[label]), flush=True)

        flips = {}
        with torch.no_grad():
            for name, qmax in quant.QUANT_LEVELS.items():
                changed = []
                for s in range(len(starts)):
                    x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
                    full = risk.forward_with_layer_masks(model, x).argmax(-1)
                    quantized = quant.forward_quantized(model, x, qmax).argmax(-1)
                    changed.append(float((full != quantized).float().mean()))
                flips[name] = {"latent_bytes_fraction_of_fp16": (0.5 if name == "int8" else 0.25) + 1.0 / d_c,
                               "greedy_flip_rate_over_positions": clustered(np.asarray(changed), 77_000 + len(flips))}
            print("flips", flips, flush=True)

    output = {
        "status": "complete" if reproduced else "failed_reproduction_check",
        "analysis": "R10b low-rank grid extension, rate-matched joint allocation, quantization flips (pre-registered)",
        "device": str(device),
        "inputs_sha256": {**hashes, "reference_summary": risk.file_sha256(args.reference_summary),
                          **({"reference_records": risk.file_sha256(args.reference_records)}
                             if args.reference_records else {}),
                          "analysis_source": risk.file_sha256(Path(__file__)),
                          "latent_quantization_source": risk.file_sha256(ROOT / "experiments/analyze_latent_quantization.py")},
        "epsilon_nats": epsilon, "d_c": d_c, "future_horizon_exact": horizon, "n_probes": len(keys),
        "low_rank_grid": list(LOW_RANKS), "check_rank": CHECK_RANK,
        "reproduction_agreement_by_tail_count": agreement,
        "check_rank_max_abs_difference": check_max_abs_diff,
        "min_reproduction_agreement": MIN_REPRODUCTION_AGREEMENT,
        "extended_spectrum": extended, "extended_max_over_mean_ratio": ratio,
        "rate_matched_joint_allocation": joint, "quantization_greedy_flips": flips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw, deltas=deltas, grid=np.asarray(grid), sequence_ids=sequence_ids)
    output["raw_npz_sha256"] = risk.file_sha256(raw)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(output["status"], agreement, {k: v["normalized_mean_r_star"] for k, v in extended.items()}, ratio)


if __name__ == "__main__":
    main()
