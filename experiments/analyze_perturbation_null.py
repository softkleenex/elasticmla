"""Equal-norm random-perturbation null for the risk-capacity spectrum (analysis R2).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R2).

Question: is the worst-offset rate requirement specific to removing the low-saliency
latent channels, or would any perturbation of the same size give the same spectrum?

For every probe (sequence, source position) and every grid rank r, two interventions
are applied to the source position's latent in every layer at once:

* truncation (the paper's probe): c -> c * m_r, i.e. remove the residual
  e_r = c * (1 - m_r) of the channels beyond the saliency prefix of length r;
* null: c -> c + ||e_r|| * u, where u is a fixed isotropic Gaussian unit direction
  per (probe, layer), drawn once from seed 60521 + probe index. The direction does not
  change across ranks, so the null's rate axis is the truncation norm schedule.

Both conditions use the published run's channel orders (read from its summary, not
recalibrated) and the published evaluation windows and positions (regenerated from the
fixed evaluation seed and checked against the published records when given). The
spectrum, the tail-capacity premium and the mean-safe cancellation diagnostic are then
computed identically for both. The pre-registered reading compares the mean normalized
r* at k = 1 and at k = H: within 0.10 absolute -> the spectrum reflects sensitivity to
perturbations of that norm; null larger -> truncation is gentler than equal-norm noise
(the saliency order carries information); null smaller -> reported as is.

Full-attention simulation, like the spectrum itself; not compressed-cache decoding.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NULL_SEED_BASE = 60_521
PAIRED_BOOTSTRAP_SEED = 61_000
WITHIN_TOLERANCE = 0.10


def load_risk_module():
    spec = importlib.util.spec_from_file_location(
        "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


risk = load_risk_module()


def null_directions(probe_index: int, n_layers: int, d_c: int) -> np.ndarray:
    """One isotropic unit direction per layer for this probe (fixed across ranks)."""
    rng = np.random.default_rng(NULL_SEED_BASE + probe_index)
    draws = rng.standard_normal((n_layers, d_c))
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def forward_with_equal_norm_noise(
    model, idx: torch.Tensor, *, channel_masks: torch.Tensor,
    directions: torch.Tensor, probe_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward that adds ||c * (1 - mask)|| * direction to each probe latent.

    ``channel_masks`` and ``directions`` are (n_layers, batch, d_c); ``probe_positions``
    is (batch,). Returns logits and the (n_layers, batch) perturbation norms applied.
    """
    batch_index = torch.arange(idx.shape[0], device=idx.device)
    norms = torch.zeros(model.n_layers, idx.shape[0], device=idx.device)
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            output = output.clone()
            rows = output[batch_index, probe_positions]
            residual = rows * (1.0 - channel_masks[layer].to(rows.dtype))
            norm = residual.norm(dim=-1, keepdim=True)
            norms[layer] = norm.squeeze(-1).float()
            output[batch_index, probe_positions] = rows + directions[layer].to(rows.dtype) * norm
            return output
        return hook

    for layer, block in enumerate(model.blocks):
        handles.append(block.attn.W_DKV.register_forward_hook(make_hook(layer)))
    try:
        logits = risk.forward_with_layer_masks(model, idx)
    finally:
        for handle in handles:
            handle.remove()
    return logits, norms


def evaluation_windows(data, block_size: int, horizon: int, n_sequences: int,
                       positions_per_sequence: int):
    """Regenerate the published evaluation starts and probe positions exactly."""
    n_possible_starts = len(data) - block_size
    evaluation_low = int(n_possible_starts * 0.55) + block_size
    eval_rng = np.random.default_rng(risk.EVALUATION_SEED)
    starts = risk.sample_starts(eval_rng, evaluation_low, n_possible_starts, n_sequences,
                                min_separation=block_size + 1)
    valid = risk.valid_probe_positions(block_size, horizon)
    positions = [np.sort(eval_rng.choice(valid, size=min(positions_per_sequence, len(valid)),
                                         replace=False)) for _ in range(n_sequences)]
    return starts, positions


def r_star_table(deltas: np.ndarray, ranks, tail_counts, epsilon: float) -> np.ndarray:
    """deltas: (probes, ranks, offsets) -> r* (probes, tail counts)."""
    out = np.empty((deltas.shape[0], len(tail_counts)), dtype=np.int64)
    for i, probe in enumerate(deltas):
        for j, k in enumerate(tail_counts):
            curve = [risk.upper_tail_mean(row, k) for row in probe]
            out[i, j] = risk.suffix_all_satisfy_r_star(curve, ranks, epsilon)
    return out


def by_sequence(values: np.ndarray, sequence_ids: np.ndarray) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for seq, value in zip(sequence_ids, values):
        grouped.setdefault(int(seq), []).append(float(value))
    return grouped


def cancellation(deltas: np.ndarray, r_star_mean: np.ndarray, ranks, epsilon: float) -> dict:
    """Mean-safe-rate diagnostics, as in the v5 summary."""
    index = {rank: i for i, rank in enumerate(ranks)}
    rows = np.stack([deltas[i, index[int(r)]] for i, r in enumerate(r_star_mean)])
    positive = np.maximum(rows, 0)
    return {
        "signed_mean": float(rows.mean()),
        "positive_part_mean": float(positive.mean()),
        "cancellation_gap": float((positive.mean(axis=1) - rows.mean(axis=1)).mean()),
        "maximum": float(rows.max(axis=1).mean()),
        "fraction_with_any_offset_above_epsilon": float(np.mean((rows > epsilon).any(axis=1))),
    }


def summarize(trunc: np.ndarray, null: np.ndarray, sequence_ids: np.ndarray, ranks,
              tail_counts, d_c: int, epsilon: float, bootstrap_draws: int) -> dict:
    r_trunc = r_star_table(trunc, ranks, tail_counts, epsilon) / d_c
    r_null = r_star_table(null, ranks, tail_counts, epsilon) / d_c
    spectrum = {}
    for j, k in enumerate(tail_counts):
        diff = r_null[:, j] - r_trunc[:, j]
        entry = {}
        for name, values, seed in (("truncation", r_trunc[:, j], 0), ("null", r_null[:, j], 1),
                                   ("null_minus_truncation", diff, 2)):
            ci = risk.sequence_cluster_bootstrap_mean_ci(
                by_sequence(values, sequence_ids), n_bootstrap=bootstrap_draws,
                seed=PAIRED_BOOTSTRAP_SEED + 10 * j + seed)
            entry[name] = {"normalized_mean_r_star": float(values.mean()),
                           "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}
        spectrum[str(k)] = entry

    def reading(k: int) -> str:
        gap = spectrum[str(k)]["null_minus_truncation"]["normalized_mean_r_star"]
        if abs(gap) <= WITHIN_TOLERANCE:
            return "within_0.10: spectrum reflects sensitivity to perturbations of that norm"
        return ("null_requires_more_rate: truncation gentler than equal-norm noise" if gap > 0
                else "null_requires_less_rate: reported as is")

    h, top = str(tail_counts[0]), str(tail_counts[-1])
    mean_col, max_col = 0, len(tail_counts) - 1
    return {
        "spectrum": spectrum,
        "tail_capacity_premium": {
            name: float(spectrum[top][name]["normalized_mean_r_star"]
                        - spectrum[h][name]["normalized_mean_r_star"])
            for name in ("truncation", "null")},
        "mean_safe_cancellation": {
            "truncation": cancellation(trunc, (r_trunc[:, mean_col] * d_c).round(), ranks, epsilon),
            "null": cancellation(null, (r_null[:, mean_col] * d_c).round(), ranks, epsilon)},
        "pre_registered_reading": {f"k={tail_counts[0]} (mean)": reading(tail_counts[0]),
                                   f"k={tail_counts[-1]} (max)": reading(tail_counts[-1])},
        "per_probe_r_star_max_column": {"truncation": (r_trunc[:, max_col] * d_c).astype(int).tolist(),
                                        "null": (r_null[:, max_col] * d_c).astype(int).tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True,
                        help="published risk_capacity_v5_summary.json (orders, grid, hashes)")
    parser.add_argument("--reference-records", type=Path, default=None,
                        help="published records; when given, windows and positions must match")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--limit-sequences", type=int, default=None,
                        help="smoke test only: evaluate the first N sequences")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_summary.read_text())
    hashes = {"checkpoint": risk.file_sha256(args.checkpoint), "data": risk.file_sha256(args.data)}
    if hashes["checkpoint"] != reference["checkpoint_sha256"] or hashes["data"] != reference["data_sha256"]:
        raise SystemExit(f"input hashes do not match the reference summary: {hashes}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = risk.MLAGPT(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    block_size, d_c, n_layers = int(config["max_len"]), int(config["d_c"]), model.n_layers
    ranks = [int(r) for r in reference["rank_grid"]]
    tail_counts = [int(k) for k in reference["tail_counts"]]
    horizon = int(reference["future_horizon_exact"])
    epsilon = float(reference["epsilon_nats"])
    layer_orders = np.asarray(reference["layer_channel_orders"], dtype=np.int64)
    if layer_orders.shape != (n_layers, d_c):
        raise SystemExit("reference channel orders do not fit this checkpoint")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    starts, positions = evaluation_windows(
        data, block_size, horizon, int(reference["n_evaluation_sequences"]),
        int(reference["positions_per_sequence"]))
    if args.limit_sequences:
        starts, positions = starts[:args.limit_sequences], positions[:args.limit_sequences]
    window_check = "not checked (no reference records)"
    if args.reference_records is not None:
        records = json.loads(args.reference_records.read_text())
        expected = [(int(r["sequence_start"]), int(r["pos"])) for r in records]
        ours = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
        if ours != expected[:len(ours)]:
            raise SystemExit("regenerated windows/positions differ from the reference records")
        window_check = "identical to reference records"

    tokens = risk.sequence_batch(data, starts, block_size)
    n_probes = sum(len(p) for p in positions)
    trunc = np.empty((n_probes, len(ranks), horizon), dtype=np.float32)
    null = np.empty_like(trunc)
    norm_by_rank = np.zeros((len(ranks), n_layers), dtype=np.float64)
    sequence_ids = np.concatenate([np.full(len(p), s) for s, p in enumerate(positions)])
    probe_offsets = np.cumsum([0] + [len(p) for p in positions])
    directions = np.stack([null_directions(i, n_layers, d_c) for i in range(n_probes)])

    with torch.no_grad():
        baseline = np.empty((len(starts), block_size), dtype=np.float32)
        for s in range(len(starts)):
            x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
            y = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
            baseline[s] = risk.per_token_loss(risk.forward_with_layer_masks(model, x), y)[0]
        for rank_idx, rank in enumerate(ranks):
            for s, pos_all in enumerate(positions):
                for offset in range(0, len(pos_all), args.probe_batch_size):
                    pos_batch = pos_all[offset:offset + args.probe_batch_size]
                    count = len(pos_batch)
                    first = probe_offsets[s] + offset
                    x = torch.from_numpy(np.repeat(tokens[s:s + 1, :-1], count, axis=0)).to(device)
                    y = torch.from_numpy(np.repeat(tokens[s:s + 1, 1:], count, axis=0)).to(device)
                    pos_t = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                    masks = risk.make_channel_masks(layer_orders, rank, count, device)
                    dirs = torch.as_tensor(directions[first:first + count].transpose(1, 0, 2).copy(),
                                           dtype=torch.float32, device=device)
                    loss_t = risk.per_token_loss(risk.forward_with_layer_masks(
                        model, x, channel_masks=masks, probe_positions=pos_t), y)
                    logits_n, norms = forward_with_equal_norm_noise(
                        model, x, channel_masks=masks, directions=dirs, probe_positions=pos_t)
                    loss_n = risk.per_token_loss(logits_n, y)
                    norm_by_rank[rank_idx] += norms.sum(dim=1).cpu().double().numpy()
                    for local, pos in enumerate(pos_batch):
                        window = risk.future_loss_slice(int(pos), horizon, block_size)
                        trunc[first + local, rank_idx] = loss_t[local, window] - baseline[s, window]
                        null[first + local, rank_idx] = loss_n[local, window] - baseline[s, window]
            risk.empty_device_cache(device)
            print(f"rank={rank} complete", flush=True)

    result = summarize(trunc, null, sequence_ids, ranks, tail_counts, d_c, epsilon, args.bootstrap_draws)
    published = {k: v["normalized_mean_r_star"] for k, v in reference["risk_capacity_spectrum"].items()}
    reproduction = {k: {"published": published[k],
                        "this_run_truncation": result["spectrum"][k]["truncation"]["normalized_mean_r_star"]}
                    for k in map(str, tail_counts)}
    if args.reference_records is not None:
        stored = {(int(r["sequence_start"]), int(r["pos"])): int(r["r_star_by_tail_count"]["1"])
                  for r in records}
        ours = [stored[(int(starts[s]), int(p))] for s in range(len(starts)) for p in positions[s]]
        agree = np.mean(np.asarray(ours) == np.asarray(result["per_probe_r_star_max_column"]["truncation"]))
        reproduction["per_probe_max_r_star_agreement_with_records"] = float(agree)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw_path, truncation=trunc, null=null, rank_grid=np.asarray(ranks),
                        sequence_ids=sequence_ids, sequence_starts=np.asarray(starts),
                        positions=np.concatenate(positions))
    output = {
        "status": "smoke" if args.limit_sequences else "complete",
        "analysis": "R2 equal-norm random-perturbation null (pre-registered)",
        "device": str(device),
        "inputs_sha256": {**hashes, "reference_summary": risk.file_sha256(args.reference_summary),
                          **({"reference_records": risk.file_sha256(args.reference_records)}
                             if args.reference_records else {}),
                          "analysis_source": risk.file_sha256(Path(__file__)),
                          "risk_v5_source": risk.file_sha256(ROOT / "experiments/analyze_risk_capacity_spectrum.py")},
        "raw_npz_sha256": risk.file_sha256(raw_path),
        "null_construction": "c + ||c*(1-m_r)|| * u, u isotropic unit per (probe, layer), "
                             f"seed {NULL_SEED_BASE} + probe index, fixed across ranks; all layers at once",
        "channel_orders": "reference summary layer_channel_orders (not recalibrated)",
        "window_check": window_check,
        "rank_grid": ranks, "tail_counts": tail_counts, "epsilon_nats": epsilon,
        "future_horizon_exact": horizon, "d_c": d_c, "n_layers": n_layers, "n_probes": n_probes,
        "mean_perturbation_norm_by_rank": {str(r): float(norm_by_rank[i].sum() / (n_probes * n_layers))
                                           for i, r in enumerate(ranks)},
        "truncation_reproduction": reproduction,
        **result,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for k in map(str, tail_counts):
        e = output["spectrum"][k]
        print(f"k={k}: trunc={e['truncation']['normalized_mean_r_star']:.3f} "
              f"null={e['null']['normalized_mean_r_star']:.3f} "
              f"diff={e['null_minus_truncation']['normalized_mean_r_star']:+.3f} "
              f"CI={e['null_minus_truncation']['sequence_cluster_bootstrap_95pct_ci']}")
    print(json.dumps(output["pre_registered_reading"], indent=2))
    print(json.dumps(reproduction, indent=2))


if __name__ == "__main__":
    main()
