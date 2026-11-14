"""Equal-norm random-perturbation null for the risk-capacity spectrum (analysis R2).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R2 and its
implementation details, including the post-review corrections).

Question: is the worst-offset rate requirement specific to removing the low-saliency
latent channels, or would any perturbation of the same size give the same spectrum?

For every probe (sequence, source position) and every grid rank r:

* truncation (the paper's probe): c -> c * m_r in every layer at once. A read-only
  hook records, per layer, the norm ||e_r|| of the residual e_r = c * (1 - m_r) that
  truncation removes, on the truncation run's own stream;
* null (pre-registered): c -> c + ||e_r|| * u in every layer at once, with exactly
  those recorded norms. u is an isotropic Gaussian unit direction per (probe, layer),
  drawn once from seed 60521 + probe index and fixed across ranks;
* covariance null (exploratory, optional): the same with u drawn from N(0, Sigma_l),
  Sigma_l the empirical latent covariance of layer l on the published calibration
  windows, seed 62521 + probe index. It answers the objection that isotropic noise
  mostly lands in directions the up-projections ignore. It is not pre-registered.

Channel orders come from the published summary (not recalibrated). Windows and
positions are regenerated from the evaluation seed and checked against the published
records (250M) or, when only a summary exists (30M), against its per-probe mean-safe
ranks. The run stops if the recomputed truncation r* agrees with the published values
on fewer than 99% of probes. The reading compares the mean normalized r* at k = 1 and
k = H: within 0.10 absolute -> the spectrum reflects sensitivity to perturbations of
that norm; null larger -> truncation is gentler than equal-norm noise; null smaller ->
reported as is.

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
COVARIANCE_SEED_BASE = 62_521
PAIRED_BOOTSTRAP_SEED = 61_000
WITHIN_TOLERANCE = 0.10
MIN_REPRODUCTION_AGREEMENT = 0.99


def load_risk_module():
    spec = importlib.util.spec_from_file_location(
        "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


risk = load_risk_module()


def null_directions(probe_index: int, n_layers: int, d_c: int,
                    sqrt_covariances: np.ndarray | None = None) -> np.ndarray:
    """One unit direction per layer for this probe (fixed across ranks).

    Isotropic by default; with ``sqrt_covariances`` (n_layers, d_c, d_c) the Gaussian
    draw is shaped by each layer's latent covariance before normalization.
    """
    base = NULL_SEED_BASE if sqrt_covariances is None else COVARIANCE_SEED_BASE
    draws = np.random.default_rng(base + probe_index).standard_normal((n_layers, d_c))
    if sqrt_covariances is not None:
        draws = np.einsum("lij,lj->li", sqrt_covariances, draws)
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def _with_latent_hooks(model, make_hook, run):
    handles = [block.attn.W_DKV.register_forward_hook(make_hook(layer))
               for layer, block in enumerate(model.blocks)]
    try:
        return run()
    finally:
        for handle in handles:
            handle.remove()


def forward_truncation_with_norms(model, idx, *, channel_masks, probe_positions):
    """The paper's truncation forward, also returning ||c * (1 - m)|| per (layer, item)."""
    batch_index = torch.arange(idx.shape[0], device=idx.device)
    norms = torch.zeros(model.n_layers, idx.shape[0], device=idx.device)

    def make_hook(layer):
        def hook(_module, _inputs, output):
            rows = output[batch_index, probe_positions]
            norms[layer] = (rows * (1.0 - channel_masks[layer].to(rows.dtype))).norm(dim=-1).float()
        return hook

    logits = _with_latent_hooks(model, make_hook, lambda: risk.forward_with_layer_masks(
        model, idx, channel_masks=channel_masks, probe_positions=probe_positions))
    return logits, norms


def forward_with_added_noise(model, idx, *, norms, directions, probe_positions):
    """Add norms[l, b] * directions[l, b] to item b's probe latent in every layer l."""
    batch_index = torch.arange(idx.shape[0], device=idx.device)

    def make_hook(layer):
        def hook(_module, _inputs, output):
            output = output.clone()
            noise = directions[layer].to(output.dtype) * norms[layer].to(output.dtype)[:, None]
            output[batch_index, probe_positions] = output[batch_index, probe_positions] + noise
            return output
        return hook

    return _with_latent_hooks(model, make_hook, lambda: risk.forward_with_layer_masks(model, idx))


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


def calibration_sqrt_covariances(model, data, reference, block_size, device) -> np.ndarray:
    """Symmetric square roots of each layer's latent covariance on the calibration windows."""
    n_possible_starts = len(data) - block_size
    calibration_high = int(n_possible_starts * 0.55)
    used: list[int] = []
    latents = [[] for _ in range(model.n_layers)]
    for repeat in range(int(reference["calibration_repeats"])):
        rng = np.random.default_rng(risk.CALIBRATION_SEED_BASE + repeat)
        starts = risk.sample_starts(rng, 0, calibration_high,
                                    int(reference["n_calibration_sequences_per_repeat"]),
                                    min_separation=block_size + 1, excluded=used)
        used.extend(map(int, starts))
        for start in starts:
            x = torch.from_numpy(risk.sequence_batch(data, [start], block_size)[:, :-1]).to(device)
            with torch.no_grad():
                _, layer_latents = risk.forward_with_all_latents(model, x)
            for layer, latent in enumerate(layer_latents):
                latents[layer].append(latent[0].float().cpu().numpy().astype(np.float64))
    roots = []
    for layer_latents in latents:
        stacked = np.concatenate(layer_latents)
        values, vectors = np.linalg.eigh(np.cov(stacked, rowvar=False))
        roots.append((vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T)
    return np.stack(roots)


def r_star_table(deltas: np.ndarray, ranks, tail_counts, epsilon: float) -> np.ndarray:
    """deltas: (probes, ranks, offsets) -> integer r* (probes, tail counts)."""
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


def compare(name: str, r_trunc: np.ndarray, r_null: np.ndarray, null_deltas: np.ndarray,
            sequence_ids, tail_counts, ranks, d_c, epsilon, bootstrap_draws, seed_offset) -> dict:
    """Spectrum, premium, cancellation and pre-registered reading of one null vs truncation."""
    spectrum = {}
    for j, k in enumerate(tail_counts):
        entry = {}
        for label, values, s in ((name, r_null[:, j] / d_c, 1),
                                 (f"{name}_minus_truncation", (r_null[:, j] - r_trunc[:, j]) / d_c, 2)):
            ci = risk.sequence_cluster_bootstrap_mean_ci(
                by_sequence(values, sequence_ids), n_bootstrap=bootstrap_draws,
                seed=PAIRED_BOOTSTRAP_SEED + seed_offset + 10 * j + s)
            entry[label] = {"normalized_mean_r_star": float(values.mean()),
                            "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}
        spectrum[str(k)] = entry

    def reading(k: int) -> str:
        gap = spectrum[str(k)][f"{name}_minus_truncation"]["normalized_mean_r_star"]
        if abs(gap) <= WITHIN_TOLERANCE:
            return "within_0.10: spectrum reflects sensitivity to perturbations of that norm"
        return ("null_requires_more_rate: truncation gentler than equal-norm noise" if gap > 0
                else "null_requires_less_rate: reported as is")

    h, top = str(tail_counts[0]), str(tail_counts[-1])
    return {
        "spectrum": spectrum,
        "tail_capacity_premium": float(spectrum[top][name]["normalized_mean_r_star"]
                                       - spectrum[h][name]["normalized_mean_r_star"]),
        "mean_safe_cancellation": cancellation(null_deltas, r_null[:, 0], ranks, epsilon),
        "reading": {f"k={tail_counts[0]} (mean)": reading(tail_counts[0]),
                    f"k={tail_counts[-1]} (max)": reading(tail_counts[-1])},
    }


def reference_r_stars(reference: dict, records: list | None, starts, positions, tail_counts):
    """Published per-probe r* in our probe order: {tail count: array} (only what exists)."""
    keys = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
    if records is not None:
        table = {(int(r["sequence_start"]), int(r["pos"])): r["r_star_by_tail_count"] for r in records}
        return {k: np.asarray([int(table[key][str(k)]) for key in keys]) for k in tail_counts}
    # Older summaries carry only the mean-safe rank per probe, in record order.
    stored = reference["mean_safe_rate_diagnostics_by_record"]
    return {tail_counts[0]: np.asarray([int(r["mean_safe_rank"]) for r in stored[:len(keys)]])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True,
                        help="published risk_capacity_v5_summary.json (orders, grid, hashes)")
    parser.add_argument("--reference-records", type=Path, default=None,
                        help="published records; windows and positions must match them exactly")
    parser.add_argument("--covariance-null", action="store_true",
                        help="also run the exploratory covariance-shaped null")
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
    records = None
    if args.reference_records is not None:
        records = json.loads(args.reference_records.read_text())
        if any(int(r.get("position_horizon", horizon)) != horizon for r in records):
            raise SystemExit("reference records were probed with a different position horizon")
        expected = [(int(r["sequence_start"]), int(r["pos"])) for r in records]
        ours = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
        if ours != (expected[:len(ours)] if args.limit_sequences else expected):
            raise SystemExit("regenerated windows/positions differ from the reference records")
        window_check = "identical to reference records"
    else:
        window_check = ("records unavailable; checked through per-probe mean-safe rank agreement "
                        "with the summary (see truncation_reproduction)")

    sqrt_cov = (calibration_sqrt_covariances(model, data, reference, block_size, device)
                if args.covariance_null else None)
    tokens = risk.sequence_batch(data, starts, block_size)
    n_probes = sum(len(p) for p in positions)
    conditions = ["truncation", "null"] + (["covariance_null"] if args.covariance_null else [])
    deltas = {c: np.empty((n_probes, len(ranks), horizon), dtype=np.float32) for c in conditions}
    norm_sum = np.zeros(len(ranks), dtype=np.float64)
    sequence_ids = np.concatenate([np.full(len(p), s) for s, p in enumerate(positions)])
    probe_offsets = np.cumsum([0] + [len(p) for p in positions])
    directions = {"null": np.stack([null_directions(i, n_layers, d_c) for i in range(n_probes)])}
    if args.covariance_null:
        directions["covariance_null"] = np.stack(
            [null_directions(i, n_layers, d_c, sqrt_cov) for i in range(n_probes)])

    with torch.no_grad():
        # Batched like the reference main(), so baselines match it bit for bit where possible.
        baseline = np.empty((len(starts), block_size), dtype=np.float32)
        for offset in range(0, len(starts), args.probe_batch_size):
            batch = tokens[offset:offset + args.probe_batch_size]
            x = torch.from_numpy(batch[:, :-1]).to(device)
            y = torch.from_numpy(batch[:, 1:]).to(device)
            baseline[offset:offset + len(batch)] = risk.per_token_loss(
                risk.forward_with_layer_masks(model, x), y)
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
                    logits, norms = forward_truncation_with_norms(
                        model, x, channel_masks=masks, probe_positions=pos_t)
                    losses = {"truncation": risk.per_token_loss(logits, y)}
                    norm_sum[rank_idx] += float(norms.sum())
                    for name, dirs_all in directions.items():
                        dirs = torch.as_tensor(dirs_all[first:first + count].transpose(1, 0, 2).copy(),
                                               dtype=torch.float32, device=device)
                        losses[name] = risk.per_token_loss(forward_with_added_noise(
                            model, x, norms=norms, directions=dirs, probe_positions=pos_t), y)
                    for local, pos in enumerate(pos_batch):
                        window = risk.future_loss_slice(int(pos), horizon, block_size)
                        for name, loss in losses.items():
                            deltas[name][first + local, rank_idx] = loss[local, window] - baseline[s, window]
            risk.empty_device_cache(device)
            print(f"rank={rank} complete", flush=True)

    r_stars = {c: r_star_table(deltas[c], ranks, tail_counts, epsilon) for c in conditions}
    trunc_spectrum = {}
    for j, k in enumerate(tail_counts):
        values = r_stars["truncation"][:, j] / d_c
        ci = risk.sequence_cluster_bootstrap_mean_ci(
            by_sequence(values, sequence_ids), n_bootstrap=args.bootstrap_draws,
            seed=PAIRED_BOOTSTRAP_SEED + 10 * j)
        trunc_spectrum[str(k)] = {"normalized_mean_r_star": float(values.mean()),
                                  "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}

    published = reference_r_stars(reference, records, starts, positions, tail_counts)
    agreement = {str(k): float(np.mean(published[k] == r_stars["truncation"][:, tail_counts.index(k)]))
                 for k in published}
    reproduction = {
        "published_normalized_means": {k: v["normalized_mean_r_star"]
                                       for k, v in reference["risk_capacity_spectrum"].items()},
        "per_probe_r_star_agreement_by_tail_count": agreement,
        "required_agreement": MIN_REPRODUCTION_AGREEMENT,
    }
    reproduced = min(agreement.values()) >= MIN_REPRODUCTION_AGREEMENT

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw_path, **deltas, rank_grid=np.asarray(ranks), sequence_ids=sequence_ids,
                        sequence_starts=np.asarray(starts), positions=np.concatenate(positions))
    output = {
        "status": ("smoke" if args.limit_sequences else "complete" if reproduced
                   else "failed_truncation_reproduction"),
        "analysis": "R2 equal-norm random-perturbation null (pre-registered); covariance null exploratory",
        "device": str(device),
        "inputs_sha256": {**hashes, "reference_summary": risk.file_sha256(args.reference_summary),
                          **({"reference_records": risk.file_sha256(args.reference_records)}
                             if args.reference_records else {}),
                          "analysis_source": risk.file_sha256(Path(__file__)),
                          "risk_v5_source": risk.file_sha256(ROOT / "experiments/analyze_risk_capacity_spectrum.py")},
        "raw_npz_sha256": risk.file_sha256(raw_path),
        "null_construction": ("c + ||e_r|| * u in all layers at once, ||e_r|| recorded per layer on the "
                              "truncation run's stream; u unit, isotropic (seed 60521 + probe index) or "
                              "covariance-shaped (exploratory, seed 62521 + probe index), fixed across ranks"),
        "channel_orders": "reference summary layer_channel_orders (not recalibrated)",
        "window_check": window_check,
        "rank_grid": ranks, "tail_counts": tail_counts, "epsilon_nats": epsilon,
        "future_horizon_exact": horizon, "d_c": d_c, "n_layers": n_layers, "n_probes": n_probes,
        "mean_truncation_residual_norm_by_rank": {
            str(r): float(norm_sum[i] / (n_probes * n_layers)) for i, r in enumerate(ranks)},
        "truncation_reproduction": reproduction,
        "truncation": {
            "spectrum": trunc_spectrum,
            "tail_capacity_premium": trunc_spectrum[str(tail_counts[-1])]["normalized_mean_r_star"]
            - trunc_spectrum[str(tail_counts[0])]["normalized_mean_r_star"],
            "mean_safe_cancellation": cancellation(deltas["truncation"], r_stars["truncation"][:, 0],
                                                   ranks, epsilon)},
        **{name: compare(name, r_stars["truncation"], r_stars[name], deltas[name], sequence_ids,
                         tail_counts, ranks, d_c, epsilon, args.bootstrap_draws, 1000 * (i + 1))
           for i, name in enumerate(directions)},
        "per_probe_r_star": {c: {str(k): r_stars[c][:, j].tolist() for j, k in enumerate(tail_counts)}
                             for c in conditions},
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for k in map(str, tail_counts):
        line = f"k={k}: trunc={trunc_spectrum[k]['normalized_mean_r_star']:.3f}"
        for name in directions:
            e = output[name]["spectrum"][k]
            line += (f" {name}={e[name]['normalized_mean_r_star']:.3f}"
                     f" diff={e[f'{name}_minus_truncation']['normalized_mean_r_star']:+.3f}")
        print(line)
    print(json.dumps({n: output[n]["reading"] for n in directions}, indent=2))
    print(json.dumps(reproduction, indent=2))
    if not reproduced and not args.limit_sequences:
        raise SystemExit(f"truncation reproduction below {MIN_REPRODUCTION_AGREEMENT}: {agreement}")


if __name__ == "__main__":
    main()
