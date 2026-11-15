"""KL/greedy-flip spectrum and bit-width spectrum (pre-registered analyses R6 and R7).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R6, R7).

R6. Same probes, windows and channel orders as the published spectrum (and as R2).
For every probe, grid rank and future offset, the truncation forward records the
target-loss change (to reproduce the published r*), KL(full || truncated) of the
next-token distribution, and whether the greedy token differs from the full model's.
r*_k(delta) uses the same top-k mean and suffix rule on the KL vectors.

R7. Only the probe token's latent is quantized, in all layers at once, with per-token
symmetric absmax quantization at b in {2, 3, 4, 5, 6, 8} bits (fp16 scale); b = 1 is
sign(x) * mean|x| (the L1-optimal binary scale). Full precision closes the axis as 16.
b*_k is the smallest width that is safe under the suffix rule on loss deltas.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
KL_BUDGETS = (0.01, 0.05, 0.10)
PRIMARY_KL_BUDGET = 0.05
BITS = (1, 2, 3, 4, 5, 6, 8)
FULL_PRECISION_BITS = 16
GAP_RATIO = 3.0
FULL_RANK_KL_LIMIT = 1e-4
BOOTSTRAP_SEED = 66_000


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


null_module = load_module("perturbation_null", "analyze_perturbation_null.py")
risk = null_module.risk


def quantize_rows(rows: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric quantize-dequantize; b = 1 is sign * mean|x|."""
    rows = rows.float()
    if bits == 1:
        scale = rows.abs().mean(dim=-1, keepdim=True).half().float()
        return torch.sign(rows) * scale
    qmax = 2 ** (bits - 1) - 1
    scale = (rows.abs().amax(dim=-1, keepdim=True) / qmax).half().float().clamp_min(1e-12)
    return (rows / scale).round().clamp(-qmax, qmax) * scale


def forward_quantized_probe(model, idx: torch.Tensor, probe_positions: torch.Tensor, bits: int):
    """Quantize item b's latent at probe_positions[b] in every layer; others untouched."""
    batch_index = torch.arange(idx.shape[0], device=idx.device)

    def make_hook(_layer):
        def hook(_module, _inputs, output):
            output = output.clone()
            rows = output[batch_index, probe_positions]
            output[batch_index, probe_positions] = quantize_rows(rows, bits).to(output.dtype)
            return output
        return hook

    return null_module._with_latent_hooks(model, make_hook,
                                          lambda: risk.forward_with_layer_masks(model, idx))


def kl_and_flips(full_logp: torch.Tensor, logits: torch.Tensor):
    """full_logp (T, V) log-probs of the full model; logits (B, T, V). Returns (B, T) arrays."""
    logp = F.log_softmax(logits.float(), dim=-1)
    kl = (full_logp.exp()[None] * (full_logp[None] - logp)).sum(dim=-1)
    flips = logits.argmax(dim=-1) != full_logp.argmax(dim=-1)[None]
    return kl.cpu().numpy(), flips.cpu().numpy()


def r_star_table(curves: np.ndarray, axis, tail_counts, budget: float) -> np.ndarray:
    """curves (probes, axis, offsets) -> integer suffix-safe values (probes, tail counts)."""
    out = np.empty((curves.shape[0], len(tail_counts)), dtype=np.int64)
    for i, probe in enumerate(curves):
        for j, k in enumerate(tail_counts):
            out[i, j] = risk.suffix_all_satisfy_r_star(
                [risk.upper_tail_mean(row, k) for row in probe], axis, budget)
    return out


def clustered_means(values: np.ndarray, sequence_ids: np.ndarray, draws: int, seed: int) -> dict:
    ci = risk.sequence_cluster_bootstrap_mean_ci(
        null_module.by_sequence(values, sequence_ids), n_bootstrap=draws, seed=seed)
    return {"mean": float(values.mean()), "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--limit-sequences", type=int, default=None)
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
    ranks = [int(r) for r in reference["rank_grid"]]
    tail_counts = [int(k) for k in reference["tail_counts"]]
    horizon = int(reference["future_horizon_exact"])
    epsilon = float(reference["epsilon_nats"])
    layer_orders = np.asarray(reference["layer_channel_orders"], dtype=np.int64)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    starts, positions = null_module.evaluation_windows(
        data, block_size, horizon, int(reference["n_evaluation_sequences"]),
        int(reference["positions_per_sequence"]))
    if args.limit_sequences:
        starts, positions = starts[:args.limit_sequences], positions[:args.limit_sequences]
    records = None
    if args.reference_records is not None:
        records = json.loads(args.reference_records.read_text())
        expected = [(int(r["sequence_start"]), int(r["pos"])) for r in records]
        ours = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
        if ours != (expected[:len(ours)] if args.limit_sequences else expected):
            raise SystemExit("regenerated windows/positions differ from the reference records")

    tokens = risk.sequence_batch(data, starts, block_size)
    n_probes = sum(len(p) for p in positions)
    sequence_ids = np.concatenate([np.full(len(p), s) for s, p in enumerate(positions)])
    probe_offsets = np.cumsum([0] + [len(p) for p in positions])
    loss_delta = np.empty((n_probes, len(ranks), horizon), dtype=np.float32)
    kl = np.empty_like(loss_delta)
    flips = np.empty(loss_delta.shape, dtype=bool)
    bit_delta = np.empty((n_probes, len(BITS), horizon), dtype=np.float32)

    with torch.no_grad():
        baseline = np.empty((len(starts), block_size), dtype=np.float32)
        for offset in range(0, len(starts), args.probe_batch_size):
            batch = tokens[offset:offset + args.probe_batch_size]
            x = torch.from_numpy(batch[:, :-1]).to(device)
            y = torch.from_numpy(batch[:, 1:]).to(device)
            baseline[offset:offset + len(batch)] = risk.per_token_loss(
                risk.forward_with_layer_masks(model, x), y)
        for s, pos_all in enumerate(positions):
            x1 = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
            full_logp = F.log_softmax(risk.forward_with_layer_masks(model, x1)[0].float(), dim=-1)
            for offset in range(0, len(pos_all), args.probe_batch_size):
                pos_batch = pos_all[offset:offset + args.probe_batch_size]
                count, first = len(pos_batch), probe_offsets[s] + offset
                x = x1.repeat(count, 1)
                y = torch.from_numpy(np.repeat(tokens[s:s + 1, 1:], count, axis=0)).to(device)
                pos_t = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                windows = [risk.future_loss_slice(int(p), horizon, block_size) for p in pos_batch]
                for rank_idx, rank in enumerate(ranks):
                    masks = risk.make_channel_masks(layer_orders, rank, count, device)
                    logits = risk.forward_with_layer_masks(
                        model, x, channel_masks=masks, probe_positions=pos_t)
                    losses = risk.per_token_loss(logits, y)
                    kl_b, flip_b = kl_and_flips(full_logp, logits)
                    for local, window in enumerate(windows):
                        loss_delta[first + local, rank_idx] = losses[local, window] - baseline[s, window]
                        kl[first + local, rank_idx] = kl_b[local, window]
                        flips[first + local, rank_idx] = flip_b[local, window]
                for bit_idx, bits in enumerate(BITS):
                    losses = risk.per_token_loss(forward_quantized_probe(model, x, pos_t, bits), y)
                    for local, window in enumerate(windows):
                        bit_delta[first + local, bit_idx] = losses[local, window] - baseline[s, window]
                risk.empty_device_cache(device)
            print(f"sequence {s + 1}/{len(starts)} complete", flush=True)

    full_rank_kl = float(np.abs(kl[:, ranks.index(d_c)]).max()) if d_c in ranks else float("nan")
    loss_r = r_star_table(loss_delta, ranks, tail_counts, epsilon)
    published = null_module.reference_r_stars(reference, records, starts, positions, tail_counts)
    agreement = {str(k): float(np.mean(published[k] == loss_r[:, tail_counts.index(k)])) for k in published}
    reproduced = min(agreement.values()) >= null_module.MIN_REPRODUCTION_AGREEMENT
    kl_ok = full_rank_kl < FULL_RANK_KL_LIMIT

    kl_spectrum = {}
    for b_idx, budget in enumerate(KL_BUDGETS):
        table = r_star_table(kl, ranks, tail_counts, budget) / d_c
        by_k = {str(k): clustered_means(table[:, j], sequence_ids, args.bootstrap_draws,
                                        BOOTSTRAP_SEED + 100 * b_idx + j)
                for j, k in enumerate(tail_counts)}
        mean_rate, max_rate = by_k[str(tail_counts[0])]["mean"], by_k[str(tail_counts[-1])]["mean"]
        ratio = max_rate / mean_rate if mean_rate > 0 else float("inf")
        kl_spectrum[f"{budget:.2f}"] = {"by_tail_count": by_k, "max_over_mean_ratio": ratio,
                                        "tail_capacity_premium": max_rate - mean_rate,
                                        "large_gap_claim_holds": bool(ratio >= GAP_RATIO)}

    rank_index = {r: i for i, r in enumerate(ranks)}
    flip_rates = {}
    for label, col in (("mean_safe", 0), ("worst_offset_safe", len(tail_counts) - 1)):
        rows = np.stack([flips[i, rank_index[int(r)]] for i, r in enumerate(loss_r[:, col])])
        flip_rates[label] = {
            "mean_rate_fraction_of_d_c": float(loss_r[:, col].mean() / d_c),
            "flip_rate_over_offsets": clustered_means(rows.mean(axis=1), sequence_ids,
                                                      args.bootstrap_draws, BOOTSTRAP_SEED + 900 + col),
            "fraction_of_probes_with_any_flip": float(rows.any(axis=1).mean()),
        }

    bit_axis = list(BITS) + [FULL_PRECISION_BITS]
    padded = np.concatenate([bit_delta, np.zeros((n_probes, 1, horizon), dtype=np.float32)], axis=1)
    bit_table = r_star_table(padded, bit_axis, tail_counts, epsilon)
    bit_spectrum = {str(k): clustered_means(bit_table[:, j].astype(float), sequence_ids,
                                            args.bootstrap_draws, BOOTSTRAP_SEED + 500 + j)
                    for j, k in enumerate(tail_counts)}
    bit_gap = bit_spectrum["1"]["mean"] - bit_spectrum[str(tail_counts[0])]["mean"]
    bit_reading = ("gap present (>= 2 bits)" if bit_gap >= 2 else
                   "gap absent (<= 1 bit)" if bit_gap <= 1 else "inconclusive (between 1 and 2 bits)")

    primary = kl_spectrum[f"{PRIMARY_KL_BUDGET:.2f}"]
    kl_reading = ("gap holds under KL (ratio >= 3 at delta = 0.05)" if primary["large_gap_claim_holds"]
                  else "gap does not hold under KL: the order-of-magnitude gap is a property of the signed target loss")
    status = ("smoke" if args.limit_sequences else
              "complete" if reproduced and kl_ok else "failed_sanity_checks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw_path, loss_delta=loss_delta, kl=kl, flips=flips, bit_delta=bit_delta,
                        rank_grid=np.asarray(ranks), bits=np.asarray(BITS), sequence_ids=sequence_ids)
    output = {
        "status": status,
        "analysis": "R6 KL/greedy-flip spectrum and R7 bit-width spectrum (pre-registered)",
        "device": str(device),
        "inputs_sha256": {**hashes, "reference_summary": risk.file_sha256(args.reference_summary),
                          **({"reference_records": risk.file_sha256(args.reference_records)}
                             if args.reference_records else {}),
                          "analysis_source": risk.file_sha256(Path(__file__)),
                          "perturbation_null_source": risk.file_sha256(ROOT / "experiments/analyze_perturbation_null.py"),
                          "risk_v5_source": risk.file_sha256(ROOT / "experiments/analyze_risk_capacity_spectrum.py")},
        "raw_npz_sha256": risk.file_sha256(raw_path),
        "n_probes": n_probes, "rank_grid": ranks, "tail_counts": tail_counts, "epsilon_nats": epsilon,
        "sanity_checks": {"loss_r_star_agreement_by_tail_count": agreement,
                          "max_full_rank_kl": full_rank_kl, "full_rank_kl_limit": FULL_RANK_KL_LIMIT},
        "R6_kl_spectrum_by_budget": kl_spectrum,
        "R6_greedy_flip_at_loss_safe_rates": flip_rates,
        "R6_pre_registered_reading": kl_reading,
        "R7_bit_axis": bit_axis,
        "R7_bit_star_by_tail_count": bit_spectrum,
        "R7_mean_bit_gap_k1_minus_kH": bit_gap,
        "R7_pre_registered_reading": bit_reading,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("status", "sanity_checks", "R6_pre_registered_reading",
                                             "R7_mean_bit_gap_k1_minus_kH", "R7_pre_registered_reading")},
                     indent=2))
    for budget, entry in kl_spectrum.items():
        print(f"KL delta={budget}: " + " ".join(f"k{k}={entry['by_tail_count'][str(k)]['mean']:.3f}"
                                                for k in tail_counts) + f" ratio={entry['max_over_mean_ratio']:.2f}")
    print("bits: " + " ".join(f"k{k}={bit_spectrum[str(k)]['mean']:.2f}" for k in tail_counts))
    print("flips:", json.dumps(flip_rates))
    if status == "failed_sanity_checks":
        raise SystemExit(f"sanity checks failed: agreement={agreement}, full-rank KL={full_rank_kl}")


if __name__ == "__main__":
    main()
