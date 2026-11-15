"""Per-token low-rank truncation spectrum of GPT-2's per-head KV cache (analysis R9).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R9).

Is the mean-vs-worst gap specific to MLA or to our training data? GPT-2 caches
per-head keys and values, not a latent. For a probe position p, every layer's cached
key and value at p are projected, head by head, onto that head's top-r principal
directions (uncentred second moments on WikiText-2 train, as in
``analyze_public_kv_basis.py``); all other positions stay full. The probe records the
next-token loss deltas over [p, p + H) and r* uses the same top-k means and suffix rule
as Section 3.3, with epsilon = 0.10 and rates normalized by d_head.

Reading (R1 criterion): gap present if mean normalized r* at k = 1 is at least 3x that
at k = H.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TAIL_COUNTS = (32, 16, 8, 4, 2, 1)
GAP_RATIO = 3.0
PROBE_SEED = 23_452
BOOTSTRAP_SEED = 67_000


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


risk = load_module("risk_v5", "analyze_risk_capacity_spectrum.py")
kv = load_module("public_kv_basis", "analyze_public_kv_basis.py")


def pca_projections(moments: dict, rank: int) -> dict[str, torch.Tensor]:
    """{key, value}: (layers, heads, d_head, d_head) row-vector projections onto top-rank PCs."""
    out = {}
    for name in ("key", "value"):
        n_layers, n_heads, d_head, _ = moments[name].shape
        proj = np.empty((n_layers, n_heads, d_head, d_head))
        for layer in range(n_layers):
            for head in range(n_heads):
                built, _, _ = kv.head_bases(moments[name][layer, head])
                forward, inverse = built["pca"]
                proj[layer, head] = kv.row_projection(forward, inverse, rank)
        out[name] = torch.as_tensor(proj, dtype=torch.float32)
    return out


def forward_with_probe_projection(model, x, probe_positions, projections):
    """Project item b's cached key and value at probe_positions[b], every layer and head."""
    config = model.config
    n_heads, d_head = config.n_head, config.n_embd // config.n_head
    slices = kv.qkv_slices(config.n_embd)
    batch_index = torch.arange(x.shape[0], device=x.device)

    def make_hook(layer):
        def hook(_module, _inputs, output):
            output = output.clone()
            for name in ("key", "value"):
                cols = slices[name]
                rows = output[batch_index, probe_positions, cols].view(-1, n_heads, d_head)
                projected = torch.einsum("bhd,hde->bhe", rows.float(),
                                         projections[name][layer].to(rows.device))
                output[batch_index, probe_positions, cols] = projected.reshape(rows.shape[0], -1).to(output.dtype)
            return output
        return hook

    handles = [block.attn.c_attn.register_forward_hook(make_hook(i))
               for i, block in enumerate(kv.attention_blocks(model))]
    try:
        return model(x).logits
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--future-horizon", type=int, default=32)
    parser.add_argument("--n-sequences", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=32)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, attn_implementation="eager").to(device).eval()
    config = model.config
    d_head = config.n_embd // config.n_head
    ranks = [r for r in (4, 8, 12, 16, 24, 32, 40, 48, 56, 64) if r <= d_head]
    if ranks[-1] != d_head:
        ranks.append(d_head)

    raw = load_dataset(args.dataset, args.dataset_config, revision=args.dataset_revision)

    def encode(split):
        text = "\n\n".join(t for t in raw[split]["text"] if t.strip())
        return np.asarray(tokenizer(text)["input_ids"], dtype=np.int64)

    train_ids, test_ids = encode("train"), encode("test")
    calibration = kv.build_batches(train_ids, args.block_size, 2, args.calibration_batches)
    moments = kv.collect_moments(model, calibration, device)

    span = args.block_size + 1
    n_windows = len(test_ids) // span
    rng = np.random.default_rng(PROBE_SEED)
    window_ids = np.sort(rng.choice(n_windows, size=args.n_sequences, replace=False))
    tokens = np.stack([test_ids[w * span:(w + 1) * span] for w in window_ids])
    # Position 0 is never probed (the attention sink); horizon must fit.
    valid = np.arange(1, args.block_size - args.future_horizon + 1)
    positions = [np.sort(rng.choice(valid, size=args.positions_per_sequence, replace=False))
                 for _ in range(args.n_sequences)]

    horizon = args.future_horizon
    n_probes = args.n_sequences * args.positions_per_sequence
    deltas = np.empty((n_probes, len(ranks), horizon), dtype=np.float32)
    sequence_ids = np.repeat(np.arange(args.n_sequences), args.positions_per_sequence)
    projections = {r: pca_projections(moments, r) for r in ranks}
    full_rank_max = 0.0
    with torch.no_grad():
        for s in range(args.n_sequences):
            x1 = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
            y1 = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
            base = risk.per_token_loss(model(x1.repeat(args.probe_batch_size, 1)).logits,
                                       y1.repeat(args.probe_batch_size, 1))[0]
            for offset in range(0, args.positions_per_sequence, args.probe_batch_size):
                pos_batch = positions[s][offset:offset + args.probe_batch_size]
                count = len(pos_batch)
                x, y = x1.repeat(count, 1), y1.repeat(count, 1)
                pos_t = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                for rank_idx, rank in enumerate(ranks):
                    losses = risk.per_token_loss(
                        forward_with_probe_projection(model, x, pos_t, projections[rank]), y)
                    for local, pos in enumerate(pos_batch):
                        row = losses[local, pos:pos + horizon] - base[pos:pos + horizon]
                        deltas[s * args.positions_per_sequence + offset + local, rank_idx] = row
                        if rank == d_head:
                            full_rank_max = max(full_rank_max, float(np.abs(row).max()))
            print(f"sequence {s + 1}/{args.n_sequences} complete", flush=True)

    spectrum = {}
    for j, k in enumerate(TAIL_COUNTS):
        per_seq: dict[int, list[float]] = {}
        for i, probe in enumerate(deltas):
            curve = [risk.upper_tail_mean(row, k) for row in probe]
            r_star = risk.suffix_all_satisfy_r_star(curve, ranks, args.epsilon)
            per_seq.setdefault(int(sequence_ids[i]), []).append(r_star / d_head)
        values = [v for vs in per_seq.values() for v in vs]
        ci = risk.sequence_cluster_bootstrap_mean_ci(per_seq, n_bootstrap=args.bootstrap_draws,
                                                     seed=BOOTSTRAP_SEED + j)
        spectrum[str(k)] = {"normalized_mean_r_star": float(np.mean(values)),
                            "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}
    mean_rate = spectrum[str(TAIL_COUNTS[0])]["normalized_mean_r_star"]
    max_rate = spectrum["1"]["normalized_mean_r_star"]
    ratio = max_rate / mean_rate if mean_rate > 0 else float("inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw_path, deltas=deltas, rank_grid=np.asarray(ranks), window_ids=window_ids,
                        positions=np.concatenate(positions))
    output = {
        "status": "complete",
        "analysis": "R9 GPT-2 per-token KV truncation spectrum (pre-registered)",
        "model": args.model, "model_commit": getattr(config, "_commit_hash", None),
        "weights_sha256": kv.state_dict_sha256(model),
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "test_tokens_sha256": kv.array_sha256(test_ids),
        "analysis_source_sha256": risk.file_sha256(Path(__file__)),
        "raw_npz_sha256": risk.file_sha256(raw_path),
        "device": str(device), "d_head": d_head, "rank_grid": ranks,
        "epsilon_nats": args.epsilon, "future_horizon_exact": horizon, "n_probes": n_probes,
        "probe_seed": PROBE_SEED, "max_abs_full_rank_delta": full_rank_max,
        "spectrum": spectrum, "max_over_mean_ratio": ratio,
        "tail_capacity_premium": max_rate - mean_rate,
        "pre_registered_reading": "gap present (ratio >= 3)" if ratio >= GAP_RATIO else "gap absent (ratio < 3)",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(" ".join(f"k{k}={spectrum[str(k)]['normalized_mean_r_star']:.3f}" for k in TAIL_COUNTS),
          f"ratio={ratio:.2f}", output["pre_registered_reading"], f"full-rank max |delta|={full_rank_max:.2e}")


if __name__ == "__main__":
    main()
