"""Latent quantization baseline and joint safe-rate allocation (analyses R4 and R5).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R4, R5).

R4. Per-token symmetric absmax quantization of the MLA latent at every position and
layer (int8, int4; one fp16 scale per token-layer; the decoupled RoPE key is left in
full precision), on the Table 2 evaluation windows. Reports the mean per-token loss
change over all positions and the bytes relative to an fp16 latent.

R5 (needs the published per-probe records). For each evaluation sequence, all of its
probe positions are truncated jointly (all layers), each at its published r* for k = H
(mean-safe) and then for k = 1 (max-safe). Reports the loss change at every position
from the first probe on: sequence mean, fraction above epsilon, maximum, and rank sums.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
QUANT_LEVELS = {"int8": 127, "int4": 7}
COMPOSE_LIMIT_IN_EPSILONS = 2.0


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


null_module = load_module("perturbation_null", "analyze_perturbation_null.py")
risk = null_module.risk


def quantize_per_token(latent: torch.Tensor, qmax: int) -> torch.Tensor:
    """Symmetric absmax quantize-dequantize along the last dim, fp16 scale per token."""
    scale = (latent.abs().amax(dim=-1, keepdim=True) / qmax).half().float().clamp_min(1e-12)
    return (latent / scale).round().clamp(-qmax, qmax) * scale


def forward_quantized(model, idx: torch.Tensor, qmax: int) -> torch.Tensor:
    def make_hook(_layer):
        def hook(_module, _inputs, output):
            return quantize_per_token(output.float(), qmax).to(output.dtype)
        return hook
    return null_module._with_latent_hooks(model, make_hook,
                                          lambda: risk.forward_with_layer_masks(model, idx))


def joint_rank_masks(layer_orders: np.ndarray, rank_by_position: dict[int, int],
                     length: int, device) -> list[torch.Tensor]:
    """Per-layer (1, length, d_c) masks: full rank except the given positions."""
    n_layers, d_c = layer_orders.shape
    masks = []
    for layer in range(n_layers):
        mask = torch.ones(1, length, d_c, device=device)
        for pos, rank in rank_by_position.items():
            mask[0, pos] = 0.0
            mask[0, pos, torch.as_tensor(layer_orders[layer, :rank].copy(), device=device)] = 1.0
        masks.append(mask)
    return masks


def forward_with_full_masks(model, idx: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    hidden = model.drop(model.tok_emb(idx))
    for block, mask in zip(model.blocks, masks):
        hidden = block(hidden, rank_mask=mask.to(hidden.dtype))
    return model.head(model.ln_f(hidden))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, default=None,
                        help="published per-probe records; enables R5")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
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
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    starts, positions = null_module.evaluation_windows(
        data, block_size, horizon, int(reference["n_evaluation_sequences"]),
        int(reference["positions_per_sequence"]))
    tokens = risk.sequence_batch(data, starts, block_size)

    def losses(forward) -> np.ndarray:
        out = np.empty((len(starts), block_size), dtype=np.float32)
        with torch.no_grad():
            for s in range(len(starts)):
                x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
                y = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
                out[s] = risk.per_token_loss(forward(x), y)[0]
        return out

    baseline = losses(lambda x: risk.forward_with_layer_masks(model, x))
    sequence_ids = np.arange(len(starts))

    def clustered(per_sequence: np.ndarray, seed: int) -> dict:
        ci = risk.sequence_cluster_bootstrap_mean_ci(
            {int(s): [float(v)] for s, v in zip(sequence_ids, per_sequence)},
            n_bootstrap=args.bootstrap_draws, seed=seed)
        return {"mean": float(per_sequence.mean()), "sequence_bootstrap_95pct_ci": [float(c) for c in ci]}

    quantization = {}
    for i, (name, qmax) in enumerate(QUANT_LEVELS.items()):
        delta = losses(lambda x, q=qmax: forward_quantized(model, x, q)) - baseline
        quantization[name] = {
            # A 2-byte fp16 scale over a 2*d_c-byte fp16 latent adds 1/d_c. Results up to
            # 2026-11-20 were written with 2/d_c (the scale counted twice); see the R10 note.
            "latent_bytes_fraction_of_fp16": (0.5 if name == "int8" else 0.25) + 1.0 / d_c,
            "mean_loss_change_nats": clustered(delta.mean(axis=1), 64_000 + i),
            "fraction_of_positions_above_epsilon": float(np.mean(delta > epsilon)),
        }
        print(name, quantization[name], flush=True)

    joint = None
    if args.reference_records is not None:
        records = json.loads(args.reference_records.read_text())
        expected = [(int(r["sequence_start"]), int(r["pos"])) for r in records]
        ours = [(int(starts[s]), int(p)) for s in range(len(starts)) for p in positions[s]]
        if ours != expected:
            raise SystemExit("regenerated windows/positions differ from the reference records")
        layer_orders = np.asarray(reference["layer_channel_orders"], dtype=np.int64)
        by_sequence: dict[int, list[dict]] = {}
        for rec in records:
            by_sequence.setdefault(int(rec["seq"]), []).append(rec)
        joint = {}
        for i, (label, k) in enumerate((("mean_safe", str(horizon)), ("max_safe", "1"))):
            seq_mean, seq_frac, seq_max, rank_fraction = [], [], [], []
            for s in range(len(starts)):
                recs = by_sequence[s]
                ranks = {int(r["pos"]): int(r["r_star_by_tail_count"][k]) for r in recs}
                masks = joint_rank_masks(layer_orders, ranks, block_size, device)
                x = torch.from_numpy(tokens[s:s + 1, :-1]).to(device)
                y = torch.from_numpy(tokens[s:s + 1, 1:]).to(device)
                with torch.no_grad():
                    delta = risk.per_token_loss(forward_with_full_masks(model, x, masks), y)[0] - baseline[s]
                tail = delta[min(ranks):]
                seq_mean.append(tail.mean())
                seq_frac.append(np.mean(tail > epsilon))
                seq_max.append(tail.max())
                rank_fraction.append(np.mean(list(ranks.values())) / d_c)
            joint[label] = {
                "tail_count": int(k),
                "mean_probe_rate_fraction_of_d_c": float(np.mean(rank_fraction)),
                "sequence_mean_loss_change_nats": clustered(np.asarray(seq_mean), 65_000 + 10 * i),
                "fraction_of_positions_above_epsilon": clustered(np.asarray(seq_frac), 65_001 + 10 * i),
                "max_position_loss_change_nats": clustered(np.asarray(seq_max), 65_002 + 10 * i),
            }
            print(label, joint[label], flush=True)
        composes = joint["mean_safe"]["sequence_mean_loss_change_nats"]["mean"] <= COMPOSE_LIMIT_IN_EPSILONS * epsilon
        joint["pre_registered_reading"] = (
            "mean-safe rates compose: joint sequence-mean change <= 2 epsilon" if composes else
            "mean-safe rates do not compose: joint sequence-mean change > 2 epsilon")

    output = {
        "status": "complete",
        "analysis": "R4 latent quantization baseline" + (" and R5 joint safe-rate allocation" if joint else ""),
        "device": str(device),
        "inputs_sha256": {**hashes, "reference_summary": risk.file_sha256(args.reference_summary),
                          **({"reference_records": risk.file_sha256(args.reference_records)}
                             if args.reference_records else {}),
                          "analysis_source": risk.file_sha256(Path(__file__))},
        "epsilon_nats": epsilon, "d_c": d_c, "n_sequences": len(starts), "block_size": block_size,
        "quantization": quantization,
        "joint_allocation": joint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
