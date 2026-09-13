"""Basis-dependence ablation for the nested latent code (theory prediction 7).

Prediction 7 (`notes/theory_contextual_tail_rate.md`): a learned orthogonal nested
basis should reduce safe rates or nonmonotonicity relative to a raw coordinate
permutation, if basis design is a material bottleneck.

Every reported rank in this project is measured in the model's own latent
coordinates, ordered by a gradient-times-activation saliency. That is a
*permutation*, the most restricted possible choice of nested basis. This script
asks whether the required rates are an artifact of that restriction by rerunning
the same truncation probe in two rotated bases:

``coord``  the published saliency permutation (baseline; U is a permutation matrix)
``pca``    eigenvectors of the uncentred second moment ``E[c c^T]``, ordered by
           eigenvalue -- the basis that minimises reconstruction error of the
           latent itself, ignoring the loss
``grad``   eigenvectors of ``sym(E[g c^T])``, ordered by |eigenvalue| -- to first
           order, dropping direction ``u`` changes the loss by ``-(g.u)(u.c)``, so
           ``u^T sym(E[g c^T]) u`` is exactly the mean first-order loss effect
           along ``u``. In the coordinate basis its diagonal is the signed version
           of the published saliency, so ``coord`` is the axis-aligned special case
           of the same criterion.

The rotation is applied to the *model*, not to the probe: since ``c = W_DKV x``
and ``k, v = W_UK c, W_UV c`` are all bias-free, substituting
``W_DKV <- U^T W_DKV`` and ``W_UK, W_UV <- W_UK U, W_UV U`` makes truncating the
first ``k`` rotated coordinates identical to projecting onto ``span(U_k)`` in the
original model, while leaving the full-rank function unchanged. That lets the
already-validated coordinate-masking probe be reused verbatim, and it gives a
cheap correctness check: at full rank every basis must reproduce the original
model's logits exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "experiments"))

from elastic_mla import MLAGPT  # noqa: E402

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"
)
_RISK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RISK)

BASES = ("coord", "pca", "grad")


def latent_moments(
    model: MLAGPT,
    data: np.memmap,
    starts_by_repeat: Sequence[Sequence[int]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
):
    """Accumulate per-layer ``E[c c^T]``, ``E[g c^T]`` and the published saliency.

    One pass produces every statistic all three bases need, so the bases are
    calibrated on exactly the same tokens and the comparison cannot be confounded
    by calibration data.
    """
    n_layers, d_c = model.n_layers, model.d_c
    cc = np.zeros((n_layers, d_c, d_c), dtype=np.float64)
    gc = np.zeros((n_layers, d_c, d_c), dtype=np.float64)
    saliency = np.zeros((n_layers, d_c), dtype=np.float64)
    total_tokens = 0
    for repeat_idx, starts in enumerate(starts_by_repeat):
        for offset in range(0, len(starts), batch_size):
            tokens = _RISK.sequence_batch(data, starts[offset: offset + batch_size], block_size)
            x = torch.from_numpy(tokens[:, :-1]).to(device)
            y = torch.from_numpy(tokens[:, 1:]).to(device)
            model.zero_grad(set_to_none=True)
            logits, latents = _RISK.forward_with_all_latents(model, x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            grads = torch.autograd.grad(loss, latents, only_inputs=True)
            for layer_idx, (latent, grad) in enumerate(zip(latents, grads)):
                # The products run in float32 on the accelerator (MPS has no
                # float64 at all, and it is slow on a T4); only the accumulation
                # across batches is float64, which is where cancellation would
                # otherwise bite.
                c = latent.detach().reshape(-1, d_c)
                # ``loss`` is a token mean, so restore the token weighting that
                # makes a short final microbatch count proportionally.
                g = grad.detach().reshape(-1, d_c) * c.shape[0]
                cc[layer_idx] += (c.T @ c).cpu().double().numpy()
                gc[layer_idx] += (g.T @ c).cpu().double().numpy()
                saliency[layer_idx] += (
                    (grad * latent.detach()).abs().sum(dim=(0, 1)).cpu().double().numpy()
                    * latent.shape[0]
                )
                if layer_idx == 0:
                    total_tokens += c.shape[0]
            del tokens, x, y, logits, loss, latents, grads
            _RISK.empty_device_cache(device)
        print(f"calibration repeat={repeat_idx + 1}/{len(starts_by_repeat)}", flush=True)
    return cc / max(total_tokens, 1), gc / max(total_tokens, 1), saliency


def build_bases(cc, gc, saliency):
    """Return an orthogonal ``U`` per layer per basis, columns in nested order."""
    n_layers, d_c, _ = cc.shape
    out = {name: np.zeros((n_layers, d_c, d_c), dtype=np.float64) for name in BASES}
    diagnostics = {name: [] for name in BASES}
    for layer in range(n_layers):
        order = np.argsort(-saliency[layer])
        permutation = np.zeros((d_c, d_c), dtype=np.float64)
        permutation[order, np.arange(d_c)] = 1.0
        out["coord"][layer] = permutation
        diagnostics["coord"].append({"criterion_values": saliency[layer][order].tolist()})

        values, vectors = np.linalg.eigh((cc[layer] + cc[layer].T) / 2)
        rank_order = np.argsort(-values)
        out["pca"][layer] = vectors[:, rank_order]
        diagnostics["pca"].append({"criterion_values": values[rank_order].tolist()})

        symmetric = (gc[layer] + gc[layer].T) / 2
        values, vectors = np.linalg.eigh(symmetric)
        rank_order = np.argsort(-np.abs(values))
        out["grad"][layer] = vectors[:, rank_order]
        diagnostics["grad"].append({"criterion_values": values[rank_order].tolist()})
    return out, diagnostics


def rotate_model(model: MLAGPT, bases: np.ndarray, device: torch.device) -> MLAGPT:
    """Return a copy whose latent coordinates are the columns of ``bases``.

    ``c' = U^T c`` so that masking the first ``k`` coordinates of ``c'`` projects
    onto ``span(U[:, :k])`` in the original latent space.
    """
    rotated = copy.deepcopy(model)
    for layer_idx, block in enumerate(rotated.blocks):
        u = torch.as_tensor(bases[layer_idx], dtype=torch.float32, device=device)
        attn = block.attn
        with torch.no_grad():
            attn.W_DKV.weight.copy_(u.T @ attn.W_DKV.weight)
            attn.W_UK.weight.copy_(attn.W_UK.weight @ u)
            attn.W_UV.weight.copy_(attn.W_UV.weight @ u)
    return rotated.to(device).eval()


def check_rotation_is_exact(model, rotated, idx, tolerance):
    """A rotation must not change the full-rank function."""
    with torch.no_grad():
        reference = model(idx)[0]
        candidate = rotated(idx)[0]
    deviation = float((reference - candidate).abs().max())
    return deviation, deviation <= tolerance


def probe_r_star(rotated, eval_tokens, positions_by_sequence, rank_grid, args, device):
    """Mean and worst-offset ``r*`` per probe, in the rotated basis's own coordinates."""
    n_sequences = len(positions_by_sequence)
    baseline = np.stack([
        _RISK.per_token_loss(
            rotated(torch.from_numpy(eval_tokens[i: i + 1, :-1]).to(device))[0],
            torch.from_numpy(eval_tokens[i: i + 1, 1:]).to(device),
        )[0]
        for i in range(n_sequences)
    ])
    curves = {
        criterion: [np.empty((len(p), len(rank_grid)), dtype=np.float32) for p in positions_by_sequence]
        for criterion in ("signed_mean", "maximum")
    }
    identity_order = np.tile(np.arange(rotated.d_c), (rotated.n_layers, 1))
    for rank_idx, rank in enumerate(rank_grid):
        for seq_idx, positions in enumerate(positions_by_sequence):
            for offset in range(0, len(positions), args.probe_batch_size):
                batch = positions[offset: offset + args.probe_batch_size]
                tokens = eval_tokens[seq_idx: seq_idx + 1]
                x = torch.from_numpy(np.repeat(tokens[:, :-1], len(batch), axis=0)).to(device)
                y = torch.from_numpy(np.repeat(tokens[:, 1:], len(batch), axis=0)).to(device)
                pos = torch.as_tensor(batch.copy(), dtype=torch.long, device=device)
                masks = _RISK.make_channel_masks(identity_order, rank, len(batch), device)
                with torch.no_grad():
                    logits = _RISK.forward_with_layer_masks(
                        rotated, x, channel_masks=masks, probe_positions=pos
                    )
                    losses = _RISK.per_token_loss(logits, y)
                for local_idx, position in enumerate(batch):
                    window = _RISK.future_loss_slice(int(position), args.future_horizon, losses.shape[1])
                    delta = losses[local_idx, window] - baseline[seq_idx, window]
                    curves["signed_mean"][seq_idx][offset + local_idx, rank_idx] = float(delta.mean())
                    curves["maximum"][seq_idx][offset + local_idx, rank_idx] = float(delta.max())
                del x, y, pos, masks, logits, losses
        _RISK.empty_device_cache(device)
        print(f"  rank={rank} complete", flush=True)
    return curves


def nonmonotonicity(table: np.ndarray, tolerance: float) -> dict:
    """How often does spending more rank make the loss delta worse?

    Prediction 7 offers two ways a better basis can show itself: lower safe rates,
    or a less ragged rank-loss curve. A perfectly nested code would be monotone
    non-increasing in rank, and every violation is a place where the conservative
    suffix rule for ``r*`` has to keep paying for a higher rank.
    """
    steps = np.diff(table, axis=1)
    violations = steps > tolerance
    return {
        "probe_fraction_with_any_violation": float(np.mean(violations.any(axis=1))),
        "violation_rate_per_step": float(np.mean(violations)),
        "mean_total_violation_magnitude": float(np.sum(np.maximum(steps, 0.0), axis=1).mean()),
        "max_single_violation": float(steps.max()) if steps.size else 0.0,
    }


def summarize(curves, rank_grid, epsilon, d_c, bootstrap_draws, seed, monotone_tolerance):
    out = {}
    for criterion, per_sequence in curves.items():
        by_sequence = {}
        for seq_idx, table in enumerate(per_sequence):
            by_sequence[seq_idx] = [
                _RISK.suffix_all_satisfy_r_star(list(row), rank_grid, epsilon) for row in table
            ]
        flat = np.concatenate([np.asarray(v) for v in by_sequence.values()])
        low, high = _RISK.sequence_cluster_bootstrap_mean_ci(
            by_sequence, n_bootstrap=bootstrap_draws, seed=seed
        )
        out[criterion] = {
            "mean_r_star": float(flat.mean()),
            "normalized_mean_r_star": float(flat.mean()) / d_c,
            "sequence_cluster_bootstrap_95pct_ci": [float(low), float(high)],
            "normalized_ci": [float(low) / d_c, float(high) / d_c],
            "r_star_histogram": {str(r): int(np.sum(flat == r)) for r in rank_grid},
            "nonmonotonicity": nonmonotonicity(
                np.concatenate(per_sequence, axis=0), monotone_tolerance
            ),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    p.add_argument("--future-horizon", type=int, default=32)
    p.add_argument("--n-calib-sequences", type=int, default=16)
    p.add_argument("--calibration-repeats", type=int, default=3)
    p.add_argument("--n-eval-sequences", type=int, default=24)
    p.add_argument("--positions-per-sequence", type=int, default=32)
    p.add_argument("--calibration-batch-size", type=int, default=1)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--rank-grid", type=int, nargs="+", default=list(_RISK.DEFAULT_RANK_GRID))
    p.add_argument("--monotone-tolerance", type=float, default=0.0,
                   help="slack before a rank step counts as a monotonicity violation")
    p.add_argument("--rotation-tolerance", type=float, default=2e-3,
                   help="max allowed full-rank logit deviation between a rotated model and the original")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(_RISK.CALIBRATION_SEED_BASE)
    np.random.seed(_RISK.CALIBRATION_SEED_BASE)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_RISK.CALIBRATION_SEED_BASE)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MLAGPT(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    block_size = int(checkpoint["config"]["max_len"])
    d_c = int(checkpoint["config"]["d_c"])
    rank_grid = _RISK.normalize_rank_grid(args.rank_grid, d_c)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_possible_starts = len(data) - block_size
    calibration_high = int(n_possible_starts * 0.55)
    evaluation_low = calibration_high + block_size
    # Reproduce the published sampling exactly: nonoverlapping windows via the
    # rejection sampler (not plain rng.integers), calibration repeats drawn
    # disjointly from each other, and the probe positions drawn from the same
    # generator immediately after the evaluation starts. Any deviation here
    # silently measures a different probe population than the published run.
    sequence_span = block_size + 1
    starts_by_repeat = []
    calibration_used: list[int] = []
    for repeat_idx in range(args.calibration_repeats):
        rng = np.random.default_rng(_RISK.CALIBRATION_SEED_BASE + repeat_idx)
        starts = _RISK.sample_starts(
            rng, 0, calibration_high, args.n_calib_sequences,
            min_separation=sequence_span, excluded=calibration_used,
        )
        starts_by_repeat.append(starts)
        calibration_used.extend(map(int, starts))
    eval_rng = np.random.default_rng(_RISK.EVALUATION_SEED)
    eval_starts = _RISK.sample_starts(
        eval_rng, evaluation_low, n_possible_starts, args.n_eval_sequences,
        min_separation=sequence_span,
    )
    eval_tokens = _RISK.sequence_batch(data, eval_starts, block_size)
    valid_positions = _RISK.valid_probe_positions(block_size, args.future_horizon)
    positions_by_sequence = [
        np.sort(eval_rng.choice(valid_positions,
                                size=min(args.positions_per_sequence, len(valid_positions)),
                                replace=False))
        for _ in range(args.n_eval_sequences)
    ]

    cc, gc, saliency = latent_moments(
        model, data, starts_by_repeat, block_size=block_size,
        batch_size=args.calibration_batch_size, device=device,
    )
    bases, diagnostics = build_bases(cc, gc, saliency)

    probe_idx = torch.from_numpy(eval_tokens[:2, :-1].copy()).to(device)
    results, rotation_checks = {}, {}
    for name in BASES:
        print(f"--- basis {name} ---", flush=True)
        rotated = rotate_model(model, bases[name], device)
        deviation, exact = check_rotation_is_exact(model, rotated, probe_idx, args.rotation_tolerance)
        rotation_checks[name] = {
            "full_rank_max_abs_logit_deviation": deviation,
            "within_tolerance": bool(exact),
        }
        if not exact:
            raise ValueError(
                f"basis {name}: rotated model deviates from the original at full rank "
                f"by {deviation:.3e} > {args.rotation_tolerance:.3e}"
            )
        curves = probe_r_star(rotated, eval_tokens, positions_by_sequence, rank_grid, args, device)
        results[name] = summarize(
            curves, rank_grid, args.epsilon, d_c, args.bootstrap_draws,
            _RISK.BOOTSTRAP_SEED + 500 + BASES.index(name),
            args.monotone_tolerance,
        )
        del rotated
        _RISK.empty_device_cache(device)

    output = {
        "status": "complete",
        "prediction": ("7: a learned orthogonal nested basis should reduce safe rates or "
                       "nonmonotonicity relative to a raw coordinate permutation if basis "
                       "design is a material bottleneck"),
        "device": str(device),
        "checkpoint_sha256": _RISK.file_sha256(args.checkpoint),
        "data_sha256": _RISK.file_sha256(args.data),
        "analysis_source_sha256": _RISK.file_sha256(Path(__file__)),
        "d_c": d_c,
        "n_layers": int(checkpoint["config"]["n_layers"]),
        "rank_grid": rank_grid,
        "epsilon_nats": args.epsilon,
        "future_horizon_exact": args.future_horizon,
        "n_positions_total": sum(len(p) for p in positions_by_sequence),
        "n_evaluation_sequences": args.n_eval_sequences,
        "bootstrap_draws": args.bootstrap_draws,
        "rotation_exactness_check": rotation_checks,
        "results_by_basis": results,
        "basis_criterion_spectra": {
            name: [layer["criterion_values"][:8] for layer in diagnostics[name]] for name in BASES
        },
    }
    for criterion in ("signed_mean", "maximum"):
        base = results["coord"][criterion]["normalized_mean_r_star"]
        output.setdefault("relative_to_coord", {})[criterion] = {
            name: results[name][criterion]["normalized_mean_r_star"] - base for name in BASES
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("BASIS_JSON_BEGIN")
    print(json.dumps(output))
    print("BASIS_JSON_END")


if __name__ == "__main__":
    main()
