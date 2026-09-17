"""Is the basis effect specific to the MLA latent, or does it hit a plain KV cache too?

The latent-basis ablation (`analyze_basis_dependence.py`) shows that measuring how
far an MLA latent can be truncated *in the model's coordinate axes* overstates the
requirement by roughly 3x. If that were peculiar to MLA's latent it would be a
property of this architecture. Most of the KV-compression literature instead
compresses the ordinary per-head key and value vectors, so this script asks the same
question of those.

Method. Truncating a cached key to the top ``k`` directions of a per-head orthonormal
basis and reconstructing is exactly ``k_hat = P k`` for the rank-``k`` projection
``P``. Attention scores then become ``q . P k``, which needs no change to the query
path because ``P`` is symmetric. Folding ``P`` into the rows of ``W_UK`` that belong
to that head therefore *is* the truncated-cache model, and the same holds for values
via ``W_UV``. Rotations are block-diagonal over heads because a KV cache stores each
head's vector separately; a global rotation would be a different (and easier) object
than what a per-head compressor can use.

Unlike the risk-capacity probe this truncates **every** position, not one: KV-cache
compression compresses the whole cache, so a uniform truncation is the comparable
measurement and is what the literature's numbers describe.

Bases compared per head:
``coord``  the head's own coordinates, ordered by energy ``E[x_i^2]`` -- the
           axis-aligned choice, and the analogue of what is usually reported
``pca``    principal directions of that head's second moment ``E[x x^T]``
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from elastic_mla import MLAGPT  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"
)
_RISK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RISK)

BASES = ("coord", "pca")


def head_second_moments(model, data, starts_by_repeat, *, block_size, batch_size, device):
    """Per-layer, per-head second moments of the cached key and value vectors."""
    attention = model.blocks[0].attn
    n_layers, n_heads, d_head = model.n_layers, attention.n_heads, attention.d_head
    moments = {
        "key": np.zeros((n_layers, n_heads, d_head, d_head), dtype=np.float64),
        "value": np.zeros((n_layers, n_heads, d_head, d_head), dtype=np.float64),
    }
    tokens_seen = 0
    with torch.no_grad():
        for repeat_idx, starts in enumerate(starts_by_repeat):
            for offset in range(0, len(starts), batch_size):
                batch = _RISK.sequence_batch(data, starts[offset: offset + batch_size], block_size)
                x = torch.from_numpy(batch[:, :-1]).to(device)
                hidden = model.drop(model.tok_emb(x))
                for layer_idx, block in enumerate(model.blocks):
                    normed = block.ln1(hidden)
                    attn = block.attn
                    latent = attn.W_DKV(normed)
                    for name, weight in (("key", attn.W_UK), ("value", attn.W_UV)):
                        projected = weight(latent)
                        per_head = projected.reshape(-1, n_heads, d_head)
                        for head in range(n_heads):
                            column = per_head[:, head, :]
                            moments[name][layer_idx, head] += (
                                (column.T @ column).cpu().double().numpy()
                            )
                    hidden = block(hidden)
                tokens_seen += x.shape[0] * x.shape[1]
                del x, hidden
                _RISK.empty_device_cache(device)
            print(f"calibration repeat={repeat_idx + 1}/{len(starts_by_repeat)}", flush=True)
    for name in moments:
        moments[name] /= max(tokens_seen, 1)
    return moments


def head_bases(moment):
    """``(coord, pca)`` orthonormal bases for one head, columns in nested order."""
    symmetric = (moment + moment.T) / 2
    energy = np.diag(symmetric)
    coord = np.eye(symmetric.shape[0])[:, np.argsort(-energy)]
    values, vectors = np.linalg.eigh(symmetric)
    pca = vectors[:, np.argsort(-values)]
    return {"coord": coord, "pca": pca}, np.sort(values)[::-1], np.sort(energy)[::-1]


def truncated_model(model, bases, rank, device, truncate=("key", "value")):
    """Copy whose cached keys/values are projected onto each head's top-``rank`` directions."""
    out = copy.deepcopy(model)
    n_heads = model.blocks[0].attn.n_heads
    d_head = model.blocks[0].attn.d_head
    for layer_idx, block in enumerate(out.blocks):
        attn = block.attn
        for name, weight in (("key", attn.W_UK), ("value", attn.W_UV)):
            if name not in truncate:
                continue
            with torch.no_grad():
                rows = weight.weight.reshape(n_heads, d_head, -1)
                for head in range(n_heads):
                    basis = bases[name][layer_idx][head][:, :rank]
                    projection = torch.as_tensor(
                        basis @ basis.T, dtype=torch.float32, device=device
                    )
                    rows[head] = projection @ rows[head]
                weight.weight.copy_(rows.reshape(weight.weight.shape))
    return out.to(device).eval()


def evaluate_loss(model, tokens, device, batch_size):
    losses = []
    with torch.no_grad():
        for offset in range(0, len(tokens), batch_size):
            chunk = tokens[offset: offset + batch_size]
            x = torch.from_numpy(chunk[:, :-1].copy()).to(device)
            y = torch.from_numpy(chunk[:, 1:].copy()).to(device)
            logits = model(x)[0]
            losses.append(float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )) * len(chunk))
    return sum(losses) / len(tokens)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    p.add_argument("--n-calib-sequences", type=int, default=16)
    p.add_argument("--calibration-repeats", type=int, default=3)
    p.add_argument("--n-eval-sequences", type=int, default=24)
    p.add_argument("--calibration-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--ranks", type=int, nargs="+", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(_RISK.CALIBRATION_SEED_BASE)
    np.random.seed(_RISK.CALIBRATION_SEED_BASE)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MLAGPT(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    block_size = int(checkpoint["config"]["max_len"])
    d_head = int(checkpoint["config"]["d_head"])
    ranks = args.ranks or [r for r in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64) if r <= d_head]
    if d_head not in ranks:
        ranks.append(d_head)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_possible_starts = len(data) - block_size
    calibration_high = int(n_possible_starts * 0.55)
    evaluation_low = calibration_high + block_size
    span = block_size + 1
    calibration_starts, used = [], []
    for repeat_idx in range(args.calibration_repeats):
        starts = _RISK.sample_starts(
            np.random.default_rng(_RISK.CALIBRATION_SEED_BASE + repeat_idx),
            0, calibration_high, args.n_calib_sequences, min_separation=span, excluded=used,
        )
        calibration_starts.append(starts)
        used.extend(map(int, starts))
    eval_rng = np.random.default_rng(_RISK.EVALUATION_SEED)
    eval_starts = _RISK.sample_starts(
        eval_rng, evaluation_low, n_possible_starts, args.n_eval_sequences, min_separation=span
    )
    eval_tokens = _RISK.sequence_batch(data, eval_starts, block_size)

    moments = head_second_moments(
        model, data, calibration_starts, block_size=block_size,
        batch_size=args.calibration_batch_size, device=device,
    )
    n_heads_model = model.blocks[0].attn.n_heads
    bases = {name: [] for name in ("key", "value")}
    spectra = {name: [] for name in ("key", "value")}
    for name in ("key", "value"):
        for layer_idx in range(model.n_layers):
            per_head, per_head_spectrum = [], []
            for head in range(n_heads_model):
                built, eigenvalues, energy = head_bases(moments[name][layer_idx, head])
                per_head.append(built)
                per_head_spectrum.append({"eigenvalues": eigenvalues.tolist(),
                                          "coordinate_energy": energy.tolist()})
            bases[name].append(per_head)
            spectra[name].append(per_head_spectrum)

    baseline = evaluate_loss(model, eval_tokens, device, args.eval_batch_size)
    print(f"full-cache loss {baseline:.4f}", flush=True)

    curves = {}
    for basis_name in BASES:
        selected = {
            name: [[bases[name][layer][head][basis_name] for head in range(n_heads_model)]
                   for layer in range(model.n_layers)]
            for name in ("key", "value")
        }
        losses = []
        for rank in ranks:
            truncated = truncated_model(model, selected, rank, device)
            loss = evaluate_loss(truncated, eval_tokens, device, args.eval_batch_size)
            losses.append(loss)
            del truncated
            _RISK.empty_device_cache(device)
            print(f"  {basis_name} rank={rank}: loss {loss:.4f} (+{loss - baseline:.4f})", flush=True)
        curves[basis_name] = losses

    def required_rank(losses):
        """Smallest rank whose loss and every larger rank's stay within epsilon."""
        for idx, rank in enumerate(ranks):
            if all(losses[j] - baseline <= args.epsilon for j in range(idx, len(ranks))):
                return rank
        return None

    output = {
        "status": "complete",
        "question": ("does the coordinate-axis measurement bias found for the MLA latent also "
                     "appear for an ordinary per-head KV cache?"),
        "method": ("uniform truncation of every cached key and value onto each head's top-k "
                   "directions, folded into W_UK/W_UV; all positions truncated, unlike the "
                   "position-isolated risk-capacity probe"),
        "device": str(device),
        "checkpoint_sha256": _RISK.file_sha256(args.checkpoint),
        "data_sha256": _RISK.file_sha256(args.data),
        "analysis_source_sha256": _RISK.file_sha256(Path(__file__)),
        "config": checkpoint["config"],
        "d_head": d_head,
        "ranks": ranks,
        "epsilon_nats": args.epsilon,
        "n_evaluation_sequences": args.n_eval_sequences,
        "full_cache_loss": baseline,
        "loss_by_rank": curves,
        "required_rank": {b: required_rank(curves[b]) for b in BASES},
        "normalized_required_rank": {
            b: (required_rank(curves[b]) / d_head if required_rank(curves[b]) else None)
            for b in BASES
        },
        "head_spectra_first_head": {
            name: spectra[name][0][0] for name in ("key", "value")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("KV_BASIS_JSON_BEGIN")
    print(json.dumps({k: v for k, v in output.items() if k != "head_spectra_first_head"}))
    print("KV_BASIS_JSON_END")


if __name__ == "__main__":
    main()
