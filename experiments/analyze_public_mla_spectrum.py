"""Risk-capacity spectrum and basis dependence on a public, production-trained MLA model.

Every Section 5.1 number and the basis ablation (prediction 7) were measured on three
models trained inside this project on TinyStories. The GPT-2 control
(`analyze_public_kv_basis.py`) moved the question to public checkpoints, but GPT-2 has
no latent: it caches per-head keys and values, so it bounds a *different* object.
DeepSeek-V2-Lite caches exactly the object this paper is about -- a 512-wide MLA latent
plus a 64-wide shared RoPE key per token per layer -- and was trained by its authors on
5.7T tokens. Its ``kv_lora_rank`` equals our 250M model's ``d_c`` (512), so required
rates compare directly.

What is truncated. In the reference implementation the cached latent is
``kv_a_layernorm(kv_a_proj_with_mqa(x)[:, :512])``: the normalized latent is what the
cache stores and what ``kv_b_proj`` expands into per-head keys and values. A forward
hook on ``kv_a_layernorm`` therefore intervenes on the cached object itself, with no
weight edit. The RoPE key is left intact, as in our models, where the decoupled RoPE
key is likewise not part of the nested code.

Probe, matching `analyze_risk_capacity_spectrum.py` and `analyze_basis_dependence.py`:
for a source position ``p``, every layer's cached latent at ``p`` is replaced by its
rank-``k`` truncation in a given basis (all other positions full), and the probe
records the next-token loss deltas over the exact horizon ``[p, p + H)``. ``r*`` is the
smallest grid rank whose upper-tail loss delta, and that of every larger rank, is at
most ``epsilon`` (the suffix-safe rule). The four bases are those of the basis
ablation, built from the same calibration pass:

``coord``         channels ordered by gradient-times-activation saliency (Section 5.1)
``pca``           eigenvectors of ``E[c c^T]``
``grad``          eigenvectors of ``sym(E[g c^T])`` ordered by ``|eigenvalue|``
``model_metric``  the frame that whitens ``kv_b_proj``'s Gram matrix (``build_bases``)

A basis is a pair ``(forward, inverse)`` with ``inverse @ forward = I``; truncating to
rank ``k`` replaces a row-vector latent ``c`` by ``c @ (inverse[:, :k] @ forward[:k]).T``.
For ``coord`` that matrix is an exact 0/1 diagonal, so its full-rank probe should
reproduce the unmodified forward; for the rotated bases it is identity up to float32
rounding. The full-rank probe delta of every basis is reported as a check.

Differences from the in-project runs, all forced by the model rather than chosen:
text is WikiText-103 (TinyStories is out of distribution for this model), every window
starts with the model's BOS token, which is never probed (it is the attention sink), and
the weights run in float16 on two T4s with accelerate's CPU offload for what does not
fit, since the GPUs have no bfloat16 tensor cores. The noise floor this introduces is
measured, not assumed: every probe row is compared with the same row of an unmodified
batch of the same shape, and the full-rank probe of every basis is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_RISK = _load("risk_v5", "analyze_risk_capacity_spectrum.py")
_BASIS = _load("basis_p7", "analyze_basis_dependence.py")

BASES = _BASIS.BASES
TAIL_CRITERIA = {"signed_mean": 32, "tail_16": 16, "tail_8": 8, "tail_4": 4, "tail_2": 2,
                 "maximum": 1}


class LatentProbe:
    """Forward hooks on every layer's ``kv_a_layernorm``: record or truncate the cached latent.

    ``mode`` is ``"off"`` (pass through), ``"record"`` or ``"probe"`` (replace the latent of
    batch item ``b`` at ``positions[b]`` by ``latent @ matrices[layer]``).

    ``"record"`` keeps, per layer, the first latent of the pass that is part of the
    autograd graph in ``latents[layer]`` and collects its gradient into ``grads[layer]``.
    "First that requires grad" makes this work under gradient checkpointing either way:
    reentrant checkpointing runs the forward without grad and records on recomputation,
    non-reentrant records on the forward, whose nodes are the ones backward reaches.
    """

    def __init__(self, norms):
        self.norms = list(norms)
        self.mode = "off"
        self.latents = {}
        self.grads = {}
        self.positions = None
        self.matrices = None
        self.handles = [norm.register_forward_hook(self._hook(i)) for i, norm in enumerate(self.norms)]

    def _hook(self, layer):
        def hook(module, inputs, output):
            if self.mode == "record":
                if output.requires_grad and layer not in self.latents:
                    self.latents[layer] = output
                    output.register_hook(lambda grad: self.grads.__setitem__(layer, grad))
                return None
            if self.mode != "probe":
                return None
            matrix = self.matrices[layer].to(output.device)
            positions = self.positions.to(output.device)
            rows = torch.arange(output.shape[0], device=output.device)
            truncated = output.clone()
            truncated[rows, positions] = (output[rows, positions].float() @ matrix).to(output.dtype)
            return truncated
        return hook

    def set_truncation(self, forward, inverse, rank):
        """Row-vector truncation matrix per layer: ``(inverse[:, :k] @ forward[:k]).T``."""
        self.matrices = [
            torch.as_tensor((inv[:, :rank] @ fwd[:rank]).T, dtype=torch.float32)
            for fwd, inv in zip(forward, inverse)
        ]

    def remove(self):
        for handle in self.handles:
            handle.remove()


def file_sha256(path):
    return _RISK.file_sha256(Path(path))


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def windows(token_ids, starts, block_size, bos_id):
    """``(n, block_size + 1)`` windows: BOS followed by ``block_size`` stream tokens."""
    body = np.stack([token_ids[int(s): int(s) + block_size] for s in starts])
    return np.concatenate([np.full((len(starts), 1), bos_id, dtype=np.int64), body], axis=1)


def hidden_states(model, x):
    return model.model(input_ids=x, use_cache=False).last_hidden_state


def _upcast(logits):
    """Losses in at least float32; float64 stays float64 (the CPU tests run in it)."""
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def window_losses(model, hidden, targets, starts, horizon):
    """Per-token loss over ``[start, start + horizon)`` of each row; lm_head only on the window."""
    index = starts[:, None] + torch.arange(horizon, device=starts.device)[None, :]
    rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    picked = hidden[rows, index.to(hidden.device)]
    logits = _upcast(model.lm_head(picked))
    gold = targets[rows.to(targets.device), index.to(targets.device)].to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1),
                           reduction="none").view(hidden.shape[0], horizon)


def full_losses(model, hidden, targets, chunk=128):
    """Per-token loss over the whole row, lm_head applied in chunks to bound memory."""
    pieces = []
    for start in range(0, hidden.shape[1], chunk):
        logits = _upcast(model.lm_head(hidden[:, start: start + chunk]))
        gold = targets[:, start: start + chunk].to(logits.device)
        pieces.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1),
                                      reduction="none").view(gold.shape))
    return torch.cat(pieces, dim=1)


def latent_moments(model, probe, calibration, d_c, device):
    """Per-layer ``E[c c^T]``, ``E[g c^T]`` and |g * c| saliency, as in `analyze_basis_dependence`.

    ``g`` is the gradient of the *summed* token loss, i.e. the token-weighted gradient the
    in-project runs reconstruct from a mean loss. Gradients flow from an embedding leaf so
    that each layer's latent gets its total derivative, including paths through later
    layers' latents.

    Layer-level gradient checkpointing is not a speed trade here but a correctness of
    memory: with accelerate's CPU offload, a plain autograd graph keeps every onloaded
    expert weight alive on GPU 0 until backward, which overflows it. Recomputing one layer
    at a time during backward onloads its weights only for that layer. The gradients are
    unchanged (checked in `tests/test_public_mla_probe.py`).
    """
    n_layers = len(probe.norms)
    cc = np.zeros((n_layers, d_c, d_c))
    gc = np.zeros((n_layers, d_c, d_c))
    saliency = np.zeros((n_layers, d_c))
    total = 0
    embed = model.get_input_embeddings()
    was_training = model.training
    model.gradient_checkpointing_enable()
    model.train()  # HF only checkpoints in training mode; this model has no dropout
    for index, row in enumerate(calibration):
        x = torch.from_numpy(row[None, :-1]).to(device)
        y = torch.from_numpy(row[None, 1:]).to(device)
        inputs = embed(x).detach().requires_grad_(True)
        probe.latents, probe.grads, probe.mode = {}, {}, "record"
        hidden = model.model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        loss = full_losses(model, hidden, y).sum()
        loss.backward()
        probe.mode = "off"
        if sorted(probe.grads) != list(range(n_layers)):
            raise AssertionError(f"gradients captured for layers {sorted(probe.grads)}")
        for layer in range(n_layers):
            c = probe.latents[layer].detach().float().reshape(-1, d_c)
            g = probe.grads[layer].detach().float().reshape(-1, d_c)
            if not (torch.isfinite(c).all() and torch.isfinite(g).all()):
                raise FloatingPointError(f"non-finite latent or gradient at layer {layer}")
            cc[layer] += (c.T @ c).cpu().double().numpy()
            gc[layer] += (g.T @ c).cpu().double().numpy()
            saliency[layer] += (g * c).abs().sum(dim=0).cpu().double().numpy()
        total += x.shape[1]
        probe.latents, probe.grads = {}, {}
        del inputs, hidden, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"calibration {index + 1}/{len(calibration)}", flush=True)
    model.gradient_checkpointing_disable()
    model.train(was_training)
    return cc / total, gc / total, saliency


def kv_b_weights(snapshot, n_layers):
    """``kv_b_proj`` weights read straight from the safetensors shards (offload-safe)."""
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    weights = []
    for layer in range(n_layers):
        name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        with safe_open(str(snapshot / index[name]), framework="pt") as handle:
            weights.append(handle.get_tensor(name).double().numpy())
    return weights


def probe_basis(model, probe, forward, inverse, eval_tokens, positions, rank_grid, args, device):
    """Upper-tail loss-delta curves ``[criterion][sequence] -> (positions, ranks)`` and raw deltas."""
    n_seq, batch, horizon = len(positions), args.probe_batch_size, args.future_horizon
    curves = {name: [np.empty((len(p), len(rank_grid)), dtype=np.float32) for p in positions]
              for name in TAIL_CRITERIA}
    raw = np.empty((n_seq, len(positions[0]), len(rank_grid), horizon), dtype=np.float32)
    baselines, row_spread = [], 0.0
    for seq in range(n_seq):
        x = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, :-1], batch, axis=0)).to(device)
        y = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, 1:], batch, axis=0)).to(device)
        probe.mode = "off"
        with torch.no_grad():
            base = full_losses(model, hidden_states(model, x), y).cpu().numpy()
        if not np.isfinite(base).all():
            raise FloatingPointError(f"non-finite baseline loss in sequence {seq}")
        # Each probe row is compared with the same row of an unmodified batch of the same
        # shape; how much identical rows disagree is recorded as a noise diagnostic.
        row_spread = max(row_spread, float(np.abs(base - base[:1]).max()))
        baselines.append(base)
    started = time.time()
    for rank_index, rank in enumerate(rank_grid):
        probe.set_truncation(forward, inverse, rank)
        for seq in range(n_seq):
            x = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, :-1], batch, axis=0)).to(device)
            y = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, 1:], batch, axis=0)).to(device)
            for offset in range(0, len(positions[seq]), batch):
                chunk = positions[seq][offset: offset + batch]
                pos = torch.as_tensor(chunk, dtype=torch.long, device=device)
                probe.positions, probe.mode = pos, "probe"
                with torch.no_grad():
                    losses = window_losses(model, hidden_states(model, x), y, pos, horizon)
                probe.mode = "off"
                losses = losses.cpu().numpy()
                for local, p in enumerate(chunk):
                    delta = losses[local] - baselines[seq][local, p: p + horizon]
                    raw[seq, offset + local, rank_index] = delta
                    for name, k in TAIL_CRITERIA.items():
                        curves[name][seq][offset + local, rank_index] = _RISK.upper_tail_mean(delta, k)
        print(f"  rank={rank} done ({time.time() - started:.0f}s)", flush=True)
    return curves, raw, np.stack(baselines)[:, 0], row_spread


def tail_capacity_premium(summary, d_c, draws):
    """Mean worst-offset minus mean-loss ``r*`` (normalized), bootstrapped by sequence."""
    hi, lo = summary["maximum"]["r_star_by_sequence"], summary["signed_mean"]["r_star_by_sequence"]
    per_sequence = {int(k): list((np.asarray(hi[k]) - np.asarray(lo[k])) / d_c) for k in hi}
    flat = np.concatenate([np.asarray(v) for v in per_sequence.values()])
    low, high = _RISK.sequence_cluster_bootstrap_mean_ci(per_sequence, n_bootstrap=draws,
                                                         seed=_RISK.BOOTSTRAP_SEED + 2)
    return {"mean": float(flat.mean()), "sequence_cluster_bootstrap_95pct_ci": [low, high]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    # accelerate runs CPU-offloaded modules on GPU 0, so it needs room for one offloaded
    # expert block (~0.7 GiB in float16) on top of its own weights and activations.
    p.add_argument("--max-gpu-memory", nargs="+", default=["10GiB", "13GiB"],
                   help="device_map weight budget per GPU, in GPU order")
    p.add_argument("--experts-implementation", default="eager",
                   help="transformers MoE kernel; grouped_mm needs a newer GPU than a T4")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--future-horizon", type=int, default=32)
    p.add_argument("--n-calib-sequences", type=int, default=32)
    p.add_argument("--n-eval-sequences", type=int, default=24)
    p.add_argument("--positions-per-sequence", type=int, default=16)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    p.add_argument("--epsilon", type=float, default=0.10)
    p.add_argument("--rank-grid", type=int, nargs="+", default=None)
    p.add_argument("--bases", nargs="+", default=list(BASES), choices=list(BASES))
    p.add_argument("--rotation-tolerance", type=float, default=0.02,
                   help="max full-rank |loss delta| allowed for a rotated basis (float16 weights)")
    args = p.parse_args()
    if args.positions_per_sequence % args.probe_batch_size:
        raise ValueError("positions-per-sequence must be a multiple of probe-batch-size, so every "
                         "probe batch has the baseline's shape")

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(_RISK.CALIBRATION_SEED_BASE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = Path(snapshot_download(args.model, revision=args.model_revision))
    shards = sorted(snapshot.glob("*.safetensors"))
    print(f"hashing {len(shards)} weight shards", flush=True)
    weight_hashes = {s.name: file_sha256(s) for s in shards}

    dtype = getattr(torch, args.dtype)
    n_gpus = torch.cuda.device_count()
    budgets = args.max_gpu_memory + args.max_gpu_memory[-1:] * n_gpus
    max_memory = {i: budgets[i] for i in range(n_gpus)}
    max_memory["cpu"] = "64GiB"
    print(f"device_map budgets: {max_memory}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=dtype, device_map="auto" if n_gpus else None, max_memory=max_memory,
        experts_implementation=args.experts_implementation,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    d_c, n_layers = int(config.kv_lora_rank), int(config.num_hidden_layers)
    device_map = getattr(model, "hf_device_map", {})
    norms = [layer.self_attn.kv_a_layernorm for layer in model.model.layers]
    probe = LatentProbe(norms)
    rank_grid = (_RISK.normalize_rank_grid(args.rank_grid, d_c) if args.rank_grid
                 else _RISK.default_rank_grid(d_c))
    print(f"{args.model}: {n_layers} layers, d_c={d_c}, dtype={args.dtype}, gpus={n_gpus}, "
          f"offloaded={sorted({str(v) for v in device_map.values()})}", flush=True)

    raw_data = load_dataset(args.dataset, args.dataset_config, revision=args.dataset_revision)
    def encode(split, limit_chars=None):
        text = "\n\n".join(t for t in raw_data[split]["text"] if t.strip())
        text = text[:limit_chars] if limit_chars else text
        return np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    # Calibration from the start of train (a few million characters is far more than
    # needed); evaluation from test, which the model's authors did not tune on.
    calibration_ids, evaluation_ids = encode("train", 8_000_000), encode("test")
    span = args.block_size
    calib_starts = _RISK.sample_starts(np.random.default_rng(_RISK.CALIBRATION_SEED_BASE), 0,
                                       len(calibration_ids) - span, args.n_calib_sequences,
                                       min_separation=span)
    eval_rng = np.random.default_rng(_RISK.EVALUATION_SEED)
    eval_starts = _RISK.sample_starts(eval_rng, 0, len(evaluation_ids) - span,
                                      args.n_eval_sequences, min_separation=span)
    bos = int(tokenizer.bos_token_id)
    calibration = windows(calibration_ids, calib_starts, span, bos)
    eval_tokens = windows(evaluation_ids, eval_starts, span, bos)
    # Input index 0 is BOS and is never probed; a probe at p needs logits [p, p + H).
    eligible = np.arange(1, span - args.future_horizon + 1)
    positions = [np.sort(eval_rng.choice(eligible, size=args.positions_per_sequence, replace=False))
                 for _ in range(args.n_eval_sequences)]

    started = time.time()
    cc, gc, saliency = latent_moments(model, probe, calibration, d_c, device)
    up = [(w,) for w in kv_b_weights(snapshot, n_layers)]
    forward, inverse, diagnostics = _BASIS.build_bases(cc, gc, saliency, up_weights=up)
    calibration_seconds = time.time() - started

    results, checks, raw_by_basis, baseline = {}, {}, {}, None
    for name in args.bases:
        print(f"--- basis {name} ---", flush=True)
        curves, raw, baseline, row_spread = probe_basis(model, probe, forward[name], inverse[name],
                                            eval_tokens, positions, rank_grid, args, device)
        full = float(np.abs(raw[:, :, -1, :]).max())
        checks[name] = {"full_rank_max_abs_loss_delta": full,
                        "identical_row_max_abs_loss_spread": row_spread}
        if full > args.rotation_tolerance:
            raise AssertionError(f"basis {name}: full-rank probe moves the loss by {full:.3e}")
        results[name] = _BASIS.summarize(curves, rank_grid, args.epsilon, d_c, args.bootstrap_draws,
                                         _RISK.BOOTSTRAP_SEED + 500 + BASES.index(name), 0.0)
        results[name]["tail_capacity_premium"] = tail_capacity_premium(results[name], d_c,
                                                                       args.bootstrap_draws)
        raw_by_basis[name] = raw
        print(json.dumps({name: {c: round(results[name][c]["normalized_mean_r_star"], 4)
                                 for c in TAIL_CRITERIA}}), flush=True)

    raw_path = args.output.with_suffix(".raw.npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, rank_grid=np.asarray(rank_grid),
                        positions=np.stack(positions), eval_tokens=eval_tokens,
                        baseline=baseline, **{f"delta_{k}": v for k, v in raw_by_basis.items()})
    import accelerate
    import datasets as datasets_lib

    output = {
        "status": "complete",
        "question": ("does the risk-capacity spectrum and its basis dependence, measured on "
                     "TinyStories models trained in this project, hold on a public MLA model?"),
        "model": args.model,
        "provenance": {
            "model_commit": getattr(config, "_commit_hash", None) or snapshot.name,
            "weight_shard_sha256": weight_hashes,
            "dataset": f"{args.dataset}/{args.dataset_config}",
            "dataset_fingerprints": {s: getattr(raw_data[s], "_fingerprint", None)
                                     for s in ("train", "test")},
            "calibration_tokens_sha256": array_sha256(calibration),
            "evaluation_tokens_sha256": array_sha256(eval_tokens),
            "analysis_source_sha256": file_sha256(Path(__file__)),
            "risk_v5_source_sha256": file_sha256(ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"),
            "basis_source_sha256": file_sha256(ROOT / "experiments" / "analyze_basis_dependence.py"),
            "raw_npz_sha256": file_sha256(raw_path),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "accelerate": accelerate.__version__,
                         "datasets": datasets_lib.__version__},
            "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
            "device_map_devices": sorted({str(v) for v in device_map.values()}),
        },
        "dtype": args.dtype,
        "d_c": d_c,
        "n_layers": n_layers,
        "block_size_with_bos": span,
        "rank_grid": rank_grid,
        "epsilon_nats": args.epsilon,
        "future_horizon_exact": args.future_horizon,
        "tail_counts": TAIL_CRITERIA,
        "n_calibration_sequences": args.n_calib_sequences,
        "n_evaluation_sequences": args.n_eval_sequences,
        "positions_per_sequence": args.positions_per_sequence,
        "n_positions_total": int(sum(len(x) for x in positions)),
        "bootstrap_draws": args.bootstrap_draws,
        "baseline_mean_token_loss": float(baseline.mean()),
        "calibration_seconds": calibration_seconds,
        "full_rank_checks": checks,
        "bases_measured": list(results),
        "results_by_basis": results,
        "basis_criterion_spectra": {name: [layer["criterion_values"][:8] for layer in diagnostics[name]]
                                    for name in results if diagnostics.get(name)},
        "saliency_sha256": array_sha256(saliency),
        "paired_comparisons": {
            criterion: {
                f"{a}_minus_{b}": _BASIS.paired_difference(
                    results[a][criterion], results[b][criterion], d_c, args.bootstrap_draws,
                    _BASIS.comparison_seed(a, b, criterion))
                for a, b in _BASIS.comparison_pairs(results)
            }
            for criterion in ("signed_mean", "maximum")
        },
    }
    args.output.write_text(json.dumps(output, indent=2))
    print("PUBLIC_MLA_JSON_BEGIN")
    print(json.dumps(output))
    print("PUBLIC_MLA_JSON_END")


if __name__ == "__main__":
    main()
