"""R2 (equal-norm perturbation null) and R5 (joint safe rates) on DeepSeek-V2-Lite (analysis R8).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md``, sections R2 (with its
implementation details and the post-review corrections), R5 and R8. Nothing here may be
retuned after its result is seen.

Setting. Exactly the published shard-a run of `analyze_public_mla_spectrum.py`
(``--bases coord pca``, defaults otherwise): same model weights, WikiText-103 windows
with a leading BOS, 24 evaluation sequences x 16 probe positions = 384 probes, the same
32 calibration windows and the same coordinate (saliency) channel order, recomputed by
that script's own ``latent_moments`` and ``build_bases``. Only the coordinate basis is
used. Windows, positions and the rank grid are checked against the published raw npz
and the run stops on any difference.

R2, per probe (sequence-major index i) and grid rank r, all 27 layers at once:

* truncation: the cached latent at the probe position is truncated to its rank-r
  coordinate prefix, as in shard a. A read-only step of the same hook records, per
  layer, ||e_r|| = ||c - c M_r|| at the probe position on this truncation stream;
* null (pre-registered): c -> c + ||e_r|| u instead, with exactly those recorded norms
  and a unit direction u per (probe, layer), isotropic Gaussian from
  ``np.random.default_rng(60521 + i)``, fixed across ranks;
* covariance null (exploratory, not pre-registered): u ~ N(0, Sigma_l), normalized,
  seed 62521 + i, with Sigma_l = E[c c^T] - mu mu^T on the calibration windows. E[c c^T]
  is ``latent_moments``' own estimate; mu is accumulated by an extra read-only hook on
  the same calibration passes.

Reading (pre-registered for R8): at k = 1, |null - truncation| <= 0.10 in mean normalized
r* means the spectrum reflects sensitivity to perturbations of that norm; larger null r*
means truncation is gentler than equal-norm noise; smaller is reported as is. k = H is
reported but not read, because 275 of the 384 shard-a probes sit at the grid floor there.

R5: for each sequence, all 16 probe positions are truncated jointly (all layers), each at
its published shard-a r* for k = H (mean-safe), then k = 1 (max-safe). The published r*
are recomputed from the shard-a raw deltas (``delta_coord``) with the suffix rule at
epsilon = 0.10. From the first probe position on: sequence-mean loss change, fraction
of positions above epsilon, maximum position change, sequence-cluster bootstrap CIs.
Reading: mean-safe rates do not compose if the joint sequence-mean change exceeds
2 epsilon.

Stop rule: the truncation r* recomputed here must agree with the published per-probe r*
on at least 99% of probes at k = H and at k = 1; otherwise the status is
``failed_truncation_reproduction`` and the script exits nonzero (after writing output).

Full-attention simulation, like the spectrum itself; not compressed-cache decoding.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 0.10
TAIL_COUNTS = [32, 16, 8, 4, 2, 1]
JOINT_SEED_BASE = 65_000
MIN_REPRODUCTION_AGREEMENT = 0.99
COMPOSE_LIMIT_IN_EPSILONS = 2.0


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PUBLIC = _load("public_mla", "analyze_public_mla_spectrum.py")
NULL = _load("perturbation_null", "analyze_perturbation_null.py")
RISK = PUBLIC._RISK


class NullProbe(PUBLIC.LatentProbe):
    """``LatentProbe`` with three more uses of the same ``kv_a_layernorm`` hooks.

    ``"probe"``  the parent's truncation, plus a read-only record of
                 ``||c - c M||`` per batch row at ``positions`` into ``residual_norms``;
    ``"noise"``  adds ``noise[layer]`` (batch, d_c) to the latent at ``positions``;
    ``"joint"``  multiplies the latents at ``joint_positions`` (batch, P) by the 0/1
                 channel masks ``joint_masks[layer]`` (batch, P, d_c).
    ``"off"`` and ``"record"`` are the parent's.
    """

    def __init__(self, norms):
        self.residual_norms = {}
        self.noise = None
        self.joint_positions = None
        self.joint_masks = None
        super().__init__(norms)

    def _hook(self, layer):
        parent = super()._hook(layer)

        def hook(module, inputs, output):
            if self.mode in ("off", "record"):
                return parent(module, inputs, output)
            rows = torch.arange(output.shape[0], device=output.device)
            if self.mode == "probe":
                positions = self.positions.to(output.device)
                picked = output[rows, positions].float()
                kept = picked @ self.matrices[layer].to(output.device)
                self.residual_norms[layer] = (picked - kept).norm(dim=-1).cpu()
                return parent(module, inputs, output)
            if self.mode == "noise":
                positions = self.positions.to(output.device)
                shifted = output.clone()
                shifted[rows, positions] = (output[rows, positions].float()
                                            + self.noise[layer].to(output.device)).to(output.dtype)
                return shifted
            if self.mode == "joint":
                positions = self.joint_positions.to(output.device)
                masked = output.clone()
                masked[rows[:, None], positions] = (
                    output[rows[:, None], positions].float()
                    * self.joint_masks[layer].to(output.device)).to(output.dtype)
                return masked
            raise ValueError(f"unknown probe mode {self.mode!r}")
        return hook

    def stacked_norms(self):
        """Recorded residual norms as ``(n_layers, batch)``; every layer must have fired."""
        if sorted(self.residual_norms) != list(range(len(self.norms))):
            raise AssertionError(f"residual norms recorded for layers {sorted(self.residual_norms)}")
        return torch.stack([self.residual_norms[layer] for layer in range(len(self.norms))])


def coordinate_masks(forward, inverse, rank):
    """``(n_layers, d_c)`` 0/1 channel masks of the rank-``rank`` coordinate truncation."""
    masks = []
    for fwd, inv in zip(forward, inverse):
        matrix = inv[:, :rank] @ fwd[:rank]
        if not np.array_equal(matrix, np.diag(np.diag(matrix))):
            raise ValueError("joint masks need the coordinate basis (a diagonal truncation)")
        masks.append(np.diag(matrix))
    return np.stack(masks)


def calibrate(model, probe, calibration, d_c, device):
    """``latent_moments`` plus the per-layer latent mean, from a read-only hook.

    Under gradient checkpointing a layer's forward runs more than once per pass; every
    run sees the same latent, so a sum divided by the number of rows seen is still the
    mean over the same tokens ``latent_moments`` averages over.
    """
    n_layers = len(probe.norms)
    sums = np.zeros((n_layers, d_c))
    counts = np.zeros(n_layers)

    def make_hook(layer):
        def hook(module, inputs, output):
            if probe.mode == "record":
                flat = output.detach().float().reshape(-1, d_c)
                sums[layer] += flat.sum(dim=0).cpu().double().numpy()
                counts[layer] += flat.shape[0]
        return hook

    handles = [norm.register_forward_hook(make_hook(i)) for i, norm in enumerate(probe.norms)]
    try:
        cc, gc, saliency = PUBLIC.latent_moments(model, probe, calibration, d_c, device)
    finally:
        for handle in handles:
            handle.remove()
    return cc, gc, saliency, sums / counts[:, None]


def sqrt_covariances(cc, mean):
    """Symmetric square roots of ``E[c c^T] - mu mu^T`` per layer (eigenvalues clipped at 0)."""
    roots = []
    for second, mu in zip(cc, mean):
        covariance = (second + second.T) / 2 - np.outer(mu, mu)
        values, vectors = np.linalg.eigh(covariance)
        roots.append((vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T)
    return np.stack(roots)


def published_r_stars(delta, rank_grid):
    """Shard-a per-probe r* ``(n_probes, len(TAIL_COUNTS))`` from its raw coordinate deltas."""
    flat = delta.reshape(-1, *delta.shape[2:])
    return NULL.r_star_table(flat, [int(r) for r in rank_grid], TAIL_COUNTS, EPSILON)


def run_r2(model, probe, forward, inverse, eval_tokens, positions, rank_grid, directions,
           batch, horizon, device):
    """Truncation and null deltas ``{condition: (n_probes, ranks, horizon)}`` and residual norms."""
    n_seq, per_seq = positions.shape
    n_probes, n_layers = n_seq * per_seq, len(probe.norms)
    conditions = ["truncation", *directions]
    deltas = {c: np.empty((n_probes, len(rank_grid), horizon), dtype=np.float32) for c in conditions}
    residual_norms = np.empty((n_probes, len(rank_grid), n_layers), dtype=np.float32)
    baselines, row_spread = [], 0.0
    for seq in range(n_seq):
        x = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, :-1], batch, axis=0)).to(device)
        y = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, 1:], batch, axis=0)).to(device)
        probe.mode = "off"
        with torch.no_grad():
            base = PUBLIC.full_losses(model, PUBLIC.hidden_states(model, x), y).cpu().numpy()
        if not np.isfinite(base).all():
            raise FloatingPointError(f"non-finite baseline loss in sequence {seq}")
        row_spread = max(row_spread, float(np.abs(base - base[:1]).max()))
        baselines.append(base)
    started = time.time()
    for rank_index, rank in enumerate(rank_grid):
        probe.set_truncation(forward, inverse, rank)
        for seq in range(n_seq):
            x = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, :-1], batch, axis=0)).to(device)
            y = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, 1:], batch, axis=0)).to(device)
            for offset in range(0, per_seq, batch):
                chunk = positions[seq, offset: offset + batch]
                first = seq * per_seq + offset
                pos = torch.as_tensor(chunk, dtype=torch.long, device=device)
                probe.positions, probe.residual_norms = pos, {}
                losses = {}
                with torch.no_grad():
                    probe.mode = "probe"
                    losses["truncation"] = PUBLIC.window_losses(
                        model, PUBLIC.hidden_states(model, x), y, pos, horizon).cpu().numpy()
                    norms = probe.stacked_norms()  # (n_layers, batch)
                    for name, dirs in directions.items():
                        unit = torch.as_tensor(dirs[first: first + len(chunk)].transpose(1, 0, 2),
                                               dtype=torch.float32)
                        probe.noise = norms[:, :, None] * unit  # (n_layers, batch, d_c)
                        probe.mode = "noise"
                        losses[name] = PUBLIC.window_losses(
                            model, PUBLIC.hidden_states(model, x), y, pos, horizon).cpu().numpy()
                probe.mode, probe.noise = "off", None
                residual_norms[first: first + len(chunk), rank_index] = norms.T.numpy()
                for local, p in enumerate(chunk):
                    reference = baselines[seq][local, p: p + horizon]
                    for name, loss in losses.items():
                        deltas[name][first + local, rank_index] = loss[local] - reference
        print(f"  rank={rank} done ({time.time() - started:.0f}s)", flush=True)
    return deltas, residual_norms, np.stack(baselines)[:, 0], row_spread


def run_r5(model, probe, masks_by_rank, eval_tokens, positions, r_stars, device):
    """Joint deltas ``(n_seq, 2, block)``: row 0 mean-safe (k = H), row 1 max-safe (k = 1)."""
    rank_index = {rank: i for i, rank in enumerate(masks_by_rank)}
    stacked = np.stack(list(masks_by_rank.values()))  # (ranks, n_layers, d_c)
    n_seq, per_seq = positions.shape
    out = np.empty((n_seq, 2, eval_tokens.shape[1] - 1), dtype=np.float32)
    for seq in range(n_seq):
        x = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, :-1], 2, axis=0)).to(device)
        y = torch.from_numpy(np.repeat(eval_tokens[seq: seq + 1, 1:], 2, axis=0)).to(device)
        ranks = r_stars[seq * per_seq: (seq + 1) * per_seq]  # (per_seq, [k=H, k=1])
        chosen = stacked[np.asarray([[rank_index[int(r)] for r in ranks[:, j]] for j in range(2)])]
        # chosen: (2 allocations, per_seq, n_layers, d_c) -> per layer (2, per_seq, d_c)
        probe.joint_masks = [torch.as_tensor(chosen[:, :, layer], dtype=torch.float32)
                             for layer in range(chosen.shape[2])]
        probe.joint_positions = torch.as_tensor(np.repeat(positions[seq: seq + 1], 2, axis=0),
                                                dtype=torch.long, device=device)
        with torch.no_grad():
            probe.mode = "off"
            base = PUBLIC.full_losses(model, PUBLIC.hidden_states(model, x), y).cpu().numpy()
            probe.mode = "joint"
            joint = PUBLIC.full_losses(model, PUBLIC.hidden_states(model, x), y).cpu().numpy()
        probe.mode, probe.joint_masks, probe.joint_positions = "off", None, None
        out[seq] = joint - base
        print(f"  joint sequence {seq + 1}/{n_seq}", flush=True)
    return out


def joint_summary(joint_deltas, positions, r_stars, d_c, draws):
    """R5 metrics per allocation, as in `analyze_latent_quantization.py`."""
    def clustered(per_sequence, seed):
        ci = RISK.sequence_cluster_bootstrap_mean_ci(
            {s: [float(v)] for s, v in enumerate(per_sequence)}, n_bootstrap=draws, seed=seed)
        return {"mean": float(np.mean(per_sequence)),
                "sequence_bootstrap_95pct_ci": [float(c) for c in ci]}

    n_seq, per_seq = positions.shape
    out = {}
    for i, (label, k) in enumerate((("mean_safe", TAIL_COUNTS[0]), ("max_safe", 1))):
        seq_mean, seq_frac, seq_max, rank_sum = [], [], [], []
        for seq in range(n_seq):
            tail = joint_deltas[seq, i, int(positions[seq].min()):]
            seq_mean.append(tail.mean())
            seq_frac.append(np.mean(tail > EPSILON))
            seq_max.append(tail.max())
            rank_sum.append(int(r_stars[seq * per_seq: (seq + 1) * per_seq, i].sum()))
        out[label] = {
            "tail_count": k,
            "mean_probe_rate_fraction_of_d_c": float(np.mean(rank_sum) / (per_seq * d_c)),
            "mean_rank_sum_per_sequence": float(np.mean(rank_sum)),
            "sequence_mean_loss_change_nats": clustered(seq_mean, JOINT_SEED_BASE + 10 * i),
            "fraction_of_positions_above_epsilon": clustered(seq_frac, JOINT_SEED_BASE + 10 * i + 1),
            "max_position_loss_change_nats": clustered(seq_max, JOINT_SEED_BASE + 10 * i + 2),
        }
    composes = (out["mean_safe"]["sequence_mean_loss_change_nats"]["mean"]
                <= COMPOSE_LIMIT_IN_EPSILONS * EPSILON)
    out["pre_registered_reading"] = (
        "mean-safe rates compose: joint sequence-mean change <= 2 epsilon" if composes else
        "mean-safe rates do not compose: joint sequence-mean change > 2 epsilon")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference-npz", type=Path, required=True,
                   help="published shard-a raw arrays (publicmla_a.raw.npz)")
    p.add_argument("--reference-summary", type=Path, default=None,
                   help="published shard-a JSON (publicmla_a.json): model commit, weight hashes, "
                        "saliency hash and published means are checked against it")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), required=True)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    p.add_argument("--max-gpu-memory", nargs="+", default=["10GiB", "13GiB"])
    p.add_argument("--experts-implementation", default="eager")
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--bootstrap-draws", type=int, default=2000, help="R2 (as in R2's script)")
    p.add_argument("--joint-bootstrap-draws", type=int, default=10000, help="R5 (as in R5's script)")
    p.add_argument("--skip-covariance-null", action="store_true",
                   help="omit the exploratory covariance-shaped null (saves a third of the probe time)")
    p.add_argument("--max-sequences", type=int, default=None,
                   help="smoke test: evaluate only the first N of the 24 sequences")
    args = p.parse_args()

    # Shard-a settings (analyze_public_mla_spectrum.py defaults); not exposed on purpose.
    block_size, horizon, n_calib, n_eval, per_seq = 512, 32, 32, 24, 16
    if per_seq % args.probe_batch_size:
        raise ValueError("positions per sequence must be a multiple of the probe batch size")

    reference = np.load(args.reference_npz)
    ref_tokens, ref_positions = reference["eval_tokens"], reference["positions"]
    ref_delta, rank_grid = reference["delta_coord"], [int(r) for r in reference["rank_grid"]]
    published_all = published_r_stars(ref_delta, rank_grid)
    summary = json.loads(args.reference_summary.read_text()) if args.reference_summary else None
    if summary is not None:
        coord = summary["results_by_basis"]["coord"]
        for j, criterion in ((0, "signed_mean"), (len(TAIL_COUNTS) - 1, "maximum")):
            if abs(published_all[:, j].mean() / summary["d_c"]
                   - coord[criterion]["normalized_mean_r_star"]) > 1e-9:
                raise SystemExit(f"npz r* do not reproduce the published {criterion} mean")

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(RISK.CALIBRATION_SEED_BASE)
    device = torch.device(args.device)
    snapshot = Path(snapshot_download(args.model, revision=args.model_revision))
    shards = sorted(snapshot.glob("*.safetensors"))
    print(f"hashing {len(shards)} weight shards", flush=True)
    weight_hashes = {s.name: PUBLIC.file_sha256(s) for s in shards}
    if summary is not None and weight_hashes != summary["provenance"]["weight_shard_sha256"]:
        raise SystemExit("weight shards differ from the published shard-a run")

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    budgets = args.max_gpu_memory + args.max_gpu_memory[-1:] * n_gpus
    max_memory = {i: budgets[i] for i in range(n_gpus)}
    max_memory["cpu"] = "64GiB"
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=getattr(torch, args.dtype), device_map="auto" if n_gpus else None,
        max_memory=max_memory, experts_implementation=args.experts_implementation,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    d_c, n_layers = int(config.kv_lora_rank), int(config.num_hidden_layers)
    if rank_grid != RISK.default_rank_grid(d_c):
        raise SystemExit("reference rank grid is not the shard-a default grid")
    device_map = getattr(model, "hf_device_map", {})
    probe = NullProbe([layer.self_attn.kv_a_layernorm for layer in model.model.layers])

    # Windows exactly as analyze_public_mla_spectrum.main().
    raw_data = load_dataset(args.dataset, args.dataset_config, revision=args.dataset_revision)

    def encode(split, limit_chars=None):
        text = "\n\n".join(t for t in raw_data[split]["text"] if t.strip())
        text = text[:limit_chars] if limit_chars else text
        return np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)

    calibration_ids, evaluation_ids = encode("train", 8_000_000), encode("test")
    calib_starts = RISK.sample_starts(np.random.default_rng(RISK.CALIBRATION_SEED_BASE), 0,
                                      len(calibration_ids) - block_size, n_calib,
                                      min_separation=block_size)
    eval_rng = np.random.default_rng(RISK.EVALUATION_SEED)
    eval_starts = RISK.sample_starts(eval_rng, 0, len(evaluation_ids) - block_size, n_eval,
                                     min_separation=block_size)
    bos = int(tokenizer.bos_token_id)
    calibration = PUBLIC.windows(calibration_ids, calib_starts, block_size, bos)
    eval_tokens = PUBLIC.windows(evaluation_ids, eval_starts, block_size, bos)
    eligible = np.arange(1, block_size - horizon + 1)
    positions = np.stack([np.sort(eval_rng.choice(eligible, size=per_seq, replace=False))
                          for _ in range(n_eval)])
    if not (np.array_equal(eval_tokens, ref_tokens) and np.array_equal(positions, ref_positions)):
        raise SystemExit("regenerated windows/positions differ from the published shard-a npz")
    n_seq = min(args.max_sequences or n_eval, n_eval)
    eval_tokens, positions = eval_tokens[:n_seq], positions[:n_seq]
    n_probes = n_seq * per_seq
    published = published_all[:n_probes]

    started = time.time()
    cc, gc, saliency, latent_mean = calibrate(model, probe, calibration, d_c, device)
    forward, inverse, _ = PUBLIC._BASIS.build_bases(cc, gc, saliency)
    forward, inverse = forward["coord"], inverse["coord"]
    calibration_seconds = time.time() - started
    saliency_hash = PUBLIC.array_sha256(saliency)

    directions = {"null": np.stack([NULL.null_directions(i, n_layers, d_c)
                                    for i in range(n_probes)])}
    if not args.skip_covariance_null:
        roots = sqrt_covariances(cc, latent_mean)
        directions["covariance_null"] = np.stack([NULL.null_directions(i, n_layers, d_c, roots)
                                                  for i in range(n_probes)])

    print("--- R2: truncation and nulls ---", flush=True)
    deltas, residual_norms, baseline, row_spread = run_r2(
        model, probe, forward, inverse, eval_tokens, positions, rank_grid, directions,
        args.probe_batch_size, horizon, device)
    r_stars = {c: NULL.r_star_table(d, rank_grid, TAIL_COUNTS, EPSILON) for c, d in deltas.items()}
    agreement = {str(k): float(np.mean(published[:, j] == r_stars["truncation"][:, j]))
                 for j, k in enumerate(TAIL_COUNTS)}
    reproduced = min(agreement[str(TAIL_COUNTS[0])], agreement["1"]) >= MIN_REPRODUCTION_AGREEMENT

    print("--- R5: joint allocation ---", flush=True)
    masks_by_rank = {rank: coordinate_masks(forward, inverse, rank) for rank in rank_grid}
    joint_r_stars = published[:, [0, len(TAIL_COUNTS) - 1]]
    joint_deltas = run_r5(model, probe, masks_by_rank, eval_tokens, positions, joint_r_stars, device)

    sequence_ids = np.repeat(np.arange(n_seq), per_seq)
    trunc_spectrum = {}
    for j, k in enumerate(TAIL_COUNTS):
        values = r_stars["truncation"][:, j] / d_c
        ci = RISK.sequence_cluster_bootstrap_mean_ci(
            NULL.by_sequence(values, sequence_ids), n_bootstrap=args.bootstrap_draws,
            seed=NULL.PAIRED_BOOTSTRAP_SEED + 10 * j)
        trunc_spectrum[str(k)] = {"normalized_mean_r_star": float(values.mean()),
                                  "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}
    floor_count = int(np.sum(published[:, 0] == rank_grid[0]))
    nulls = {}
    for i, name in enumerate(directions):
        nulls[name] = NULL.compare(name, r_stars["truncation"], r_stars[name], deltas[name],
                                   sequence_ids, TAIL_COUNTS, rank_grid, d_c, EPSILON,
                                   args.bootstrap_draws, 1000 * (i + 1))
        nulls[name]["reading"][f"k={TAIL_COUNTS[0]} (mean)"] = (
            f"reported, not read (R8 pre-registration): {floor_count} of {n_probes} published "
            f"probes sit at the grid floor r = {rank_grid[0]} at k = {TAIL_COUNTS[0]}")
    nulls.setdefault("covariance_null", None)

    status = ("failed_truncation_reproduction" if not reproduced
              else "smoke" if n_seq < n_eval else "complete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.npz")
    np.savez_compressed(raw_path, **deltas, residual_norms=residual_norms, joint_deltas=joint_deltas,
                        joint_r_stars=joint_r_stars, baseline=baseline, rank_grid=np.asarray(rank_grid),
                        positions=positions, sequence_ids=sequence_ids, latent_mean=latent_mean,
                        saliency=saliency)
    import accelerate
    import datasets as datasets_lib

    output = {
        "status": status,
        "analysis": ("R8: R2 equal-norm perturbation null (isotropic pre-registered, covariance "
                     "exploratory) and R5 joint safe-rate allocation on DeepSeek-V2-Lite, "
                     "coordinate basis, shard-a probes"),
        "model": args.model,
        "inputs_sha256": {
            "reference_npz": PUBLIC.file_sha256(args.reference_npz),
            **({"reference_summary": PUBLIC.file_sha256(args.reference_summary)}
               if args.reference_summary else {}),
            "weight_shards": weight_hashes,
            "calibration_tokens": PUBLIC.array_sha256(calibration),
            "evaluation_tokens": PUBLIC.array_sha256(eval_tokens),
            "analysis_source": PUBLIC.file_sha256(Path(__file__)),
            "public_mla_source": PUBLIC.file_sha256(ROOT / "experiments" / "analyze_public_mla_spectrum.py"),
            "perturbation_null_source": PUBLIC.file_sha256(ROOT / "experiments" / "analyze_perturbation_null.py"),
            "risk_v5_source": PUBLIC.file_sha256(ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"),
            "basis_source": PUBLIC.file_sha256(ROOT / "experiments" / "analyze_basis_dependence.py"),
        },
        "raw_npz_sha256": PUBLIC.file_sha256(raw_path),
        "provenance": {
            "model_commit": getattr(config, "_commit_hash", None) or snapshot.name,
            "dataset": f"{args.dataset}/{args.dataset_config}",
            "dataset_fingerprints": {s: getattr(raw_data[s], "_fingerprint", None)
                                     for s in ("train", "test")},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__,
                         "accelerate": accelerate.__version__,
                         "datasets": datasets_lib.__version__},
            "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
            "device_map_devices": sorted({str(v) for v in device_map.values()}),
        },
        "window_check": "eval tokens and probe positions identical to the published shard-a npz",
        "calibration": {
            "seconds": calibration_seconds,
            "saliency_sha256": saliency_hash,
            "saliency_sha256_matches_published": (saliency_hash == summary["saliency_sha256"]
                                                  if summary else None),
            "channel_order": "coord basis from build_bases on this run's latent_moments saliency",
        },
        "null_construction": (
            "c + ||e_r|| u in all layers at once; ||e_r|| = ||c - c M_r|| recorded per layer at the "
            "probe position on the truncation run's own stream; u unit, fixed across ranks: "
            "isotropic (rng 60521 + probe index, pre-registered) or covariance-shaped (rng 62521 + "
            "probe index, exploratory) with Sigma_l = E[cc^T] - mu mu^T (mean-subtracted) on the "
            "32 calibration windows; probe index sequence-major as in the published raw arrays"),
        "dtype": args.dtype, "d_c": d_c, "n_layers": n_layers, "rank_grid": rank_grid,
        "tail_counts": TAIL_COUNTS, "epsilon_nats": EPSILON, "future_horizon_exact": horizon,
        "n_sequences": n_seq, "n_probes": n_probes, "probe_batch_size": args.probe_batch_size,
        "checks": {
            "truncation_full_rank_max_abs_loss_delta": float(np.abs(deltas["truncation"][:, -1]).max()),
            **{f"{name}_full_rank_max_abs_loss_delta": float(np.abs(deltas[name][:, -1]).max())
               for name in directions},
            "identical_row_max_abs_loss_spread": row_spread,
            "max_abs_truncation_delta_minus_published": float(np.abs(
                deltas["truncation"] - ref_delta.reshape(-1, *ref_delta.shape[2:])[:n_probes]).max()),
            "max_abs_baseline_minus_published": float(np.abs(baseline - reference["baseline"][:n_seq]).max()),
        },
        "truncation_reproduction": {
            "per_probe_r_star_agreement_by_tail_count": agreement,
            "required_agreement_at_k_H_and_k_1": MIN_REPRODUCTION_AGREEMENT,
            "published_normalized_means": {str(k): float(published[:, j].mean() / d_c)
                                           for j, k in enumerate(TAIL_COUNTS)},
            "published_probes_at_grid_floor_k_H": floor_count,
        },
        "mean_truncation_residual_norm_by_rank": {
            str(r): float(residual_norms[:, i].mean()) for i, r in enumerate(rank_grid)},
        "truncation": {
            "spectrum": trunc_spectrum,
            "tail_capacity_premium": trunc_spectrum["1"]["normalized_mean_r_star"]
            - trunc_spectrum[str(TAIL_COUNTS[0])]["normalized_mean_r_star"],
            "mean_safe_cancellation": NULL.cancellation(deltas["truncation"], r_stars["truncation"][:, 0],
                                                        rank_grid, EPSILON),
        },
        **nulls,
        "joint_allocation": {
            "r_star_source": "published shard-a per-probe r* (suffix rule, epsilon 0.10, delta_coord)",
            **joint_summary(joint_deltas, positions, joint_r_stars, d_c, args.joint_bootstrap_draws),
        },
        "per_probe_r_star": {c: {str(k): r_stars[c][:, j].tolist() for j, k in enumerate(TAIL_COUNTS)}
                             for c in r_stars},
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for k in map(str, TAIL_COUNTS):
        line = f"k={k}: trunc={trunc_spectrum[k]['normalized_mean_r_star']:.3f}"
        for name in directions:
            e = nulls[name]["spectrum"][k]
            line += (f" {name}={e[name]['normalized_mean_r_star']:.3f}"
                     f" diff={e[f'{name}_minus_truncation']['normalized_mean_r_star']:+.3f}")
        print(line)
    print(json.dumps({n: nulls[n]["reading"] for n in directions}, indent=2))
    print(json.dumps(output["joint_allocation"], indent=2))
    print(json.dumps(output["truncation_reproduction"], indent=2))
    if not reproduced:
        raise SystemExit(f"truncation reproduction below {MIN_REPRODUCTION_AGREEMENT}: {agreement}")


if __name__ == "__main__":
    main()
