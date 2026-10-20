"""The KV-cache basis measurement, on public pretrained models.

`notes/kv_cache_basis_generalization.md` shows that measuring how far an ordinary
per-head KV cache can be truncated *in the model's coordinate axes* overstates the
requirement by 1.65-3.2x. Those are three models trained inside this project, so
the obvious objection is that the finding is a property of how they were trained.
This runs the same measurement on public pretrained checkpoints.

Why GPT-2. The cached key is what gets truncated, and in a RoPE model that is the
key *after* a position-dependent rotation, which does not commute with a fixed
projection -- so folding the projection into the key weights would be measuring a
different object than the one a cache actually stores. GPT-2 uses learned absolute
position embeddings, so its cached key is exactly ``W_k x + b_k`` and the fold is
exact. Extending this to RoPE models needs the projection applied after the
rotation, which is a hook rather than a weight edit, and is left out rather than
approximated.

Method, matching `analyze_kv_basis.py`: per-head second moments of the cached key
and value over calibration text; per head, compare the model's own axes ordered by
energy against that head's principal directions; fold the rank-``k`` projection into
the key and value columns of ``c_attn`` (a Conv1D whose weight is ``[in, out]``, with
queries, keys and values concatenated on the output axis); truncate every position,
and report validation loss against retained rank.

A third basis, ``task_aware``, tests why energy-based bases underdeliver. Dropping
direction ``u`` from a cached key perturbs the attention score by ``-(q.u)(u.k)``, so
what a truncation costs depends on the queries, which ``E[kk^T]`` never sees. Treating
queries and keys as independent, the expected squared score error of truncating to a
subspace is ``tr((I-P) Sigma_k (I-P) Sigma_q)``; storing ``T k`` for
``T = U^T Sigma_q^{1/2}``, with ``U`` the eigenvectors of
``Sigma_q^{1/2} Sigma_k Sigma_q^{1/2}``, makes the cost of dropping the last
``d_head - k`` coordinates exactly the tail sum of those eigenvalues -- the minimum at
that rank. Values are treated the same way with ``W_O``'s per-head Gram matrix as the
metric, since a value's effect is what ``W_O`` writes into the residual stream. These
frames are not orthogonal, so the truncation is an oblique projection; it still keeps
the cache's float count, but not vector norms.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASES = ("coord", "pca", "task_aware")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model):
    """Hash of every parameter/buffer (name, dtype, shape, bytes) in sorted name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def attention_blocks(model):
    return model.transformer.h


def qkv_slices(hidden_size):
    """Column ranges of ``c_attn`` holding queries, keys and values."""
    return {"query": slice(0, hidden_size),
            "key": slice(hidden_size, 2 * hidden_size),
            "value": slice(2 * hidden_size, 3 * hidden_size)}


def output_grams(model):
    """Per-layer, per-head ``W_O W_O^T``: the metric a value's error is felt through."""
    config = model.config
    n_heads, d_head = config.n_head, config.n_embd // config.n_head
    grams = np.zeros((config.n_layer, n_heads, d_head, d_head), dtype=np.float64)
    for layer_idx, block in enumerate(attention_blocks(model)):
        weight = block.attn.c_proj.weight.detach().cpu().double().numpy()  # [in, out]
        for head in range(n_heads):
            rows = weight[head * d_head: (head + 1) * d_head, :]
            grams[layer_idx, head] = rows @ rows.T
    return grams


def collect_moments(model, batches, device):
    """Per-layer, per-head uncentred second moments of the cached key and value."""
    config = model.config
    n_layers, n_heads = config.n_layer, config.n_head
    d_head = config.n_embd // n_heads
    moments = {name: np.zeros((n_layers, n_heads, d_head, d_head), dtype=np.float64)
               for name in ("key", "value", "query")}
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            captured[layer_idx] = output.detach()
        return hook

    handles = [attention_blocks(model)[i].attn.c_attn.register_forward_hook(make_hook(i))
               for i in range(n_layers)]
    tokens_seen = 0
    slices = qkv_slices(config.n_embd)
    with torch.no_grad():
        for step, batch in enumerate(batches):
            model(batch.to(device))
            for layer_idx in range(n_layers):
                projected = captured[layer_idx]
                for name in ("key", "value", "query"):
                    part = projected[..., slices[name]].reshape(-1, n_heads, d_head)
                    for head in range(n_heads):
                        column = part[:, head, :].float()
                        moments[name][layer_idx, head] += (
                            (column.T @ column).cpu().double().numpy()
                        )
            tokens_seen += batch.shape[0] * batch.shape[1]
            captured.clear()
            print(f"  calibration batch {step + 1}/{len(batches)}", flush=True)
    for handle in handles:
        handle.remove()
    for name in moments:
        moments[name] /= max(tokens_seen, 1)
    return moments


def metric_frame(metric, target, floor=1e-10):
    """``(forward, inverse, spectrum)`` minimising truncation cost under ``metric``.

    Whitening by ``metric^{1/2}`` and taking principal directions there makes the
    cost of dropping the last coordinates the tail sum of ``spectrum``.
    """
    metric = (metric + metric.T) / 2
    values, vectors = np.linalg.eigh(metric)
    values = np.clip(values, floor * max(float(values.max()), floor), None)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    whitened = root @ target @ root
    whitened = (whitened + whitened.T) / 2
    spectrum, directions = np.linalg.eigh(whitened)
    order = np.argsort(-spectrum)
    directions = directions[:, order]
    return directions.T @ root, inverse_root @ directions, spectrum[order]


def head_bases(moment, task_metric=None):
    """``{name: (forward, inverse)}`` per head; orthogonal bases use ``(U^T, U)``."""
    symmetric = (moment + moment.T) / 2
    energy = np.diag(symmetric)
    coord = np.eye(symmetric.shape[0])[:, np.argsort(-energy)]
    values, vectors = np.linalg.eigh(symmetric)
    pca = vectors[:, np.argsort(-values)]
    built = {"coord": (coord.T, coord), "pca": (pca.T, pca)}
    if task_metric is not None:
        forward, inverse, _ = metric_frame(task_metric, symmetric)
        built["task_aware"] = (forward, inverse)
    return built, np.sort(values)[::-1], np.sort(energy)[::-1]


def row_projection(forward, inverse, rank):
    """Rank-``rank`` projection acting on row vectors: ``k_hat = k @ projection``."""
    keep = np.zeros(forward.shape[0])
    keep[:rank] = 1.0
    return forward.T @ np.diag(keep) @ inverse.T


def truncated_model(model, bases, rank, device):
    """Copy whose cached keys and values are projected onto each head's top-``rank`` directions.

    ``c_attn`` is a Conv1D: ``y = x W + b`` with ``W`` of shape ``[in, 3*hidden]``. For a
    head's column block ``W_h`` the truncated cache is ``k_h P`` (``P`` symmetric), which
    is ``W_h <- W_h P`` and ``b_h <- b_h P``.
    """
    out = copy.deepcopy(model)
    config = out.config
    n_heads, d_head = config.n_head, config.n_embd // config.n_head
    slices = qkv_slices(config.n_embd)
    for layer_idx, block in enumerate(attention_blocks(out)):
        c_attn = block.attn.c_attn
        with torch.no_grad():
            for name in ("key", "value"):
                start = slices[name].start
                for head in range(n_heads):
                    forward, inverse = bases[name][layer_idx][head]
                    projection = torch.as_tensor(
                        row_projection(forward, inverse, rank),
                        dtype=c_attn.weight.dtype, device=c_attn.weight.device,
                    )
                    lo, hi = start + head * d_head, start + (head + 1) * d_head
                    c_attn.weight[:, lo:hi] = c_attn.weight[:, lo:hi] @ projection
                    c_attn.bias[lo:hi] = c_attn.bias[lo:hi] @ projection
    return out.to(device).eval()


def evaluate_loss(model, batches, device):
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            ids = batch.to(device)
            logits = model(ids).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
            )
            total += float(loss) * ids.shape[0]
            count += ids.shape[0]
    return total / count


def build_batches(token_ids, block_size, batch_size, count, offset=0):
    usable = (len(token_ids) - offset) // block_size
    blocks = min(count * batch_size, usable)
    chunk = token_ids[offset: offset + blocks * block_size].reshape(blocks, block_size)
    return [torch.as_tensor(chunk[i: i + batch_size], dtype=torch.long)
            for i in range(0, blocks, batch_size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    # Newer `datasets` requires a namespaced repo id; the bare "wikitext" alias
    # fails with an HfUriError rather than resolving.
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--calibration-batches", type=int, default=16)
    p.add_argument("--evaluation-batches", type=int, default=24)
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50])
    p.add_argument("--ranks", type=int, nargs="+", default=None)
    p.add_argument("--model-revision", default=None, help="Hub revision to pin (default: main)")
    p.add_argument("--dataset-revision", default=None, help="Hub revision to pin (default: main)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision).to(device).eval()
    weights_sha256 = state_dict_sha256(model)
    config = model.config
    d_head = config.n_embd // config.n_head
    ranks = args.ranks or sorted({r for r in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, d_head)
                                  if r <= d_head})

    raw = load_dataset(args.dataset, args.dataset_config, revision=args.dataset_revision)
    def encode(split):
        text = "\n\n".join(t for t in raw[split]["text"] if t.strip())
        return np.asarray(tokenizer(text)["input_ids"], dtype=np.int64)
    calibration_ids, evaluation_ids = encode("train"), encode("test")
    calibration = build_batches(calibration_ids, args.block_size, args.batch_size,
                                args.calibration_batches)
    evaluation = build_batches(evaluation_ids, args.block_size, args.batch_size,
                               args.evaluation_batches)
    print(f"{args.model}: {config.n_layer} layers, {config.n_head} heads, d_head={d_head}; "
          f"{len(calibration)} calibration and {len(evaluation)} evaluation batches", flush=True)

    moments = collect_moments(model, calibration, device)
    grams = output_grams(model)
    # The metric each cached object is actually judged through: keys enter the score
    # against the queries, values reach the residual stream through W_O.
    task_metrics = {"key": moments["query"], "value": grams}
    bases, spectra = {}, {}
    for name in ("key", "value"):
        bases[name] = []
        spectra[name] = []
        for layer_idx in range(config.n_layer):
            per_head, per_head_spectrum = [], []
            for head in range(config.n_head):
                built, eigenvalues, energy = head_bases(
                    moments[name][layer_idx, head], task_metrics[name][layer_idx, head])
                per_head.append(built)
                per_head_spectrum.append({"eigenvalues": eigenvalues.tolist(),
                                          "coordinate_energy": energy.tolist()})
            bases[name].append(per_head)
            spectra[name].append(per_head_spectrum)

    baseline = evaluate_loss(model, evaluation, device)
    print(f"full-cache loss {baseline:.4f}", flush=True)

    curves = {}
    for basis_name in BASES:
        selected = {name: [[bases[name][layer][head][basis_name]
                            for head in range(config.n_head)]
                           for layer in range(config.n_layer)]
                    for name in ("key", "value")}
        losses = []
        for rank in ranks:
            truncated = truncated_model(model, selected, rank, device)
            loss = evaluate_loss(truncated, evaluation, device)
            losses.append(loss)
            del truncated
            print(f"  {basis_name} rank={rank}: loss {loss:.4f} (+{loss - baseline:.4f})", flush=True)
        curves[basis_name] = losses

    def interpolated_rank(losses, epsilon):
        """Retained rank at which the loss curve first crosses the budget."""
        deltas = np.asarray(losses) - baseline
        grid = np.asarray(ranks, dtype=float)
        below = np.where(deltas <= epsilon)[0]
        if len(below) == 0:
            return None
        j = int(below[0])
        if j == 0:
            return float(grid[0])
        return float(grid[j - 1] + (deltas[j - 1] - epsilon) * (grid[j] - grid[j - 1])
                     / (deltas[j - 1] - deltas[j]))

    budgets = {}
    for epsilon in args.epsilon:
        entry = {b: interpolated_rank(curves[b], epsilon) for b in BASES}
        if entry["coord"] and entry["pca"]:
            entry["ratio"] = entry["coord"] / entry["pca"]
        budgets[str(epsilon)] = entry

    first = spectra["key"][0][0]["eigenvalues"]
    # Top-8 principal energy share for every key head, not only layer 0 head 0.
    top8 = np.asarray([sum(h["eigenvalues"][:8]) / sum(h["eigenvalues"])
                       for layer in spectra["key"] for h in layer])
    import datasets as datasets_lib
    import transformers
    provenance = {
        "analysis_source_sha256": file_sha256(__file__),
        "model_revision_requested": args.model_revision,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "model_state_dict_sha256": weights_sha256,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_fingerprints": {s: getattr(raw[s], "_fingerprint", None) for s in ("train", "test")},
        "calibration_token_ids_sha256": array_sha256(calibration_ids),
        "evaluation_token_ids_sha256": array_sha256(evaluation_ids),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets_lib.__version__,
    }
    output = {
        "status": "complete",
        "question": ("does the coordinate-axis KV measurement bias appear in a public "
                     "pretrained model, not only in models trained for this project?"),
        "model": args.model,
        "dataset": f"{args.dataset}/{args.dataset_config}",
        "device": str(device),
        "n_layers": config.n_layer, "n_heads": config.n_head, "d_head": d_head,
        "block_size": args.block_size,
        "calibration_batches": len(calibration), "evaluation_batches": len(evaluation),
        "ranks": ranks,
        "full_cache_loss": baseline,
        "loss_by_rank": curves,
        "required_rank_by_budget": budgets,
        "head_spectra_first_head": {n: spectra[n][0][0] for n in ("key", "value")},
        "key_effective_rank_layer0_head0": float(sum(first) / first[0]),
        "key_top8_energy_fraction_all_heads": {
            "n_heads_total": int(top8.size), "mean": float(top8.mean()),
            "median": float(np.median(top8)), "min": float(top8.min()), "max": float(top8.max()),
        },
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print("PUBLIC_KV_JSON_BEGIN")
    print(json.dumps({k: v for k, v in output.items() if k != "head_spectra_first_head"}))
    print("PUBLIC_KV_JSON_END")


if __name__ == "__main__":
    main()
