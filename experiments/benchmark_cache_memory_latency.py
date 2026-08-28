"""Measured GPU peak memory, resident cache bytes, and decode latency.

Compares three real incremental-decode configurations at matched average rank:
full-width dense MLA cache, a uniform fixed-rank packed cache, and the frozen
contextual router's packed cache. This turns the paper's derived byte-formula
claims into actually measured numbers, and honestly reports whether persistent-
storage savings translate into peak-memory or latency gains in the current
correctness-first implementation (they are not expected to, because both dense
and packed decode steps recompute per-head K/V from the full cached latent
history every step).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from elastic_mla import ContextualElasticMLAGPT, MLAGPT


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_bytes(device):
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def run_full(base, prefill, decode_tokens, device):
    reset_peak(device)
    with torch.no_grad():
        _, cache = base.forward_cached(prefill)
        sync(device)
        t0 = time.perf_counter()
        for step in range(decode_tokens.shape[1]):
            token = decode_tokens[:, step : step + 1]
            _, cache = base.forward_cached(token, caches=cache)
        sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "resident_cache_bytes": base.cache_num_bytes(cache),
        "peak_allocated_bytes": peak_bytes(device),
        "mean_decode_step_seconds": elapsed / decode_tokens.shape[1],
    }


def run_packed_uniform(base, orders, rank, prefill, decode_tokens, device):
    reset_peak(device)
    with torch.no_grad():
        prefill_ranks = torch.full(prefill.shape, rank, dtype=torch.long, device=device)
        _, cache = base.forward_cached_packed(prefill, ranks=prefill_ranks, channel_orders=orders)
        sync(device)
        t0 = time.perf_counter()
        for step in range(decode_tokens.shape[1]):
            token = decode_tokens[:, step : step + 1]
            step_ranks = torch.full(token.shape, rank, dtype=torch.long, device=device)
            _, cache = base.forward_cached_packed(token, ranks=step_ranks, channel_orders=orders, caches=cache)
        sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "resident_cache_bytes": base.packed_cache_num_bytes(cache),
        "peak_allocated_bytes": peak_bytes(device),
        "mean_decode_step_seconds": elapsed / decode_tokens.shape[1],
    }


def run_router(model, prefill, decode_tokens, device):
    reset_peak(device)
    ranks_seen = []
    with torch.no_grad():
        _, cache, ranks, _ = model.forward_cached_packed(prefill)
        ranks_seen.append(ranks.float().mean().item())
        sync(device)
        t0 = time.perf_counter()
        for step in range(decode_tokens.shape[1]):
            token = decode_tokens[:, step : step + 1]
            _, cache, ranks, _ = model.forward_cached_packed(token, caches=cache)
            ranks_seen.append(ranks.float().mean().item())
        sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "resident_cache_bytes": model.packed_cache_num_bytes(cache),
        "peak_allocated_bytes": peak_bytes(device),
        "mean_decode_step_seconds": elapsed / decode_tokens.shape[1],
        "mean_downstream_rank": float(np.mean(ranks_seen)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--contextual-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--prefill-length", type=int, default=192)
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base = MLAGPT(**checkpoint["config"]).to(device).eval()
    base.load_state_dict(checkpoint["model"], strict=True)
    policy = torch.load(args.policy, map_location="cpu", weights_only=False)
    contextual = json.load(open(args.contextual_summary))
    if policy["checkpoint_sha256"] != sha(args.checkpoint) or policy["data_sha256"] != sha(args.data):
        raise ValueError("policy provenance does not match checkpoint/data")
    if policy["channel_orders"] != contextual["layer_channel_orders"]:
        raise ValueError("policy channel orders do not match contextual oracle")
    orders = [torch.tensor(order, device=device) for order in policy["channel_orders"]]
    model = ContextualElasticMLAGPT(base, orders, policy["tiers"]).to(device).eval()
    model.router.load_state_dict(policy["router"])

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    total_needed = args.prefill_length + args.decode_steps + 1
    rng = np.random.default_rng(args.seed)
    starts = rng.integers(0, len(data) - total_needed, size=args.batch_size)
    real_tokens = np.stack([
        np.asarray(data[s: s + total_needed], dtype=np.int64) for s in starts
    ])
    prefill = torch.from_numpy(real_tokens[:, : args.prefill_length].copy()).to(device)
    decode_tokens = torch.from_numpy(
        real_tokens[:, args.prefill_length : args.prefill_length + args.decode_steps].copy()
    ).to(device)

    results = {}
    results["full_mla"] = run_full(base, prefill, decode_tokens, device)
    with torch.no_grad():
        _, _, router_ranks, _ = model.forward_cached_packed(prefill)
    matched_rank = int(round(float(router_ranks.float().mean())))
    matched_rank = min(policy["tiers"], key=lambda t: abs(t - matched_rank))
    results["packed_uniform_matched_rank"] = run_packed_uniform(
        base, orders, matched_rank, prefill, decode_tokens, device
    )
    results["packed_router"] = run_router(model, prefill, decode_tokens, device)

    output = {
        "status": "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_sha256": sha(args.checkpoint),
        "data_sha256": sha(args.data),
        "policy_sha256": sha(args.policy),
        "config": checkpoint["config"],
        "batch_size": args.batch_size,
        "prefill_length": args.prefill_length,
        "decode_steps": args.decode_steps,
        "matched_uniform_rank": matched_rank,
        "results": results,
        "note": (
            "peak_allocated_bytes is None off CUDA. resident_cache_bytes is the exact "
            "tensor-payload size of the cache object retained after the run, matching "
            "cache_num_bytes/packed_cache_num_bytes used elsewhere in this repository."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
