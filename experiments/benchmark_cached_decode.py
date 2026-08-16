"""Correctness/prototype benchmark for compressed-cache MLA decoding.

The timing baseline is deliberately *naive full-prefix recomputation*, not an
optimized MHA KV-cache implementation.  The reported speedup must not be cited as
MLA-vs-MHA speedup.  This correctness-first MLA path also reconstructs all cached
content K/V tensors every decode step, so its temporary memory and compute remain
linear in context length per step despite the smaller persistent cache payload.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="experiments/exp0_rank_variance/ckpt/latest.pt")
    p.add_argument("--prompt-length", type=int, default=64)
    p.add_argument("--decode-steps", type=int, default=32)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = p.parse_args()

    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MLAGPT(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    total = args.prompt_length + args.decode_steps
    generator = torch.Generator(device="cpu").manual_seed(123)
    tokens = torch.randint(
        0, ckpt["config"]["vocab_size"], (1, total), generator=generator
    ).to(device)

    # Warm up both paths without including compilation/startup effects.
    with torch.no_grad():
        model(tokens[:, :8])
        _, warm_cache = model.forward_cached(tokens[:, :8])
        model.forward_cached(tokens[:, 8:9], caches=warm_cache)
    sync(device)

    naive_predictions = []
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(args.decode_steps):
            prefix_len = args.prompt_length + step
            logits, _ = model(tokens[:, :prefix_len])
            naive_predictions.append(logits[:, -1].cpu())
    sync(device)
    naive_s = time.perf_counter() - t0

    cached_predictions = []
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, caches = model.forward_cached(tokens[:, : args.prompt_length])
        cached_predictions.append(logits[:, -1].cpu())
        for step in range(1, args.decode_steps):
            token_pos = args.prompt_length + step - 1
            logits, caches = model.forward_cached(
                tokens[:, token_pos : token_pos + 1], caches=caches
            )
            cached_predictions.append(logits[:, -1].cpu())
    sync(device)
    cached_s = time.perf_counter() - t0

    naive_predictions = torch.stack(naive_predictions, dim=1)
    cached_predictions = torch.stack(cached_predictions, dim=1)
    max_diff = (naive_predictions - cached_predictions).abs().max().item()
    mla_bytes = model.cache_num_bytes(caches)
    cached_length = args.prompt_length + args.decode_steps - 1
    mha_bytes = model.theoretical_mha_cache_num_bytes(
        batch_size=1,
        sequence_length=cached_length,
        dtype=next(model.parameters()).dtype,
    )

    print(f"device={device}")
    print(f"prompt_length={args.prompt_length} decode_steps={args.decode_steps}")
    print(f"naive_seconds={naive_s:.6f}")
    print(f"cached_seconds={cached_s:.6f}")
    print(f"cached_vs_naive_prefix_recompute_speedup={naive_s / cached_s:.3f}x")
    print("timing_baseline_warning=NOT an optimized MHA KV-cache baseline")
    print(f"max_abs_logit_diff={max_diff:.8g}")
    print(f"mla_cache_bytes={mla_bytes}")
    print(f"standard_mha_theoretical_cache_bytes={mha_bytes}")
    print(f"cache_ratio={mla_bytes / mha_bytes:.6f}")


if __name__ == "__main__":
    main()
