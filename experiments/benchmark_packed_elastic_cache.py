"""Actual-checkpoint persistent-memory benchmark for packed ElasticMLA cache.

This is a correctness and storage benchmark.  The Python pack/unpack prototype
rebuilds dense temporary tensors and is not a latency-optimized kernel.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(
        "experiments/exp0_rank_variance/ckpt/latest.pt",
        map_location=device,
        weights_only=False,
    )
    model = MLAGPT(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    summary = json.load(open(
        "experiments/exp0_rank_variance/results/exp0_v3_summary.json"
    ))
    orders = [torch.tensor(order, device=device) for order in summary["layer_channel_orders"]]

    torch.manual_seed(101)
    B, T = 1, 64
    idx = torch.randint(0, ckpt["config"]["vocab_size"], (B, T), device=device)
    # Illustrative tier mix, not a trained-router result: 70/10/10/10 percent.
    pattern = torch.tensor([16] * 7 + [64, 160, 256], device=device)
    token_ranks = pattern[torch.arange(T, device=device) % pattern.numel()][None, :]
    layer_ranks = [token_ranks for _ in range(model.n_layers)]

    with torch.no_grad():
        packed_logits, packed = model.forward_cached_packed(
            idx, ranks=layer_ranks, channel_orders=orders
        )
        masks = []
        for order in orders:
            mask = torch.zeros(B, T, model.d_c, device=device)
            for t in range(T):
                mask[0, t, order[: int(token_ranks[0, t])]] = 1
            masks.append(mask)
        dense_logits, dense = model.forward_cached(idx, rank_masks=masks)

    max_diff = (packed_logits - dense_logits).abs().max().item()
    packed_bytes = model.packed_cache_num_bytes(packed)
    fixed_mla_bytes = model.cache_num_bytes(dense)
    standard_mha_bytes = model.theoretical_mha_cache_num_bytes(
        B, T, dtype=next(model.parameters()).dtype
    )
    avg_rank = token_ranks.float().mean().item()

    print(f"device={device}")
    print(f"average_rank={avg_rank:.3f}")
    print(f"packed_vs_dense_mask_max_abs_logit_diff={max_diff:.8g}")
    print(f"packed_cache_bytes={packed_bytes}")
    print(f"fixed_width_mla_cache_bytes={fixed_mla_bytes}")
    print(f"standard_mha_theoretical_cache_bytes={standard_mha_bytes}")
    print(f"packed_over_fixed_mla={packed_bytes/fixed_mla_bytes:.6f}")
    print(f"packed_over_standard_mha={packed_bytes/standard_mha_bytes:.6f}")
    print("warning=illustrative tier mix; not a trained-router quality result")


if __name__ == "__main__":
    main()
