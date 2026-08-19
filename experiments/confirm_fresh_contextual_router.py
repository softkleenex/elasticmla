"""One-shot fresh-window confirmation for a frozen contextual routing policy.

The script samples new nonoverlapping windows that are disjoint from every
oracle/training/validation/exploratory window, then compares the frozen router
against two exact-byte noncontextual controls: (1) static floor/ceil rank
interpolation and (2) shuffled router tier assignments.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from elastic_mla import ContextualElasticMLAGPT, MLAGPT


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ce(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).item()


def sample_fresh_starts(data_length, block_size, count, seed, excluded):
    n_possible = data_length - block_size
    calibration_high = int(n_possible * 0.55)
    low, high = calibration_high + block_size, n_possible
    separation = block_size + 1
    rng = np.random.default_rng(seed)
    accepted = []
    forbidden = list(map(int, excluded))
    for _ in range(max(100_000, count * 20_000)):
        if len(accepted) == count:
            return sorted(accepted)
        candidate = int(rng.integers(low, high))
        if all(abs(candidate - other) >= separation for other in forbidden + accepted):
            accepted.append(candidate)
    raise RuntimeError("could not sample fresh nonoverlapping confirmation windows")


def bootstrap_ci(values, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        means[draw] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def sign_flip_p_less(values, seed, draws=100_000):
    """Monte Carlo paired randomization p-value for mean(values) < 0."""
    values = np.asarray(values, dtype=np.float64)
    observed = values.mean()
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for draw in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        null[draw] = (values * signs).mean()
    return float((1 + np.count_nonzero(null <= observed)) / (draws + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--oracle-records", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--contextual-summary", type=Path, required=True)
    parser.add_argument("--scale", choices=("30m", "122m"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=91_827)
    parser.add_argument("--n-sequences", type=int, default=24)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)

    oracle = json.load(open(args.oracle_summary))
    old_records = json.load(open(args.oracle_records))
    contextual = json.load(open(args.contextual_summary))
    manifest = json.load(open(args.protocol_manifest))
    expected = manifest["scales"][args.scale]
    actual_inputs = {
        "policy_sha256": file_sha(args.policy),
        "source_summary_sha256": file_sha(args.oracle_summary),
        "source_records_sha256": file_sha(args.oracle_records),
        "contextual_summary_sha256": file_sha(args.contextual_summary),
    }
    if actual_inputs != {key: expected[key] for key in actual_inputs}:
        raise ValueError("run inputs do not match frozen protocol manifest")
    if (args.seed, args.n_sequences, args.control_repeats) != (
        manifest["seed"], manifest["n_sequences"], manifest["control_repeats"]
    ):
        raise ValueError("run parameters do not match frozen protocol manifest")
    if (contextual["source_summary_sha256"] != actual_inputs["source_summary_sha256"]
            or contextual["source_records_sha256"] != actual_inputs["source_records_sha256"]):
        raise ValueError("source summary/records are not authenticated by contextual oracle")
    policy = torch.load(args.policy, map_location="cpu", weights_only=False)
    checkpoint_hash, data_hash = file_sha(args.checkpoint), file_sha(args.data)
    if checkpoint_hash != oracle["checkpoint_sha256"] or data_hash != oracle["data_sha256"]:
        raise ValueError("checkpoint/data do not match oracle provenance")
    if policy["checkpoint_sha256"] != checkpoint_hash or policy["data_sha256"] != data_hash:
        raise ValueError("policy does not match checkpoint/data")
    if policy["channel_orders"] != oracle["layer_channel_orders"]:
        raise ValueError("policy channel orders do not match oracle")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base = MLAGPT(**checkpoint["config"]).to(device).eval()
    base.load_state_dict(checkpoint["model"], strict=True)
    orders = [torch.tensor(order) for order in policy["channel_orders"]]
    model = ContextualElasticMLAGPT(base, orders, policy["tiers"]).to(device).eval()
    model.router.load_state_dict(policy["router"])

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    block_size = int(checkpoint["config"]["max_len"])
    old_starts = sorted({int(row["sequence_start"]) for row in old_records})
    starts = sample_fresh_starts(
        len(data), block_size, args.n_sequences, args.seed, old_starts
    )
    if any(abs(a - b) < block_size + 1 for a in starts for b in old_starts):
        raise AssertionError("fresh window overlaps a previously used window")

    rows = []
    with torch.no_grad():
        for seq_idx, start in enumerate(starts):
            array = np.asarray(data[start : start + block_size + 1], dtype=np.int64)
            x = torch.from_numpy(array[:-1].copy())[None].to(device)
            y = torch.from_numpy(array[1:].copy())[None].to(device)
            full_logits, _ = base(x)
            router_logits, router_caches, ranks, _ = model.forward_cached_packed(x)
            full_loss, router_loss = ce(full_logits, y), ce(router_logits, y)
            router_bytes = model.packed_cache_num_bytes(router_caches)

            # Exact-byte static interpolation.  Every repeat has the same total
            # downstream rank as the router, but q/q+1 locations ignore content.
            total = int(ranks.sum())
            q, remainder = divmod(total, ranks.numel())
            static_losses, static_bytes = [], []
            flat_size = ranks.numel()
            for repeat in range(args.control_repeats):
                generator = torch.Generator().manual_seed(
                    300_000_000 + args.seed * 10_000 + seq_idx * 100 + repeat
                )
                static = torch.full_like(ranks.cpu(), q)
                if remainder:
                    indices = torch.randperm(flat_size, generator=generator)[:remainder]
                    static.view(-1)[indices] += 1
                static = static.to(device)
                layer_ranks = [torch.full_like(static, base.d_c)] + [
                    static for _ in range(base.n_layers - 1)
                ]
                logits, caches = base.forward_cached_packed(
                    x, ranks=layer_ranks, channel_orders=orders
                )
                static_losses.append(ce(logits, y))
                static_bytes.append(base.packed_cache_num_bytes(caches))

            # Exact-histogram shuffle destroys content/rank alignment only.
            shuffled_losses, shuffled_bytes = [], []
            flat = ranks.cpu().flatten()
            for repeat in range(args.control_repeats):
                generator = torch.Generator().manual_seed(
                    400_000_000 + args.seed * 10_000 + seq_idx * 100 + repeat
                )
                shuffled = flat[
                    torch.randperm(flat.numel(), generator=generator)
                ].view_as(ranks).to(device)
                logits, caches, _, _ = model.forward_cached_packed(
                    x, forced_ranks=shuffled
                )
                shuffled_losses.append(ce(logits, y))
                shuffled_bytes.append(model.packed_cache_num_bytes(caches))

            if any(value != router_bytes for value in static_bytes + shuffled_bytes):
                raise AssertionError("all controls must exactly match router bytes")
            rows.append({
                "sequence": seq_idx,
                "start": start,
                "full_loss": full_loss,
                "router_loss": router_loss,
                "router_bytes": router_bytes,
                "average_downstream_rank": float(ranks.float().mean()),
                "static_loss_mean": float(np.mean(static_losses)),
                "shuffle_loss_mean": float(np.mean(shuffled_losses)),
                "router_minus_static": router_loss - float(np.mean(static_losses)),
                "router_minus_shuffle": router_loss - float(np.mean(shuffled_losses)),
                "rank_histogram": {
                    str(int(tier)): int((ranks == int(tier)).sum())
                    for tier in model.router.tiers
                },
            })
            print(json.dumps(rows[-1]), flush=True)

    router_static = [row["router_minus_static"] for row in rows]
    router_shuffle = [row["router_minus_shuffle"] for row in rows]
    router_full = [row["router_loss"] - row["full_loss"] for row in rows]
    fixed_dense_bytes = (
        block_size * base.n_layers
        * (base.d_c + base.blocks[0].attn.d_rope) * 4
    )
    output = {
        "status": "complete",
        "protocol": "pre-registered fresh-window one-shot confirmation",
        "checkpoint_sha256": checkpoint_hash,
        "data_sha256": data_hash,
        "policy_sha256": file_sha(args.policy),
        "policy_file": args.policy.name,
        "protocol_commit": manifest["protocol_commit"],
        "protocol_manifest_sha256": file_sha(args.protocol_manifest),
        "source_summary_sha256": actual_inputs["source_summary_sha256"],
        "source_records_sha256": actual_inputs["source_records_sha256"],
        "contextual_summary_sha256": actual_inputs["contextual_summary_sha256"],
        "tiers": policy["tiers"],
        "seed": args.seed,
        "n_sequences": args.n_sequences,
        "control_repeats": args.control_repeats,
        "excluded_previous_starts": old_starts,
        "fresh_starts": starts,
        "minimum_fresh_start_separation": int(np.diff(starts).min()),
        "fixed_dense_mla_bytes": fixed_dense_bytes,
        "mean_router_bytes": float(np.mean([row["router_bytes"] for row in rows])),
        "router_over_fixed_dense_mla": float(
            np.mean([row["router_bytes"] for row in rows]) / fixed_dense_bytes
        ),
        "mean_average_downstream_rank": float(np.mean([
            row["average_downstream_rank"] for row in rows
        ])),
        "mean_router_delta_loss": float(np.mean(router_full)),
        "mean_router_minus_exact_static": float(np.mean(router_static)),
        "router_minus_exact_static_bootstrap_95pct_ci": bootstrap_ci(
            router_static, args.seed + 1
        ),
        "router_better_than_static_sequence_count": int(
            np.count_nonzero(np.asarray(router_static) < 0)
        ),
        "router_minus_static_sign_flip_p_less": sign_flip_p_less(
            router_static, args.seed + 2
        ),
        "mean_router_minus_tier_shuffle": float(np.mean(router_shuffle)),
        "router_minus_tier_shuffle_bootstrap_95pct_ci": bootstrap_ci(
            router_shuffle, args.seed + 3
        ),
        "router_better_than_shuffle_sequence_count": int(
            np.count_nonzero(np.asarray(router_shuffle) < 0)
        ),
        "router_minus_shuffle_sign_flip_p_less": sign_flip_p_less(
            router_shuffle, args.seed + 4
        ),
        "success_criterion": (
            "upper 95% paired sequence-bootstrap bound for router-minus-exact-static < 0"
        ),
        "success": bootstrap_ci(router_static, args.seed + 1)[1] < 0,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
