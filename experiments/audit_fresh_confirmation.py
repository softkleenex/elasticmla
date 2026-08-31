"""Audit a completed fresh-window confirmation against frozen provenance."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(values, seed, draws=10_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return np.percentile(means, [2.5, 97.5])


def sign_flip(values, seed, draws=100_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        null[i] = (values * rng.choice((-1.0, 1.0), len(values))).mean()
    return (1 + np.count_nonzero(null <= values.mean())) / (draws + 1)


def exact_sign_flip_p_less(values, batch_size=500_000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 24:
        raise ValueError("exact enumeration is limited to at most 24 pairs")
    observed_sum = values.sum()
    bit_positions = np.arange(len(values), dtype=np.uint32)
    count = 0
    total = 1 << len(values)
    for start in range(0, total, batch_size):
        masks = np.arange(start, min(start + batch_size, total), dtype=np.uint32)
        null_sums = observed_sum - 2 * (((masks[:, None] >> bit_positions) & 1).dot(values))
        count += int(np.count_nonzero(null_sums <= observed_sum + 1e-15))
    return count / total


def close(a, b, atol=1e-12):
    if not np.allclose(a, b, rtol=0, atol=atol):
        raise AssertionError(f"mismatch: {a!r} != {b!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=("30m", "122m", "250m", "250m_fine"), required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-summary", type=Path, required=True)
    p.add_argument("--source-records", type=Path, required=True)
    p.add_argument("--contextual-summary", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    manifest = json.load(open(args.manifest))
    expected = manifest["scales"][args.scale]
    paths = {
        "policy": args.policy,
        "source_summary": args.source_summary,
        "source_records": args.source_records,
        "contextual_summary": args.contextual_summary,
    }
    actual = {name + "_sha256": sha(path) for name, path in paths.items()}
    for key, value in actual.items():
        if value != expected[key]:
            raise AssertionError(f"{key} differs from frozen manifest")

    source = json.load(open(args.source_summary))
    records = json.load(open(args.source_records))
    contextual = json.load(open(args.contextual_summary))
    result = json.load(open(args.result))
    if contextual["source_summary_sha256"] != actual["source_summary_sha256"]:
        raise AssertionError("contextual oracle does not authenticate source summary")
    if contextual["source_records_sha256"] != actual["source_records_sha256"]:
        raise AssertionError("contextual oracle does not authenticate source records")
    if result["policy_sha256"] != actual["policy_sha256"]:
        raise AssertionError("result policy differs from manifest")
    if result["checkpoint_sha256"] != source["checkpoint_sha256"] or sha(args.checkpoint) != source["checkpoint_sha256"]:
        raise AssertionError("result/checkpoint provenance mismatch")
    if result["data_sha256"] != source["data_sha256"] or sha(args.data) != source["data_sha256"]:
        raise AssertionError("result/data provenance mismatch")
    for key in ("seed", "n_sequences", "control_repeats"):
        if result[key] != manifest[key]:
            raise AssertionError(f"result {key} differs from protocol")

    old_starts = sorted({int(row["sequence_start"]) for row in records})
    if old_starts != result["excluded_previous_starts"]:
        raise AssertionError("result did not exclude exactly all authenticated prior starts")
    starts = result["fresh_starts"]
    rows = result["rows"]
    if len(starts) != manifest["n_sequences"] or len(rows) != len(starts):
        raise AssertionError("unexpected confirmation sequence count")
    if [row["start"] for row in rows] != starts:
        raise AssertionError("row starts differ from declared fresh starts")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    block_size = int(config["max_len"])
    if any(sum(row["rank_histogram"].values()) != block_size for row in rows):
        raise AssertionError("rank histogram length differs from authenticated block size")
    data_length = args.data.stat().st_size // np.dtype(np.uint16).itemsize
    n_possible = data_length - block_size
    low, high = int(n_possible * 0.55) + block_size, n_possible
    rng = np.random.default_rng(manifest["seed"])
    reconstructed = []
    for _ in range(max(100_000, manifest["n_sequences"] * 20_000)):
        if len(reconstructed) == manifest["n_sequences"]:
            break
        candidate = int(rng.integers(low, high))
        if all(abs(candidate - other) >= block_size + 1 for other in old_starts + reconstructed):
            reconstructed.append(candidate)
    if sorted(reconstructed) != starts:
        raise AssertionError("fresh starts do not reproduce from the frozen seed and sampler")
    if not all(low <= start < high for start in starts):
        raise AssertionError("fresh start outside authenticated evaluation region")
    separation = block_size + 1
    if any(abs(a - b) < separation for a in starts for b in old_starts):
        raise AssertionError("fresh/prior window overlap")
    if any(b - a < separation for a, b in zip(starts, starts[1:])):
        raise AssertionError("fresh/fresh window overlap")
    if min(np.diff(starts)) != result["minimum_fresh_start_separation"]:
        raise AssertionError("incorrect minimum separation")

    static = []
    shuffle = []
    delta_full = []
    avg_ranks = []
    bytes_ = []
    n_layers, d_c, d_rope = int(config["n_layers"]), int(config["d_c"]), int(config["d_rope"])
    if (n_layers, d_c) != (source["n_layers"], source["d_c"]):
        raise AssertionError("checkpoint dimensions differ from authenticated source summary")
    expected_fixed_dense = block_size * n_layers * (d_c + d_rope) * 4
    if result["fixed_dense_mla_bytes"] != expected_fixed_dense:
        raise AssertionError("fixed dense MLA denominator differs from authenticated config")
    for row in rows:
        close(row["router_minus_static"], row["router_loss"] - row["static_loss_mean"])
        close(row["router_minus_shuffle"], row["router_loss"] - row["shuffle_loss_mean"])
        hist_sum = sum(int(tier) * count for tier, count in row["rank_histogram"].items())
        close(row["average_downstream_rank"], hist_sum / block_size, atol=2e-5)
        rank_sum = hist_sum
        expected_bytes = (
            4 * (block_size * d_c + (n_layers - 1) * rank_sum)
            + n_layers * (block_size + 1) * 4  # int32 offsets
            + n_layers * block_size * 2        # int16 ranks
            + n_layers * d_c * 2               # int16 channel orders
            + n_layers * block_size * d_rope * 4
        )
        if row["router_bytes"] != expected_bytes:
            raise AssertionError("packed byte total inconsistent with rank histogram")
        static.append(row["router_minus_static"])
        shuffle.append(row["router_minus_shuffle"])
        delta_full.append(row["router_loss"] - row["full_loss"])
        avg_ranks.append(row["average_downstream_rank"])
        bytes_.append(row["router_bytes"])

    close(result["mean_router_minus_exact_static"], np.mean(static))
    close(result["mean_router_minus_tier_shuffle"], np.mean(shuffle))
    close(result["mean_router_delta_loss"], np.mean(delta_full))
    close(result["mean_average_downstream_rank"], np.mean(avg_ranks), atol=1e-8)
    close(result["mean_router_bytes"], np.mean(bytes_), atol=1e-8)
    close(result["router_over_fixed_dense_mla"], np.mean(bytes_) / result["fixed_dense_mla_bytes"])
    close(result["router_minus_exact_static_bootstrap_95pct_ci"], bootstrap(static, result["seed"] + 1))
    close(result["router_minus_tier_shuffle_bootstrap_95pct_ci"], bootstrap(shuffle, result["seed"] + 3))
    close(result["router_minus_static_sign_flip_p_less"], sign_flip(static, result["seed"] + 2))
    close(result["router_minus_shuffle_sign_flip_p_less"], sign_flip(shuffle, result["seed"] + 4))
    if result["router_better_than_static_sequence_count"] != int(np.sum(np.asarray(static) < 0)):
        raise AssertionError("static win count mismatch")
    if result["router_better_than_shuffle_sequence_count"] != int(np.sum(np.asarray(shuffle) < 0)):
        raise AssertionError("shuffle win count mismatch")
    if result["success"] != (result["router_minus_exact_static_bootstrap_95pct_ci"][1] < 0):
        raise AssertionError("success criterion mismatch")

    report = {
        "status": "passed",
        "scale": args.scale,
        "protocol_commit": manifest["protocol_commit"],
        "manifest_sha256": sha(args.manifest),
        "result_sha256": sha(args.result),
        "authenticated_inputs": actual,
        "n_authenticated_prior_starts": len(old_starts),
        "n_fresh_starts": len(starts),
        "minimum_fresh_prior_distance": min(abs(a-b) for a in starts for b in old_starts),
        "minimum_fresh_fresh_distance": int(min(np.diff(starts))),
        "deterministic_sampler_reconstructed": True,
        "authenticated_checkpoint_config": config,
        "arithmetic_recomputed": True,
        "packed_bytes_recomputed_from_histograms": True,
        "exact_sign_flip_p_less_router_vs_static": exact_sign_flip_p_less(static),
        "exact_sign_flip_p_less_router_vs_shuffle": exact_sign_flip_p_less(shuffle),
        "success": result["success"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
