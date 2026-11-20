"""Post-hoc descriptive statistics requested by the 2026-11-20 internal review (R10a).

NOT pre-registered. Everything here is computed offline from records that already exist and
is reported as descriptive; no threshold or reading is attached. Logged in
``notes/2026-11-14_tmlr_resubmission_plan.md`` (R10).

From the Table 2 250M records (H = 32) and the extended-horizon 250M records (H = 128):

1. grid_floor: the fraction of probes whose r* sits at the smallest tested rank, per tail
   count (also for 30M/122M from their v4 records, mean and max only).
2. horizon: mean-safe (k = H') and worst-offset-safe (k = 1) mean normalized r* when only the
   first H' offsets are kept, H' in {4, 8, 16, 32} (from H = 32 records) and
   {32, 64, 128} (from H = 128 records), with the k = 1 / k = H' ratio.
3. worst_offset_location: histogram of argmax_j delta_j at each probe's own mean-safe rank
   (the allocation used in Section 5.4).
4. per_probe_ratio: quantiles of r*_1 / r*_H over probes.
5. order_statistic_paired: observed minus Gaussian-null normalized r* at every tail count, with
   a sequence-cluster bootstrap CI on the paired per-probe difference (the null rows are
   regenerated with the R3 seed rule, so they are identical to R3's).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 0.10
BOOTSTRAP_SEED = 71_000
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


risk = load_module("risk_v5", "analyze_risk_capacity_spectrum.py")
order_null = load_module("order_statistic_null", "analyze_order_statistic_null.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_deltas(path: Path) -> tuple[list[int], np.ndarray, np.ndarray, list[dict]]:
    """-> ranks, deltas (probes, ranks, offsets), sequence ids, raw records."""
    records = json.loads(path.read_text())
    ranks = sorted(int(r) for r in records[0]["future_delta_by_rank"])
    deltas = np.array([[rec["future_delta_by_rank"][str(r)] for r in ranks] for rec in records],
                      dtype=np.float64)
    seqs = np.array([int(rec["seq"]) for rec in records])
    return ranks, deltas, seqs, records


def r_star(rows: np.ndarray, ranks: list[int], k: int) -> int:
    curve = [risk.upper_tail_mean(row, k) for row in rows]
    return risk.suffix_all_satisfy_r_star(curve, ranks, EPSILON)


def cluster_ci(values: np.ndarray, seqs: np.ndarray, draws: int, seed: int) -> list[float]:
    groups: dict[int, list[float]] = {}
    for v, s in zip(values, seqs):
        groups.setdefault(int(s), []).append(float(v))
    return [float(c) for c in risk.sequence_cluster_bootstrap_mean_ci(groups, n_bootstrap=draws, seed=seed)]


def horizon_table(ranks, deltas, seqs, horizons, d_c, draws, seed) -> dict:
    out = {}
    for i, h in enumerate(horizons):
        cut = deltas[:, :, :h]
        mean_safe = np.array([r_star(p, ranks, h) for p in cut]) / d_c
        worst_safe = np.array([r_star(p, ranks, 1) for p in cut]) / d_c
        out[str(h)] = {
            "mean_safe": {"mean": float(mean_safe.mean()),
                          "ci": cluster_ci(mean_safe, seqs, draws, seed + 2 * i)},
            "worst_offset_safe": {"mean": float(worst_safe.mean()),
                                  "ci": cluster_ci(worst_safe, seqs, draws, seed + 2 * i + 1)},
            "ratio": float(worst_safe.mean() / mean_safe.mean()),
            "fraction_mean_safe_at_grid_floor": float(np.mean(mean_safe == ranks[0] / d_c)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-h32", type=Path, default=ROOT / "kaggle_output/review_reruns/spectrum250/spectrum250_v5/risk_capacity_v5_records.json")
    parser.add_argument("--records-h128", type=Path, default=ROOT / "kaggle_output/review_reruns/spectrum250/spectrum250_A_extended/risk_capacity_v5_records.json")
    parser.add_argument("--v4-records", type=Path, nargs="*", default=[
        ROOT / "experiments/exp0_rank_variance/results_v4/exp0_v4_records.json",
        ROOT / "experiments/exp1_rank_variance_122m/results_v4/exp0_v4_records.json",
        ROOT / "experiments/exp2_rank_variance_250m/results_v4/exp0_v4_records.json"])
    parser.add_argument("--d-c", type=int, default=512)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/review_descriptives.json")
    args = parser.parse_args()

    ranks, deltas, seqs, records = load_deltas(args.records_h32)
    tail_counts = order_null.TAIL_COUNTS
    observed = {k: np.array([r_star(p, ranks, k) for p in deltas]) for k in tail_counts}
    for k in tail_counts:  # stop rule: must reproduce the stored r* exactly
        stored = np.array([int(rec["r_star_by_tail_count"][str(k)]) for rec in records])
        if not np.array_equal(stored, observed[k]):
            raise SystemExit(f"stored r* not reproduced at k={k}")

    # 1. grid floor
    grid_floor = {"250m_h32": {str(k): float(np.mean(observed[k] == ranks[0])) for k in tail_counts}}
    for path in args.v4_records:
        v4 = json.loads(path.read_text())
        floor = min(int(r) for r in v4[0]["future_mean_delta_by_rank"])
        grid_floor[str(path.relative_to(ROOT))] = {
            "grid_floor_rank": floor,
            "mean_criterion": float(np.mean([int(r["r_star_future_mean"]) == floor for r in v4])),
            "max_criterion": float(np.mean([int(r["r_star_future_max"]) == floor for r in v4]))}

    # 2. horizon dependence
    horizon = {"from_h32_records": horizon_table(ranks, deltas, seqs, (4, 8, 16, 32), args.d_c,
                                                 args.bootstrap_draws, BOOTSTRAP_SEED)}
    ranks128, deltas128, seqs128, _ = load_deltas(args.records_h128)
    horizon["from_h128_records"] = horizon_table(ranks128, deltas128, seqs128, (32, 64, 128), args.d_c,
                                                 args.bootstrap_draws, BOOTSTRAP_SEED + 100)

    # 3. location of the worst offset at each probe's own mean-safe rank
    rank_index = {r: i for i, r in enumerate(ranks)}
    argmax = np.array([int(np.argmax(deltas[p, rank_index[int(observed[32][p])]]))
                       for p in range(len(deltas))])
    hist = np.bincount(argmax, minlength=deltas.shape[2])
    location = {"offset_histogram_j1_first": hist.tolist(),
                "fraction_j1": float(hist[0] / hist.sum()),
                "fraction_j_le_4": float(hist[:4].sum() / hist.sum()),
                "uniform_share_per_offset": 1.0 / deltas.shape[2]}

    # 4. per-probe ratio
    ratio = observed[1] / observed[32]
    per_probe = {f"q{int(q * 100)}": float(np.quantile(ratio, q)) for q in QUANTILES}

    # 5. paired order-statistic null
    null_rstar = {k: np.empty(len(deltas)) for k in tail_counts}
    for i, probe in enumerate(deltas):
        gaussian = order_null.null_rows(probe, i)["gaussian"]
        for k in tail_counts:
            null_rstar[k][i] = r_star(gaussian, ranks, k)
    paired = {}
    for j, k in enumerate(tail_counts):
        diff = (observed[k] - null_rstar[k]) / args.d_c
        paired[str(k)] = {"observed": float(observed[k].mean() / args.d_c),
                          "gaussian_null": float(null_rstar[k].mean() / args.d_c),
                          "observed_minus_null": float(diff.mean()),
                          "ci": cluster_ci(diff, seqs, args.bootstrap_draws, BOOTSTRAP_SEED + 200 + j)}

    output = {
        "status": "complete",
        "analysis": "R10a post-hoc descriptives (not pre-registered)",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in [args.records_h32, args.records_h128, *args.v4_records]},
        "analysis_source_sha256": sha256(Path(__file__)),
        "epsilon_nats": EPSILON, "rank_grid_250m": ranks, "n_probes": len(deltas),
        "grid_floor_fraction": grid_floor, "horizon": horizon,
        "worst_offset_location_at_mean_safe_rank": location,
        "per_probe_ratio_k1_over_kH": per_probe,
        "order_statistic_paired": paired,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("grid_floor_fraction", "per_probe_ratio_k1_over_kH")}, indent=1))
    for h, row in {**horizon["from_h32_records"], **{f"{h}*": v for h, v in horizon["from_h128_records"].items()}}.items():
        print(f"H={h}: mean-safe {row['mean_safe']['mean']:.3f} worst {row['worst_offset_safe']['mean']:.3f} "
              f"ratio {row['ratio']:.2f} floor {row['fraction_mean_safe_at_grid_floor']:.2f}")
    print("location", location["fraction_j1"], location["fraction_j_le_4"], hist[:8].tolist())
    for k, row in paired.items():
        print(f"k={k} obs {row['observed']:.3f} null {row['gaussian_null']:.3f} diff {row['observed_minus_null']:+.3f} {row['ci']}")


if __name__ == "__main__":
    main()
