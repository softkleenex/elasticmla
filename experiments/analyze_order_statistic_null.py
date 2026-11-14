"""Order-statistic null for the mean/worst gap (pre-registered analysis R3).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (R3).

Is the worst-offset rate just what "the maximum of 32 dispersed values versus their
mean" implies? From the Table 2 run's raw 250M deltas, each (probe, rank) row of 32
future-offset loss changes is replaced by

* gaussian: 32 i.i.d. draws from N(row mean, row standard deviation^2);
* sign_randomized: row mean + (delta - row mean) with each offset's sign flipped at
  random (keeps the absolute deviations, destroys any sign structure),

both seeded 63521 + probe index. r* is recomputed at every tail count with epsilon = 0.10.
Reading: if the Gaussian null reproduces the k = 1 mean normalized r* within 0.10, the
worst-offset requirement is what dispersion implies by order statistics; if the observed
rate exceeds it by more than 0.10, the worst offsets are heavier-tailed than Gaussian.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = 63_521
TAIL_COUNTS = (32, 16, 8, 4, 2, 1)
EPSILON = 0.10
D_C = 512
WITHIN = 0.10


def load_risk_module():
    spec = importlib.util.spec_from_file_location(
        "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def null_rows(deltas: np.ndarray, probe_index: int) -> dict[str, np.ndarray]:
    """deltas: (ranks, offsets) -> the two null versions of the same shape."""
    rng = np.random.default_rng(SEED_BASE + probe_index)
    mean = deltas.mean(axis=1, keepdims=True)
    std = deltas.std(axis=1, keepdims=True)
    gaussian = mean + std * rng.standard_normal(deltas.shape)
    signs = rng.choice((-1.0, 1.0), size=deltas.shape)
    return {"gaussian": gaussian, "sign_randomized": mean + signs * (deltas - mean)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=ROOT / "kaggle_output/review_reruns/spectrum250/spectrum250_v5/risk_capacity_v5_records.json")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/order_statistic_null.json")
    args = parser.parse_args()
    risk = load_risk_module()

    records = json.loads(args.records.read_text())
    ranks = sorted(int(r) for r in records[0]["future_delta_by_rank"])
    conditions = ("observed", "gaussian", "sign_randomized")
    per_seq = {c: {k: {} for k in TAIL_COUNTS} for c in conditions}
    mismatches = 0
    for index, rec in enumerate(records):
        deltas = np.array([rec["future_delta_by_rank"][str(r)] for r in ranks], dtype=np.float64)
        rows = {"observed": deltas, **null_rows(deltas, index)}
        for condition, values in rows.items():
            for k in TAIL_COUNTS:
                curve = [risk.upper_tail_mean(row, k) for row in values]
                r_star = risk.suffix_all_satisfy_r_star(curve, ranks, EPSILON)
                if condition == "observed" and r_star != int(rec["r_star_by_tail_count"][str(k)]):
                    mismatches += 1
                per_seq[condition][k].setdefault(int(rec["seq"]), []).append(r_star / D_C)
    if mismatches:
        raise SystemExit(f"observed r* not reproduced for {mismatches} entries")

    spectrum = {}
    for c_idx, condition in enumerate(conditions):
        spectrum[condition] = {}
        for j, k in enumerate(TAIL_COUNTS):
            groups = per_seq[condition][k]
            values = [v for vs in groups.values() for v in vs]
            ci = risk.sequence_cluster_bootstrap_mean_ci(
                groups, n_bootstrap=args.bootstrap_draws, seed=SEED_BASE + 100 * c_idx + j)
            spectrum[condition][str(k)] = {"normalized_mean_r_star": float(np.mean(values)),
                                           "sequence_cluster_bootstrap_95pct_ci": [float(x) for x in ci]}

    def observed_minus(condition: str, k: int) -> float:
        return (spectrum["observed"][str(k)]["normalized_mean_r_star"]
                - spectrum[condition][str(k)]["normalized_mean_r_star"])

    gap = observed_minus("gaussian", 1)
    reading = ("within_0.10: worst-offset requirement is what Gaussian dispersion implies"
               if abs(gap) <= WITHIN else
               "observed_exceeds_by_more_than_0.10: worst offsets heavier-tailed than Gaussian"
               if gap > 0 else "observed_below_null_by_more_than_0.10: reported as is")
    output = {
        "status": "complete",
        "analysis": "R3 order-statistic null (pre-registered)",
        "inputs_sha256": {str(args.records.relative_to(ROOT)): sha256(args.records),
                          "analysis_source": sha256(Path(__file__)),
                          "risk_v5_source": sha256(ROOT / "experiments/analyze_risk_capacity_spectrum.py")},
        "epsilon_nats": EPSILON, "tail_counts": list(TAIL_COUNTS), "rank_grid": ranks,
        "n_probes": len(records), "seed_rule": f"{SEED_BASE} + probe index",
        "spectrum": spectrum,
        "observed_minus_null_at_k1": {c: observed_minus(c, 1) for c in ("gaussian", "sign_randomized")},
        "observed_minus_null_at_kH": {c: observed_minus(c, 32) for c in ("gaussian", "sign_randomized")},
        "pre_registered_reading": reading,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for condition in conditions:
        print(condition, " ".join(f"k{k}={spectrum[condition][str(k)]['normalized_mean_r_star']:.3f}"
                                  for k in TAIL_COUNTS))
    print(reading)


if __name__ == "__main__":
    main()
