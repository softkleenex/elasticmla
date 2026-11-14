"""Budget sensitivity of the risk-capacity spectrum (pre-registered analysis R1).

Pre-registration: ``notes/2026-11-14_tmlr_resubmission_plan.md`` (commit 3427fa0).

Recomputes the suffix-safe rate r*_k(epsilon) for epsilon in {0.05, 0.10, 0.20, 0.50}
nat from stored per-offset future-loss deltas -- no model is run:

* 250M: the Table 2 run's raw records (768 probes, H = 32).
* DeepSeek-V2-Lite: the coordinate-basis raw deltas (public-MLA shard a, 384
  probes, H = 32).

Before anything else it checks that epsilon = 0.10 reproduces the stored per-probe r*
(250M) and the published normalized means (DeepSeek); on a mismatch it stops. The
pre-registered reading: the large-gap claim holds at an epsilon when the mean
normalized r* at k = 1 is at least 3x that at k = H.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPSILONS = (0.05, 0.10, 0.20, 0.50)
TAIL_COUNTS = (32, 16, 8, 4, 2, 1)
GAP_RATIO = 3.0
DS_TAIL_KEYS = {32: "signed_mean", 16: "tail_16", 8: "tail_8", 4: "tail_4", 2: "tail_2", 1: "maximum"}


def load_risk_module():
    spec = importlib.util.spec_from_file_location(
        "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(probes, ranks, d_c, risk, bootstrap_draws, seed):
    """probes: list of (sequence_id, deltas[rank, offset]). Returns per-epsilon spectra."""
    out = {}
    for eps in EPSILONS:
        by_k = {}
        for j, k in enumerate(TAIL_COUNTS):
            per_seq: dict[int, list[float]] = {}
            for seq, deltas in probes:
                curve = [risk.upper_tail_mean(row, k) for row in deltas]
                r_star = risk.suffix_all_satisfy_r_star(curve, ranks, eps)
                per_seq.setdefault(seq, []).append(r_star / d_c)
            values = [v for vs in per_seq.values() for v in vs]
            ci = risk.sequence_cluster_bootstrap_mean_ci(
                per_seq, n_bootstrap=bootstrap_draws, seed=seed + 100 * j + int(eps * 1000))
            by_k[str(k)] = {"normalized_mean_r_star": float(np.mean(values)),
                            "sequence_cluster_bootstrap_95pct_ci": [float(c) for c in ci]}
        mean_rate = by_k[str(TAIL_COUNTS[0])]["normalized_mean_r_star"]
        max_rate = by_k["1"]["normalized_mean_r_star"]
        ratio = max_rate / mean_rate if mean_rate > 0 else float("inf")
        out[f"{eps:.2f}"] = {
            "by_tail_count": by_k,
            "tail_capacity_premium": max_rate - mean_rate,
            "max_over_mean_ratio": ratio,
            "large_gap_claim_holds": bool(ratio >= GAP_RATIO),
        }
    return out


def load_250m(path: Path, risk):
    records = json.loads(path.read_text())
    ranks = sorted(int(r) for r in records[0]["future_delta_by_rank"])
    probes, mismatches = [], 0
    for rec in records:
        deltas = np.array([rec["future_delta_by_rank"][str(r)] for r in ranks], dtype=np.float32)
        probes.append((int(rec["seq"]), deltas))
        for k in TAIL_COUNTS:
            curve = [risk.upper_tail_mean(row, k) for row in deltas]
            if risk.suffix_all_satisfy_r_star(curve, ranks, 0.10) != int(rec["r_star_by_tail_count"][str(k)]):
                mismatches += 1
    return ranks, probes, mismatches


def load_deepseek(paths):
    ranks, probes = None, []
    for shard, path in enumerate(paths):
        z = np.load(path, allow_pickle=False)
        grid = [int(r) for r in z["rank_grid"]]
        if ranks is None:
            ranks = grid
        elif grid != ranks:
            raise ValueError("DeepSeek shards disagree on the rank grid")
        delta = z["delta_coord"]  # (sequences, positions, ranks, offsets)
        for s in range(delta.shape[0]):
            for p in range(delta.shape[1]):
                probes.append((shard * 1000 + s, delta[s, p]))
    return ranks, probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-250m", type=Path, default=ROOT / "kaggle_output/review_reruns/spectrum250/spectrum250_v5/risk_capacity_v5_records.json")
    # Only shard a holds the coordinate basis (shard b holds grad/model-metric).
    parser.add_argument("--deepseek-raw", type=Path, nargs="+", default=[
        ROOT / "kaggle_output/review_reruns/publicmla_a/publicmla_a.raw.npz"])
    parser.add_argument("--deepseek-summary", type=Path, default=ROOT / "experiments/public_mla/deepseek_v2_lite_combined.json")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/budget_sensitivity.json")
    args = parser.parse_args()
    risk = load_risk_module()

    ranks_250, probes_250, mismatches = load_250m(args.records_250m, risk)
    if mismatches:
        raise SystemExit(f"250M sanity check failed: {mismatches} stored r* values not reproduced")
    ranks_ds, probes_ds = load_deepseek(args.deepseek_raw)
    published = json.loads(args.deepseek_summary.read_text())["spectra"]["coord"]

    result_250 = spectrum(probes_250, ranks_250, 512, risk, args.bootstrap_draws, 71000)
    result_ds = spectrum(probes_ds, ranks_ds, 512, risk, args.bootstrap_draws, 72000)
    for k, key in DS_TAIL_KEYS.items():
        ours = result_ds["0.10"]["by_tail_count"][str(k)]["normalized_mean_r_star"]
        if abs(ours - published[key]["normalized_mean_r_star"]) > 1e-9:
            raise SystemExit(f"DeepSeek sanity check failed at k={k}: {ours} vs {published[key]['normalized_mean_r_star']}")

    inputs = [args.records_250m, *args.deepseek_raw, args.deepseek_summary, Path(__file__)]
    output = {
        "status": "complete",
        "analysis": "R1 budget sensitivity (pre-registered, notes/2026-11-14_tmlr_resubmission_plan.md)",
        "epsilons_nats": list(EPSILONS),
        "tail_counts": list(TAIL_COUNTS),
        "future_horizon_exact": 32,
        "gap_criterion": f"max/mean normalized r* >= {GAP_RATIO}",
        "sanity_checks": {"250m_stored_r_star_mismatches_at_eps_0.10": mismatches,
                          "deepseek_published_means_reproduced_at_eps_0.10": True},
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "n_probes": {"250m": len(probes_250), "deepseek_v2_lite": len(probes_ds)},
        "results": {"250m": result_250, "deepseek_v2_lite_coord": result_ds},
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, res in output["results"].items():
        print(name)
        for eps, r in res.items():
            rates = " ".join(f"k{k}={r['by_tail_count'][str(k)]['normalized_mean_r_star']:.3f}" for k in TAIL_COUNTS)
            print(f"  eps={eps}: {rates}  TCP={r['tail_capacity_premium']:.3f}  ratio={r['max_over_mean_ratio']:.2f}  holds={r['large_gap_claim_holds']}")


if __name__ == "__main__":
    main()
