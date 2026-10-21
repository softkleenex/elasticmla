"""Pair the bases of several `analyze_public_mla_spectrum.py` runs on the same probes.

The four bases did not fit one Kaggle session, so they were measured in two runs
(``coord``/``pca`` and ``grad``/``model_metric``). Both runs sample the same evaluation
windows and probe positions from fixed seeds, so every basis is still measured on
identical probes and the paired per-probe comparison of `analyze_basis_dependence.py`
applies across runs -- provided that is checked rather than assumed. This script
refuses to combine unless each run's raw ``.npz`` matches the ``raw_npz_sha256`` its JSON
records, and the runs agree bit-for-bit on evaluation tokens, probe positions, rank grid
and unmodified baseline losses, and on every model, data and source hash.

The calibration pass is *not* required to match: its backward runs in float16 on GPU,
whose reductions are not deterministic, so the saliency hash differs between runs.
Each basis was therefore built from its own run's calibration statistics (same tokens,
same model); this is reported, not hidden.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_RISK = _load("risk_v5", "analyze_risk_capacity_spectrum.py")
_BASIS = _load("basis_p7", "analyze_basis_dependence.py")

SHARED_ARRAYS = ("eval_tokens", "positions", "rank_grid", "baseline")
SHARED_PROVENANCE = ("model_commit", "weight_shard_sha256", "dataset", "dataset_fingerprints",
                     "calibration_tokens_sha256", "evaluation_tokens_sha256",
                     "analysis_source_sha256", "risk_v5_source_sha256", "basis_source_sha256",
                     "versions")
SHARED_SETTINGS = ("dtype", "d_c", "n_layers", "block_size_with_bos", "rank_grid", "epsilon_nats",
                   "future_horizon_exact", "tail_counts", "n_calibration_sequences",
                   "n_evaluation_sequences", "positions_per_sequence", "n_positions_total",
                   "bootstrap_draws", "baseline_mean_token_loss")


def load_run(path):
    """A run's JSON and raw arrays, with the raw file authenticated against the JSON."""
    run = json.loads(path.read_text())
    raw_path = path.with_suffix(".raw.npz")
    digest = _RISK.file_sha256(raw_path)
    if digest != run["provenance"]["raw_npz_sha256"]:
        raise SystemExit(f"{raw_path}: sha256 {digest} does not match its JSON")
    return run, np.load(raw_path), digest


def check_same_probes(runs):
    """Raise unless every run measured the same probes on the same model and data."""
    (first, first_raw, _), *rest = runs
    for run, raw, _ in rest:
        for key in SHARED_ARRAYS:
            if not np.array_equal(first_raw[key], raw[key]):
                raise SystemExit(f"runs differ in raw array {key!r}")
        for key in SHARED_PROVENANCE:
            if first["provenance"][key] != run["provenance"][key]:
                raise SystemExit(f"runs differ in provenance {key!r}")
        for key in SHARED_SETTINGS:
            if first[key] != run[key]:
                raise SystemExit(f"runs differ in setting {key!r}")


def mean_safe_diagnostics(result, deltas, rank_grid, epsilon):
    """Section 5.1's cancellation diagnostic at each probe's mean-safe rank.

    Same quantities as `analyze_risk_capacity_spectrum.py`: at the rank the signed-mean
    criterion accepts, the positive-part mean of the horizon deltas against the signed
    mean, and the fraction of probes with any offset above ``epsilon``.
    """
    column = {int(rank): i for i, rank in enumerate(rank_grid)}
    rows = []
    for seq, ranks in result["signed_mean"]["r_star_by_sequence"].items():
        for position, rank in enumerate(ranks):
            rows.append(deltas[int(seq), position, column[int(rank)]].astype(np.float64))
    delta = np.stack(rows)
    signed, positive = delta.mean(axis=1), np.maximum(delta, 0).mean(axis=1)
    return {
        "signed_mean": float(signed.mean()),
        "positive_part_mean": float(positive.mean()),
        "positive_to_signed_ratio": float(positive.mean() / signed.mean()),
        "fraction_with_any_offset_above_epsilon": float(np.mean((delta > epsilon).any(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="run JSONs, each with its .raw.npz alongside")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_run(path) for path in args.runs]
    check_same_probes(runs)
    first = runs[0][0]
    results, diagnostics = {}, {}
    for run, raw, _ in runs:
        for name, result in run["results_by_basis"].items():
            if name in results:
                raise SystemExit(f"basis {name!r} appears in more than one run")
            results[name] = result
            diagnostics[name] = mean_safe_diagnostics(result, raw[f"delta_{name}"],
                                                      raw["rank_grid"], run["epsilon_nats"])
    d_c, draws = first["d_c"], first["bootstrap_draws"]
    criteria = list(first["tail_counts"])

    output = {
        "status": "combined",
        "inputs": [
            {"path": str(path), "sha256": _RISK.file_sha256(path), "raw_npz_sha256": digest,
             "bases": run["bases_measured"], "saliency_sha256": run["saliency_sha256"],
             "full_rank_checks": run["full_rank_checks"]}
            for path, (run, _, digest) in zip(args.runs, runs)
        ],
        "combine_source_sha256": _RISK.file_sha256(Path(__file__)),
        "shared_provenance": {key: first["provenance"][key] for key in SHARED_PROVENANCE},
        "shared_settings": {key: first[key] for key in SHARED_SETTINGS},
        "spectra": {
            name: {criterion: {"normalized_mean_r_star": result[criterion]["normalized_mean_r_star"],
                               "normalized_ci": result[criterion]["normalized_ci"]}
                   for criterion in criteria}
            | {"tail_capacity_premium": result["tail_capacity_premium"]}
            for name, result in results.items()
        },
        "mean_safe_diagnostics": diagnostics,
        "paired_comparisons": {
            criterion: {
                f"{a}_minus_{b}": _BASIS.paired_difference(
                    results[a][criterion], results[b][criterion], d_c, draws,
                    _BASIS.comparison_seed(a, b, criterion))
                for a, b in _BASIS.comparison_pairs(results)
            }
            for criterion in criteria
        },
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["mean_safe_diagnostics"], indent=1))


if __name__ == "__main__":
    main()
