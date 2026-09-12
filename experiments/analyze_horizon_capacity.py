"""Test theory prediction 3 against the risk-capacity summaries' horizon spectrum.

Prediction 3 (`notes/theory_contextual_tail_rate.md`): the worst-offset safe rate
should rise or saturate with a longer matched-prefix horizon, while the signed mean
need not be monotone.

`analyze_risk_capacity_spectrum.py` already emits `horizon_capacity_spectrum`:
for each criterion and each horizon ``h`` it recomputes ``r*`` from the first ``h``
offsets of *the same* probes' future-delta vectors. The horizons are therefore
nested prefixes of one probe population, so the sweep carries no position confound
-- unlike separate runs at different ``--future-horizon``, where a longer horizon
restricts probes to earlier positions.

This script reads summaries only; it runs no model and needs no GPU.
"""
import argparse, json
from pathlib import Path

CRITERIA = ("maximum", "positive_mean", "signed_mean")


def load(path):
    summary = json.loads(Path(path).read_text())
    if "horizon_capacity_spectrum" not in summary:
        raise ValueError(f"{path} has no horizon_capacity_spectrum")
    return summary


def curve(summary, criterion):
    spectrum = summary["horizon_capacity_spectrum"][criterion]
    horizons = sorted(int(h) for h in spectrum)
    d_c = summary["d_c"]
    return horizons, [
        {
            "horizon": h,
            "normalized_r_star": spectrum[str(h)]["normalized_mean_r_star"],
            "mean_r_star": spectrum[str(h)]["mean_r_star"],
            "normalized_ci": [c / d_c for c in spectrum[str(h)]["sequence_cluster_bootstrap_95pct_ci"]],
        }
        for h in horizons
    ]


def verdict(points, tolerance):
    values = [p["normalized_r_star"] for p in points]
    decreases = [
        {"from": points[i]["horizon"], "to": points[i + 1]["horizon"],
         "drop": values[i] - values[i + 1]}
        for i in range(len(values) - 1)
        if values[i + 1] < values[i] - tolerance
    ]
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return {
        "nondecreasing": not decreases,
        "decreases": decreases,
        "step_sizes": steps,
        "steps_are_shrinking": all(
            steps[i + 1] <= steps[i] + tolerance for i in range(len(steps) - 1)
        ),
        "final_step": steps[-1] if steps else None,
        "total_change": values[-1] - values[0],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", action="append", required=True, metavar="LABEL=PATH",
                   help="repeat per scale, e.g. --summary 30m=experiments/.../risk_capacity_v5_summary.json")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="normalized slack before a step counts as a decrease")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    summaries = {}
    for item in args.summary:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--summary expects LABEL=PATH, got {item!r}")
        summaries[label] = load(path)

    report = {"epsilon_nats": {k: v["epsilon_nats"] for k, v in summaries.items()},
              "future_horizon_exact": {k: v["future_horizon_exact"] for k, v in summaries.items()},
              "n_positions_total": {k: v["n_positions_total"] for k, v in summaries.items()},
              "checkpoint_sha256": {k: v["checkpoint_sha256"] for k, v in summaries.items()},
              "criteria": {}}
    for criterion in CRITERIA:
        entry = {}
        for label, summary in summaries.items():
            _, points = curve(summary, criterion)
            entry[label] = {"points": points, "verdict": verdict(points, args.tolerance)}
        report["criteria"][criterion] = entry

    report["prediction_3"] = {
        "statement": ("worst-offset safe rate rises or saturates with a longer "
                      "matched-prefix horizon; signed mean need not be monotone"),
        "maximum_nondecreasing_at_all_scales": all(
            report["criteria"]["maximum"][k]["verdict"]["nondecreasing"] for k in summaries),
        "maximum_saturating_at_all_scales": all(
            report["criteria"]["maximum"][k]["verdict"]["steps_are_shrinking"] for k in summaries),
        "signed_mean_nonmonotone_at_all_scales": all(
            not report["criteria"]["signed_mean"][k]["verdict"]["nondecreasing"] for k in summaries),
    }

    for criterion in CRITERIA:
        print(f"\n=== {criterion}: normalized mean r* (95% sequence-cluster bootstrap CI) ===")
        labels = list(summaries)
        horizons = [p["horizon"] for p in report["criteria"][criterion][labels[0]]["points"]]
        print(f"{'h':>5s} |" + "".join(f"{l:>26s}" for l in labels))
        for idx, h in enumerate(horizons):
            row = f"{h:>5d} |"
            for label in labels:
                pt = report["criteria"][criterion][label]["points"][idx]
                lo, hi = pt["normalized_ci"]
                row += f"  {pt['normalized_r_star']*100:6.2f}% [{lo*100:5.2f},{hi*100:5.2f}]"
            print(row)
        for label in labels:
            v = report["criteria"][criterion][label]["verdict"]
            print(f"   {label:>6s}: nondecreasing={v['nondecreasing']} "
                  f"shrinking_steps={v['steps_are_shrinking']} "
                  f"final_step={v['final_step']*100:+.2f}pp total={v['total_change']*100:+.2f}pp")
    print("\nprediction 3:", json.dumps(report["prediction_3"], indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
