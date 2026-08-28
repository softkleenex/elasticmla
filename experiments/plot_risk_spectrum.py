"""Risk-capacity spectrum and cancellation-diagnostic figure (30M/122M)."""
import json, os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
s30 = json.load(open(ROOT / "experiments/contextual_router_30m/risk_capacity_v5_summary.json"))
s122 = json.load(open(ROOT / "experiments/contextual_router_122m/risk_capacity_v5_summary.json"))
s250 = json.load(open(ROOT / "experiments/contextual_router_250m/risk_capacity_v5_summary.json"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                      "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42})
colors = {"30M": "#2468B4", "122M": "#D55E00", "250M": "#009E73"}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), constrained_layout=True)

ax = axes[0]
for name, s in (("30M", s30), ("122M", s122), ("250M", s250)):
    spectrum = s["risk_capacity_spectrum"]
    tails = sorted((int(k) for k in spectrum), reverse=True)
    alphas = [spectrum[str(k)]["upper_tail_alpha"] for k in tails]
    means = [spectrum[str(k)]["normalized_mean_r_star"] for k in tails]
    ci_lo = [spectrum[str(k)]["sequence_cluster_bootstrap_95pct_ci"][0] / s["d_c"] for k in tails]
    ci_hi = [spectrum[str(k)]["sequence_cluster_bootstrap_95pct_ci"][1] / s["d_c"] for k in tails]
    ax.plot(alphas, means, "-o", color=colors[name], label=name, ms=4)
    ax.fill_between(alphas, ci_lo, ci_hi, color=colors[name], alpha=.15, linewidth=0)
ax.set_xlabel(r"Upper-tail level $\alpha = 1 - k/H$")
ax.set_ylabel("Normalized safe rate $r^*/d_c$")
ax.set_title("(a) Monotone risk-capacity spectrum")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="0.9", lw=.6)

ax = axes[1]
labels = ["Signed mean", "Positive-part mean", "Maximum"]
x = np.arange(len(labels)); width = 0.25
for i, (name, s) in enumerate((("30M", s30), ("122M", s122), ("250M", s250))):
    d = s["mean_safe_rate_positive_part_diagnostics"]
    vals = [d["signed_mean"], d["positive_part_mean"], d["maximum"]]
    ax.bar(x + (i - 1) * width, vals, width, color=colors[name], label=name)
ax.set_xticks(x, labels, rotation=12)
ax.set_ylabel("Loss delta (nat) at the mean-safe rank")
ax.set_title("(b) Cancellation, not rare spikes")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="0.9", lw=.6)
for i, (name, s) in enumerate((("30M", s30), ("122M", s122), ("250M", s250))):
    frac = s["mean_safe_rate_positive_part_diagnostics"]["fraction_with_any_offset_above_epsilon"]
    ax.text(0 + (i - 1) * width, s["mean_safe_rate_positive_part_diagnostics"]["signed_mean"] + .01,
            f"{frac*100:.0f}%", fontsize=6.5, ha="center", color=colors[name])

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"elasticmla_risk_spectrum.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print(FIG / "elasticmla_risk_spectrum.png")
