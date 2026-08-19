"""Generate paper-ready figures from committed ElasticMLA result JSONs."""
import json, os
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
scale = json.load(open(ROOT / "experiments/exp0_v4_scale_comparison.json"))
confirm = json.load(open(ROOT / "experiments/fresh_confirmation_comparison.json"))
raw = {
    "30M": json.load(open(ROOT / "experiments/contextual_router_30m/fresh_confirmation.json")),
    "122M": json.load(open(ROOT / "experiments/contextual_router_122m/fresh_confirmation.json")),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
})
colors = {"30M": "#2468B4", "122M": "#D55E00"}
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)

# (a) Corrected capacity sensitivity.
ax = axes[0]
x = np.arange(2); width = 0.34
mean_vals = [100 * scale["models"][s]["future_mean"]["normalized_mean"] for s in ("30M", "122M")]
max_vals = [100 * scale["models"][s]["future_max"]["normalized_mean"] for s in ("30M", "122M")]
ax.bar(x - width/2, mean_vals, width, color="#56B4E9", label="Mean-over-horizon criterion")
ax.bar(x + width/2, max_vals, width, color="#CC79A7", label="Max-over-horizon criterion")
ax.set_xticks(x, ("30M", "122M")); ax.set_ylabel("Required latent rank (% of $d_c$)")
ax.set_ylim(0, 82); ax.set_xlim(-.5, 2.05); ax.legend(frameon=False, loc="center right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=8)
ax.set_title("(a) Capacity sensitivity")

# (b) Fresh paired router-static differences.
ax = axes[1]
for idx, name in enumerate(("30M", "122M")):
    values = np.array([r["router_minus_static"] for r in raw[name]["rows"]])
    rng = np.random.default_rng(700 + idx)
    jitter = rng.uniform(-0.085, 0.085, len(values))
    ax.scatter(idx + jitter, values, s=17, alpha=.60, color=colors[name], edgecolor="none")
    ci = confirm["scales"][name.lower()]["bootstrap_95pct_ci"]
    ax.errorbar(idx, values.mean(), yerr=[[values.mean()-ci[0]], [ci[1]-values.mean()]],
                fmt="D", ms=5, color="black", capsize=4, zorder=5)
ax.axhline(0, color="0.35", lw=1, ls="--")
ax.set_xticks((0, 1), ("30M", "122M")); ax.set_ylabel("Router − exact-byte static loss (nat)")
ax.set_title("(b) Untouched 24-sequence confirmation")
ax.text(.02, .02, "Lower is better", transform=ax.transAxes, fontsize=8, color="0.35")

# (c) Quality-memory operating points.
ax = axes[2]
markers = {"Router": "o", "Exact-byte static": "s", "Tier shuffle": "^"}
ax.scatter(1.0, 0.0, marker="*", s=70, color="0.25", zorder=4)
for name in ("30M", "122M"):
    result = raw[name]; ratio = result["router_over_fixed_dense_mla"]
    router_delta = result["mean_router_delta_loss"]
    values = {
        "Router": router_delta,
        "Exact-byte static": router_delta - result["mean_router_minus_exact_static"],
        "Tier shuffle": router_delta - result["mean_router_minus_tier_shuffle"],
    }
    for label, y in values.items():
        ax.scatter(ratio, y, marker=markers[label], s=42, color=colors[name],
                   edgecolor="white", linewidth=.5)
ax.axhline(.15, color="0.4", ls=":", lw=1, label="+0.15 nat budget")
ax.set_xlim(.55, 1.04); ax.set_ylim(-.015, .36)
ax.set_xlabel("Persistent bytes / fixed-width MLA"); ax.set_ylabel("Loss increase vs full MLA (nat)")
ax.set_title("(c) Fresh quality–storage points")
# Two legends: scale colors/quality budget and marker semantics.
scale_handles = [plt.Line2D([], [], marker="o", linestyle="none", color=colors[name], label=name)
                 for name in ("30M", "122M")]
scale_handles.append(plt.Line2D([], [], linestyle=":", color="0.4", label="+0.15 nat budget"))
leg1 = ax.legend(handles=scale_handles, frameon=False, loc="upper right")
handles = [plt.Line2D([], [], marker="*", markersize=9, linestyle="none", color="0.25", label="Full MLA")]
handles += [plt.Line2D([], [], marker=m, linestyle="none", color="0.25", label=l) for l,m in markers.items()]
ax.add_artist(leg1); ax.legend(handles=handles, frameon=False, loc="center right", bbox_to_anchor=(1,.53))

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=.6, zorder=0)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"elasticmla_main_results.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print(FIG / "elasticmla_main_results.png")
