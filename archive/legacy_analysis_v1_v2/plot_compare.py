
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

uniform = json.load(open(os.path.join(RES_DIR, "exp0_summary.json")))
layerwise = json.load(open(os.path.join(RES_DIR, "exp0_layerwise_summary.json")))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ranks = [int(k) for k in uniform["r_star_histogram"].keys()]
u_counts = [uniform["r_star_histogram"][str(r)] for r in ranks]
l_counts = [layerwise["r_star_histogram"][str(r)] for r in ranks]
u_pct = [c / sum(u_counts) * 100 for c in u_counts]
l_pct = [c / sum(l_counts) * 100 for c in l_counts]

x_pos = np.arange(len(ranks))
w = 0.38
axes[0].bar(x_pos - w/2, u_pct, width=w, label="uniform (last-layer importance)", color="#4C72B0")
axes[0].bar(x_pos + w/2, l_pct, width=w, label="layer-wise importance", color="#55A868")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([str(r) for r in ranks], rotation=45)
axes[0].set_xlabel("minimal sufficient rank r_t*")
axes[0].set_ylabel("% of tokens")
axes[0].set_title("Uniform vs layer-wise truncation")
axes[0].legend(fontsize=8)

types = list(uniform["by_token_type_mean_rstar"].keys())
u_vals = [uniform["by_token_type_mean_rstar"][t] for t in types]
l_vals = [layerwise["by_token_type_mean_rstar"][t] for t in types]
x2 = np.arange(len(types))
axes[1].bar(x2 - w/2, u_vals, width=w, label="uniform", color="#4C72B0")
axes[1].bar(x2 + w/2, l_vals, width=w, label="layer-wise", color="#55A868")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(types)
axes[1].set_ylabel("mean r_t*")
axes[1].set_title("By token type")
axes[1].legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "exp0_uniform_vs_layerwise.png")
plt.savefig(out_path, dpi=150)
print("saved:", out_path)
