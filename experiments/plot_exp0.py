
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

summary = json.load(open(os.path.join(RES_DIR, "exp0_summary.json")))
r_star = np.load(os.path.join(RES_DIR, "r_star_per_token.npy"))

r_mean = summary["r_star_mean"]
r_std = summary["r_star_std"]
n_tok = summary["n_tokens_probed"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

hist = summary["r_star_histogram"]
ranks = [int(k) for k in hist.keys()]
counts = [hist[str(r)] for r in ranks]
total = sum(counts)
pct = [c / total * 100 for c in counts]
axes[0].bar([str(r) for r in ranks], pct, color="#4C72B0")
axes[0].set_xlabel("minimal sufficient latent rank r_t* (out of d_c=256)")
axes[0].set_ylabel("% of tokens")
title1 = "Per-token effective rank distribution\n(mean=%.1f, std=%.1f, N=%d)" % (r_mean, r_std, n_tok)
axes[0].set_title(title1)
axes[0].tick_params(axis="x", rotation=45)

by_type = summary["by_token_type_mean_rstar"]
types = list(by_type.keys())
vals = [by_type[t] for t in types]
counts_t = [summary["by_token_type_count"][t] for t in types]
bars = axes[1].bar(types, vals, color="#DD8452")
axes[1].set_ylabel("mean r_t*")
axes[1].set_title("Mean effective rank by crude token type")
for b, c in zip(bars, counts_t):
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 3, "n=%d" % c, ha="center", fontsize=8)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "exp0_rank_variance.png")
plt.savefig(out_path, dpi=150)
print("saved:", out_path)
