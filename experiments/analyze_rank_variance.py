
"""
Experiment 0 core analysis: does the MLA KV-latent's *effective rank requirement*
vary meaningfully across tokens?

Method:
  1. Load a trained MLAGPT checkpoint.
  2. Collect the KV-latent (c_KV) activations for a probe layer across many
     validation tokens.
  3. Rank the d_c latent channels by their aggregate variance (proxy for
     "importance", cheap stand-in for a full covariance/SVD analysis).
  4. For each candidate rank budget r in a grid, zero out the least-important
     (d_c - r) channels *uniformly across all layers* and re-run the forward
     pass, computing PER-TOKEN next-token loss (not just the mean).
  5. For each token position, find the minimal r such that the loss increase
     vs. full-rank stays below a tolerance epsilon -> r*_t.
  6. Save the per-token r*_t distribution + histogram, and break it down by
     token frequency (proxy for content vs. function words) and by whether
     the token is punctuation/whitespace-ish (cheap syntactic proxy).
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from elastic_mla import MLAGPT

DATA_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "ckpt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "results")
os.makedirs(OUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)

ckpt = torch.load(os.path.join(CKPT_DIR, "latest.pt"), map_location=device, weights_only=False)
cfg = ckpt["config"]
model = MLAGPT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print("loaded checkpoint at step", ckpt["step"], flush=True)

D_C = cfg["d_c"]
BLOCK_SIZE = cfg["max_len"]
enc = tiktoken.get_encoding("gpt2")

val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")

N_SEQ = 64          # number of validation sequences to probe
RANK_GRID = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
EPS = 0.10          # loss-increase tolerance (nats) to call a rank "sufficient"

rng = np.random.RandomState(0)
starts = rng.randint(0, len(val_data) - BLOCK_SIZE - 1, size=N_SEQ)
batch = np.stack([val_data[s:s+BLOCK_SIZE+1].astype(np.int64) for s in starts])
x = torch.from_numpy(batch[:, :-1]).to(device)
y = torch.from_numpy(batch[:, 1:]).to(device)

# ---- Step 1: collect latent activations (full rank) to rank channels by variance ----
with torch.no_grad():
    logits_full, _, c_kv = model(x, layer_idx_for_latent=cfg["n_layers"] - 1)  # probe last layer
    logits_full_all = model(x)[0]  # full model forward (all layers full rank) for baseline loss

c_kv_flat = c_kv.reshape(-1, D_C)  # (N_SEQ*T, D_C)
channel_var = c_kv_flat.var(dim=0).cpu().numpy()
importance_order = np.argsort(-channel_var)  # descending importance

def per_token_loss(logits, targets):
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    return loss.view(B, T).detach().cpu().numpy()

baseline_loss = per_token_loss(logits_full_all, y)  # (B,T)

results_per_rank = {}
with torch.no_grad():
    for r in RANK_GRID:
        keep_idx = importance_order[:r]
        mask = torch.zeros(D_C, device=device)
        mask[keep_idx] = 1.0
        logits_r, _ = model(x, rank_mask=mask)
        loss_r = per_token_loss(logits_r, y)
        results_per_rank[r] = loss_r
        delta = float(np.mean(loss_r - baseline_loss))
        print(f"rank={r:4d}  mean_loss={loss_r.mean():.4f}  delta_vs_full={delta:.4f}", flush=True)

# ---- Step 2: per-token minimal sufficient rank ----
B, T = baseline_loss.shape
r_star = np.full((B, T), RANK_GRID[-1], dtype=np.int32)
for r in RANK_GRID:
    delta = results_per_rank[r] - baseline_loss
    sufficient = delta < EPS
    still_default = (r_star == RANK_GRID[-1])
    update = sufficient & still_default
    r_star[update] = r

flat_r_star = r_star.flatten()
flat_tokens = x.detach().cpu().numpy().flatten()

# crude token-type proxy: decode single token id -> string, check punctuation/space-ish
def token_type(tid):
    s = enc.decode([int(tid)])
    stripped = s.strip()
    if stripped == "":
        return "space"
    if all(ch in ".,!?;:\'\"-()" for ch in stripped):
        return "punct"
    if stripped[0].isupper():
        return "capitalized"
    return "other"

types = np.array([token_type(t) for t in flat_tokens])

summary = {
    "rank_grid": RANK_GRID,
    "epsilon": EPS,
    "n_tokens_probed": int(len(flat_r_star)),
    "r_star_mean": float(flat_r_star.mean()),
    "r_star_std": float(flat_r_star.std()),
    "r_star_min": int(flat_r_star.min()),
    "r_star_max": int(flat_r_star.max()),
    "r_star_histogram": {int(r): int((flat_r_star == r).sum()) for r in RANK_GRID},
    "by_token_type_mean_rstar": {
        t: float(flat_r_star[types == t].mean()) if (types == t).sum() > 0 else None
        for t in np.unique(types)
    },
    "by_token_type_count": {t: int((types == t).sum()) for t in np.unique(types)},
    "checkpoint_step": ckpt["step"],
    "d_c": D_C,
}

with open(os.path.join(OUT_DIR, "exp0_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

np.save(os.path.join(OUT_DIR, "r_star_per_token.npy"), r_star)
np.save(os.path.join(OUT_DIR, "channel_importance_order.npy"), importance_order)

print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
print("DONE", flush=True)
