"""
Layer-wise channel importance analysis for ElasticMLA.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F
import tiktoken
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT

DATA_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "ckpt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "results")
os.makedirs(OUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

ckpt = torch.load(os.path.join(CKPT_DIR, "latest.pt"), map_location=device, weights_only=False)
cfg = ckpt["config"]
model = MLAGPT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

D_C = cfg["d_c"]
BLOCK_SIZE = cfg["max_len"]
enc = tiktoken.get_encoding("gpt2")

val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
N_SEQ = 64
rng = np.random.RandomState(0)
starts = rng.randint(0, len(val_data) - BLOCK_SIZE - 1, size=N_SEQ)
batch = np.stack([val_data[s:s+BLOCK_SIZE+1].astype(np.int64) for s in starts])
x = torch.from_numpy(batch[:, :-1]).to(device)
y = torch.from_numpy(batch[:, 1:]).to(device)

n_layers = cfg["n_layers"]
# Compute per-layer channel importance order
layer_importance_orders = []
with torch.no_grad():
    for layer_idx in range(n_layers):
        logits_full, _, c_kv = model(x, layer_idx_for_latent=layer_idx)
        c_kv_flat = c_kv.reshape(-1, D_C)
        var = c_kv_flat.var(dim=0).cpu().numpy()
        order = np.argsort(-var)
        layer_importance_orders.append(order)

# Helper to run forward with list of masks (one per layer)
def forward_with_layer_masks(x_tensor, masks):
    h = model.tok_emb(x_tensor)
    h = model.drop(h)
    for i, block in enumerate(model.blocks):
        mask = masks[i] if isinstance(masks, (list, tuple)) else masks
        h = block(h, rank_mask=mask)
    h = model.ln_f(h)
    logits = model.head(h)
    return logits

def per_token_loss(logits, targets):
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    return loss.view(B, T).cpu().numpy()

RANK_GRID = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
EPS = 0.10

# Baseline full-rank loss (no truncation)
with torch.no_grad():
    logits_full_all = model(x)[0]
baseline_loss = per_token_loss(logits_full_all, y)

results_per_rank = {}
with torch.no_grad():
    for r in RANK_GRID:
        masks = []
        for order in layer_importance_orders:
            keep_idx = order[:r]
            mask = torch.zeros(D_C, device=device)
            mask[keep_idx] = 1.0
            masks.append(mask)
        logits_r = forward_with_layer_masks(x, masks)
        loss_r = per_token_loss(logits_r, y)
        results_per_rank[r] = loss_r
        delta = float(np.mean(loss_r - baseline_loss))
        print(f"rank={r:4d}  mean_loss={loss_r.mean():.4f}  delta_vs_full={delta:.4f}")

# Per-token minimal sufficient rank
B, T = baseline_loss.shape
r_star = np.full((B, T), RANK_GRID[-1], dtype=np.int32)
for r in RANK_GRID:
    delta = results_per_rank[r] - baseline_loss
    sufficient = delta < EPS
    r_star[sufficient] = np.minimum(r_star[sufficient], r)

# Summary statistics
epsilon = EPS
r_star_mean = float(r_star.mean())
r_star_std = float(r_star.std())
hist, bin_edges = np.histogram(r_star, bins=np.arange(0, 257, 16))
# Token type breakdown (same crude categories as analyze_rank_variance.py for a fair comparison)
def token_type(tid):
    s = enc.decode([int(tid)])
    stripped = s.strip()
    if stripped == "":
        return "space"
    if all(ch in ".,!?;:'\"-()" for ch in stripped):
        return "punct"
    if stripped[0].isupper():
        return "capitalized"
    return "other"

flat_tokens_in = x.detach().cpu().numpy().flatten()  # use input tokens, matches original script
types = np.array([token_type(t) for t in flat_tokens_in])
flat_r_star = r_star.flatten()
by_token_type_mean_rstar = {
    t: float(flat_r_star[types == t].mean()) if (types == t).sum() > 0 else None
    for t in np.unique(types)
}
by_token_type_count = {t: int((types == t).sum()) for t in np.unique(types)}

# Rank grid aligned histogram (match RANK_GRID buckets instead of fixed-width bins, for comparability)
r_star_histogram = {int(r): int((flat_r_star == r).sum()) for r in RANK_GRID}

summary = {
    "rank_grid": RANK_GRID,
    "epsilon": epsilon,
    "n_tokens_probed": int(flat_r_star.size),
    "r_star_mean": r_star_mean,
    "r_star_std": r_star_std,
    "r_star_min": int(flat_r_star.min()),
    "r_star_max": int(flat_r_star.max()),
    "r_star_histogram": r_star_histogram,
    "by_token_type_mean_rstar": by_token_type_mean_rstar,
    "by_token_type_count": by_token_type_count,
    "checkpoint_step": ckpt["step"],
    "d_c": D_C,
    "method": "layerwise (per-layer independent channel importance + per-layer truncation to same rank budget r)",
}

out_path = os.path.join(OUT_DIR, "exp0_layerwise_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary to", out_path)
