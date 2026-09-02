"""
Experiment 0 v2: corrected methodology for measuring per-token effective KV-latent
rank requirements in ElasticMLA.

This is a from-scratch redesign of analyze_rank_variance.py / _layerwise.py that
fixes the P1 methodology bugs flagged in codex review:

  P1-a) "Per-token r*" from a *global* rank mask does not measure a per-token
        quantity. A single (d_c,) mask applied to every position and every layer
        entangles the loss at position t with the KV-truncation of the entire
        causal prefix [0..t], not with token t's own KV latent. Fix: mask only
        ONE sequence position i at a time (mask shape (T, d_c) that is all-ones
        except row i), leaving every other position's KV latent at full rank, and
        measure how *that single position's* truncation affects downstream loss.

  P1-b) Raw latent variance is not a valid importance proxy because it is not
        invariant to the linear reparametrization  c_kv[:, c] *= a ; W_UK[:, c] /= a
        (and W_UV symmetric), which leaves the network's function unchanged but can
        make variance rank arbitrarily. Fix: use a first-order Taylor / saliency
        score  |dL/d c_kv[:,:,c] * c_kv[:,:,c]|  computed with one backward pass.
        This "gradient x activation" saliency is far less sensitive to this kind of
        rescaling because it approximates the actual change in loss from ablating
        the channel, not just how much the channel's raw coordinate wiggles.

  P1-c) model.forward(rank_mask=..., layer_idx_for_latent=...) used to silently
        drop rank_mask on every layer except the probed one. Fixed in
        code/elastic_mla/model.py (rank_mask is now applied to every layer
        regardless of layer_idx_for_latent, which only controls which layer's
        latent activation is returned to the caller).

Also addressed (P2):
  - y-based (not x-based) token type labeling: loss at position t predicts y[:,t],
    so the "content vs. function word" label must be computed on y[:,t], not x[:,t].
  - calibration/evaluation split: channel-importance (saliency, variance) is
    estimated on a disjoint calibration set of sequences; r* is measured only on a
    held-out evaluation set.
  - CUDA is included as a device candidate (cuda > mps > cpu), so this also runs
    unmodified on a 4090 / Kaggle GPU.
  - numpy RNG is explicitly seeded (in addition to torch).
  - the large `logits` tensors are deleted (+ device cache emptied) as soon as we
    are done with them, since they are (B, T, vocab=50257) and dominate memory.

WHAT "r*_t" MEANS HERE (READ THIS BEFORE INTERPRETING RESULTS)
----------------------------------------------------------------
For a chosen sequence position i, we truncate ONLY c_kv at position i (in every
layer, since d_c-channel importance is shared/global here -- see NOTE below) to
rank r, run the model, and record two different, deliberately separated effects:

  1. "self effect" -- the change in loss at position i itself, i.e. loss for
     predicting y[:, i] from x[:, i]. Position i's *query* (c_Q, used to compute
     the attention query at position i) is NOT touched by rank_mask at all: c_Q is
     derived from x through W_DQ/W_UQ, entirely independent of c_kv. HOWEVER this
     does not make the self effect zero: the causal mask in this codebase is
     `triu(..., diagonal=1)`, i.e. position i is allowed to attend to itself
     (j <= i, not just j < i). That means position i can attend to its own,
     truncated, key/value pair, so predicting y[:, i] can still be affected by
     truncating c_kv[i]. We report this self effect explicitly instead of assuming
     it is (or must be) zero -- it typically is small but not exactly zero.
  2. "future effect" -- the mean loss increase, over all positions t > i, from
     truncating position i's KV latent. This is the quantity that actually
     measures "how much rank does token i's key/value need, as something *other*
     tokens attend back to" -- i.e., a genuine per-token/per-position effective
     rank requirement, unconfounded by what happens to any other position's KV.

  r*_i (the number we report as "the effective rank of token i") is defined via
  the FUTURE effect: the smallest r in RANK_GRID such that the mean future loss
  increase from truncating only position i to rank r stays below EPS nats. This
  is the fix for P1-a above -- every other position keeps its full d_c-dim latent
  while we probe position i, so nothing about the rest of the causal prefix is
  confounded into r*_i.

NOTE on "global" channel importance: we still share ONE channel-importance
ordering (from the probe layer, saliency- or variance-based) across ALL layers
and ALL probed positions, exactly like v1. This part of the v1 design is kept
deliberately, because it is orthogonal to the two P1 bugs above: which *channels*
are called "important" (this section) is a separate question from which
*positions* the resulting mask is applied to (fixed above) and from what that
importance is *measured with* (fixed above, variance -> saliency).
"""
import os, sys, json, gc
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

# ---------------------------------------------------------------------------
# Reproducibility: seed both torch and numpy RNGs explicitly (P2 fix).
# ---------------------------------------------------------------------------
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Device selection: cuda > mps > cpu (P2 fix -- cuda was missing before, so
# this script could not run on a 4090 / Kaggle GPU box).
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device:", device, flush=True)


def empty_cache():
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


ckpt = torch.load(os.path.join(CKPT_DIR, "latest.pt"), map_location=device, weights_only=False)
cfg = ckpt["config"]
model = MLAGPT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print("loaded checkpoint at step", ckpt["step"], flush=True)

D_C = cfg["d_c"]
N_LAYERS = cfg["n_layers"]
BLOCK_SIZE = cfg["max_len"]
PROBE_LAYER = N_LAYERS - 1  # same probe layer as v1, for comparability
enc = tiktoken.get_encoding("gpt2")

val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
n_tokens = len(val_data)

# ---------------------------------------------------------------------------
# Calibration / evaluation split (P2 fix): contiguous, non-overlapping regions
# of the validation stream, each with its own RNG, so that channel-importance
# estimation (calibration) and r* measurement (evaluation) never touch the
# same tokens.
# ---------------------------------------------------------------------------
split_point = int(n_tokens * 0.6)
calib_region = (0, split_point - BLOCK_SIZE - 1)
eval_region = (split_point, n_tokens - BLOCK_SIZE - 1)

N_SEQ_CALIB = 48
N_SEQ_EVAL = 24
POS_PER_SEQ = 32     # sampled probe positions per eval sequence (P1-a fix makes
                      # per-position forward passes expensive, so we subsample
                      # instead of probing every position)
MIN_FUTURE = 8        # require at least this many future tokens after position i
RANK_GRID = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
EPS = 0.10             # mean future-loss-increase tolerance (nats), same units/
                       # value as v1's per-token epsilon for rough comparability

rng_calib = np.random.RandomState(SEED + 1)
rng_eval = np.random.RandomState(SEED + 2)

calib_starts = rng_calib.randint(calib_region[0], calib_region[1], size=N_SEQ_CALIB)
eval_starts = rng_eval.randint(eval_region[0], eval_region[1], size=N_SEQ_EVAL)

calib_batch = np.stack([val_data[s:s + BLOCK_SIZE + 1].astype(np.int64) for s in calib_starts])
eval_batch = np.stack([val_data[s:s + BLOCK_SIZE + 1].astype(np.int64) for s in eval_starts])

x_cal = torch.from_numpy(calib_batch[:, :-1]).to(device)
y_cal = torch.from_numpy(calib_batch[:, 1:]).to(device)
x_eval = torch.from_numpy(eval_batch[:, :-1]).to(device)
y_eval = torch.from_numpy(eval_batch[:, 1:]).to(device)

assert calib_region[1] + BLOCK_SIZE < eval_region[0] or eval_region[0] > calib_region[1], \
    "calibration/evaluation token regions must not overlap"


def per_token_loss(logits, targets):
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    return loss.view(B, T).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Sanity check for the model.py P1-c fix: rank_mask must now apply to ALL
# layers even when layer_idx_for_latent is also requested. We check this by
# comparing a forward with rank_mask+layer_idx_for_latent against an
# equivalent forward using only rank_mask (which is known to already apply the
# mask to every layer).
# ---------------------------------------------------------------------------
with torch.no_grad():
    probe_mask = torch.zeros(D_C, device=device)
    probe_mask[: D_C // 2] = 1.0
    logits_a, _ = model(x_cal[:2], rank_mask=probe_mask)
    logits_b, _, _ = model(x_cal[:2], rank_mask=probe_mask, layer_idx_for_latent=PROBE_LAYER)
    max_abs_diff = (logits_a - logits_b).abs().max().item()
    assert max_abs_diff < 1e-5, (
        "rank_mask + layer_idx_for_latent regression check FAILED: "
        f"max abs diff = {max_abs_diff} (expected ~0). "
        "rank_mask is not being applied uniformly across all layers."
    )
    print(f"[sanity check] rank_mask now applies uniformly regardless of "
          f"layer_idx_for_latent (max abs logit diff = {max_abs_diff:.2e}). OK.", flush=True)
    del logits_a, logits_b
    empty_cache()

# ---------------------------------------------------------------------------
# Step 1: channel importance on the CALIBRATION set only.
#   (a) saliency = |dL/dc_kv * c_kv| summed over calibration tokens (P1-b fix)
#   (b) raw variance, kept around only so we can report how different the two
#       rankings are (rank correlation) -- NOT used for the actual r* analysis.
# ---------------------------------------------------------------------------
model.zero_grad(set_to_none=True)
logits_cal, loss_cal, c_kv_cal = model(x_cal, targets=y_cal, layer_idx_for_latent=PROBE_LAYER)
c_kv_cal.retain_grad()
loss_cal.backward()
grad_cal = c_kv_cal.grad.detach()  # (N_SEQ_CALIB, T, D_C)
saliency = (grad_cal * c_kv_cal.detach()).abs().sum(dim=(0, 1)).cpu().numpy()  # (D_C,)
variance = c_kv_cal.detach().reshape(-1, D_C).var(dim=0).cpu().numpy()  # (D_C,)
model.zero_grad(set_to_none=True)
del logits_cal, loss_cal, c_kv_cal, grad_cal
empty_cache()

importance_order_saliency = np.argsort(-saliency)
importance_order_variance = np.argsort(-variance)


def spearman_rho(order_a, order_b, d):
    # manual Spearman rank correlation (avoids a hard scipy dependency)
    rank_a = np.empty(d, dtype=np.float64)
    rank_b = np.empty(d, dtype=np.float64)
    rank_a[order_a] = np.arange(d)
    rank_b[order_b] = np.arange(d)
    diff2 = np.sum((rank_a - rank_b) ** 2)
    return 1.0 - (6.0 * diff2) / (d * (d ** 2 - 1))


saliency_vs_variance_spearman = float(spearman_rho(importance_order_saliency, importance_order_variance, D_C))
top32_overlap = float(len(set(importance_order_saliency[:32]) & set(importance_order_variance[:32])) / 32.0)
print(f"[channel importance] saliency vs. variance Spearman rho = "
      f"{saliency_vs_variance_spearman:.4f}, top-32 overlap = {top32_overlap:.3f}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: baseline (full-rank) per-token loss on the EVALUATION set.
# ---------------------------------------------------------------------------
with torch.no_grad():
    logits_full_eval, _ = model(x_eval)
    baseline_loss = per_token_loss(logits_full_eval, y_eval)  # (N_SEQ_EVAL, T)
del logits_full_eval
empty_cache()

B_eval, T_eval = baseline_loss.shape

# ---------------------------------------------------------------------------
# Step 3: sample probe positions per evaluation sequence.
# ---------------------------------------------------------------------------
valid_hi = T_eval - 1 - MIN_FUTURE
pos_per_seq = []
for b in range(B_eval):
    positions = rng_eval.choice(valid_hi + 1, size=min(POS_PER_SEQ, valid_hi + 1), replace=False)
    positions.sort()
    pos_per_seq.append(positions)

# ---------------------------------------------------------------------------
# Step 4: per-(sequence, position, rank) truncated forward passes.
# For a fixed r, we batch ALL sampled positions of one sequence together: the
# mask has shape (K, T, D_C) where K = number of probed positions in that
# sequence, row `pos_k` of mask k is the rank-r channel mask and every other
# row is all-ones (full rank) -- this is the actual per-token isolation fix
# (P1-a). Everything else about the sequence is untouched.
# ---------------------------------------------------------------------------
# self_delta[b][pos] -> {r: delta}, future_delta[b][pos] -> {r: mean delta}
self_delta = {b: {} for b in range(B_eval)}
future_delta = {b: {} for b in range(B_eval)}
future_delta_max = {b: {} for b in range(B_eval)}

for r in RANK_GRID:
    keep_idx = importance_order_saliency[:r]
    chan_mask = torch.zeros(D_C, device=device)
    chan_mask[keep_idx] = 1.0

    for b in range(B_eval):
        positions = pos_per_seq[b]
        K = len(positions)
        x_b = x_eval[b:b + 1]
        y_b = y_eval[b:b + 1]
        x_rep = x_b.expand(K, -1)
        mask_batch = torch.ones(K, T_eval, D_C, device=device)
        for k, pos in enumerate(positions):
            mask_batch[k, pos, :] = chan_mask

        with torch.no_grad():
            logits_r, _ = model(x_rep, rank_mask=mask_batch)
            loss_r = per_token_loss(logits_r, y_b.expand(K, -1))  # (K, T)
        del logits_r

        base_b = baseline_loss[b]
        for k, pos in enumerate(positions):
            pos = int(pos)
            self_d = float(loss_r[k, pos] - base_b[pos])
            if pos + 1 < T_eval:
                fut_diff = loss_r[k, pos + 1:] - base_b[pos + 1:]
                future_d = float(np.mean(fut_diff))
                future_d_max = float(np.max(fut_diff))
            else:
                future_d = 0.0
                future_d_max = 0.0
            self_delta[b].setdefault(pos, {})[r] = self_d
            future_delta[b].setdefault(pos, {})[r] = future_d
            future_delta_max[b].setdefault(pos, {})[r] = future_d_max

    empty_cache()
    print(f"rank={r:4d}  done ({B_eval} sequences x up to {POS_PER_SEQ} positions)", flush=True)

# ---------------------------------------------------------------------------
# Step 5: r*_i from the FUTURE effect only (this is the corrected, genuinely
# per-token quantity -- see module docstring).
# ---------------------------------------------------------------------------
records = []  # one dict per probed (sequence, position)
for b in range(B_eval):
    for pos, r_to_future in future_delta[b].items():
        r_star = RANK_GRID[-1]
        for r in RANK_GRID:
            if r_to_future[r] < EPS:
                r_star = r
                break
        r_to_future_max = future_delta_max[b][pos]
        r_star_max = RANK_GRID[-1]
        for r in RANK_GRID:
            if r_to_future_max[r] < EPS:
                r_star_max = r
                break
        target_token = int(y_eval[b, pos].item())
        records.append({
            "seq": b,
            "pos": pos,
            "target_token_id": target_token,
            "r_star_future": r_star,
            "r_star_future_maxagg": r_star_max,
            "self_delta_at_r_star": self_delta[b][pos][r_star],
            "future_delta_at_r_star": r_to_future[r_star],
            "future_delta_max_at_r_star_maxagg": r_to_future_max[r_star_max],
            "self_delta_by_rank": self_delta[b][pos],
            "future_delta_by_rank": r_to_future,
            "future_delta_max_by_rank": r_to_future_max,
        })

r_star_arr = np.array([rec["r_star_future"] for rec in records], dtype=np.int32)
r_star_arr_maxagg = np.array([rec["r_star_future_maxagg"] for rec in records], dtype=np.int32)
self_delta_at_full_trunc = np.array([rec["self_delta_by_rank"][RANK_GRID[0]] for rec in records])


# token type proxy, computed on the TARGET token y[:, pos] (P2 fix: was on x before)
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


types = np.array([token_type(rec["target_token_id"]) for rec in records])

by_token_type_mean_rstar = {
    t: float(r_star_arr[types == t].mean()) if (types == t).sum() > 0 else None
    for t in np.unique(types)
}
by_token_type_mean_rstar_maxagg = {
    t: float(r_star_arr_maxagg[types == t].mean()) if (types == t).sum() > 0 else None
    for t in np.unique(types)
}
by_token_type_count = {t: int((types == t).sum()) for t in np.unique(types)}

summary = {
    "method": "per-token (single-position) KV-latent truncation on a held-out "
              "evaluation set, using saliency-based (calibration-set) channel "
              "importance; see module docstring for full methodology notes.",
    "seed": SEED,
    "device": device,
    "rank_grid": RANK_GRID,
    "epsilon_future_mean_nats": EPS,
    "probe_layer": PROBE_LAYER,
    "n_layers": N_LAYERS,
    "d_c": D_C,
    "checkpoint_step": ckpt["step"],
    "n_calib_sequences": N_SEQ_CALIB,
    "n_eval_sequences": N_SEQ_EVAL,
    "positions_probed_per_sequence": POS_PER_SEQ,
    "min_future_tokens_required": MIN_FUTURE,
    "n_positions_probed_total": int(len(records)),
    "channel_importance": {
        "saliency_vs_variance_spearman_rho": saliency_vs_variance_spearman,
        "saliency_vs_variance_top32_overlap_frac": top32_overlap,
    },
    "r_star_future_mean": float(r_star_arr.mean()),
    "r_star_future_std": float(r_star_arr.std()),
    "r_star_future_min": int(r_star_arr.min()),
    "r_star_future_max": int(r_star_arr.max()),
    "r_star_future_histogram": {int(r): int((r_star_arr == r).sum()) for r in RANK_GRID},
    "by_token_type_mean_rstar_future": by_token_type_mean_rstar,
    "by_token_type_count": by_token_type_count,
    "maxagg_diagnostic": {
        "note": "r_star_future is computed from the MEAN loss increase over ALL future "
                "positions t>i. Because a single position i is usually attended to "
                "strongly by only a few future tokens (not all of them), averaging over "
                "the whole future window dilutes the signal and can make r_star_future "
                "look artificially small/uniform. r_star_future_maxagg uses instead the "
                "MAX loss increase over future positions (i.e. the single worst-affected "
                "future token), which is a more sensitive, less diluted alternative "
                "definition of the same quantity. Both are reported; see notes/ for "
                "discussion of which one is more meaningful.",
        "r_star_future_maxagg_mean": float(r_star_arr_maxagg.mean()),
        "r_star_future_maxagg_std": float(r_star_arr_maxagg.std()),
        "r_star_future_maxagg_min": int(r_star_arr_maxagg.min()),
        "r_star_future_maxagg_max": int(r_star_arr_maxagg.max()),
        "r_star_future_maxagg_histogram": {int(r): int((r_star_arr_maxagg == r).sum()) for r in RANK_GRID},
        "by_token_type_mean_rstar_future_maxagg": by_token_type_mean_rstar_maxagg,
    },
    "self_effect_diagnostics": {
        "note": "self_delta = loss increase at position i itself when only c_kv[i] is "
                "truncated. This is expected to be small (query at i is unaffected by "
                "rank_mask) but NOT exactly zero, because the causal mask allows i to "
                "attend to its own (truncated) key/value -- see module docstring.",
        "mean_abs_self_delta_at_rank16": float(np.mean(np.abs(self_delta_at_full_trunc))),
        "mean_abs_self_delta_fraction_of_mean_abs_future_delta_at_rank16": float(
            np.mean(np.abs(self_delta_at_full_trunc))
            / max(1e-9, np.mean(np.abs([rec["future_delta_by_rank"][RANK_GRID[0]] for rec in records])))
        ),
    },
    "sanity_check_rank_mask_applies_to_all_layers_max_abs_logit_diff": max_abs_diff,
}

with open(os.path.join(OUT_DIR, "exp0_v2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# also dump the raw per-position records (without the huge per-rank dicts, to keep
# the file small) for downstream analysis / plotting.
records_light = [
    {
        "seq": rec["seq"], "pos": rec["pos"], "target_token_id": rec["target_token_id"],
        "r_star_future": rec["r_star_future"],
        "r_star_future_maxagg": rec["r_star_future_maxagg"],
        "self_delta_at_r_star": rec["self_delta_at_r_star"],
        "future_delta_at_r_star": rec["future_delta_at_r_star"],
        "future_delta_max_at_r_star_maxagg": rec["future_delta_max_at_r_star_maxagg"],
    }
    for rec in records
]
with open(os.path.join(OUT_DIR, "exp0_v2_records.json"), "w") as f:
    json.dump(records_light, f, indent=2, ensure_ascii=False)

np.save(os.path.join(OUT_DIR, "exp0_v2_channel_importance_saliency.npy"), importance_order_saliency)
np.save(os.path.join(OUT_DIR, "exp0_v2_channel_importance_variance.npy"), importance_order_variance)

print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
print("DONE", flush=True)
