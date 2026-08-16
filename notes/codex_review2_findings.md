# Codex Review 2 — Exp0 v2 methodology + Kaggle training (2026-08-16)

Reviewer: standalone `codex review` via codex-lb, model `gpt-5.6-sol`.

## Verdict
- Exp0 v2 as paper evidence: **No — re-experiment required**.
- Continue Kaggle Exp1 training: **Conditional** — training math is correct, but runtime/resume risks remain.

## P1 — invalidates/requires re-experiment
1. **Layer misalignment**: v2 computes channel saliency/order from the final layer, then applies that same channel-index order to every layer. MLA latent channel indices are not aligned across independently trained layers. Compute separate saliency/order per layer, or mask only the explicitly defined probe layer.
2. **Variable future-window confound**: future mean/max aggregation uses all `t > pos`; earlier probe positions have longer horizons. Mean is more diluted for early positions; max has more chances for extremes. Use a fixed horizon or discounted/position-controlled aggregation.

## P2
- For causal intervention, manipulated KV belongs to `x[:, pos]`; token-type attribution should use x, not y.
- r* must account for non-monotonic loss vs rank; use suffix-all-satisfy or a monotone envelope, not the first isolated threshold crossing.
- Current code does not implement real compressed-cache autoregressive decoding. Claims must be limited to full-attention training/truncation simulation until cache-aware decode exists.
- Kaggle checkpoints need optimizer/scaler/RNG state and resume support.
- Package installs must fail fast (`check=True`).
- Inline Kaggle model copy still contains the previously fixed rank-mask/latent-return API bug and must be synced.
- Statistical reporting needs sequence-cluster bootstrap and calibration-seed repeats.

## Confirmed correct by review
- c_kv gradient retention and gradient×activation saliency mechanics.
- Batched `(K,T,D_C)` position isolation.
- Calibration/evaluation split.
- Gradient accumulation scaling by 1/6 and fp16 GradScaler flow.
