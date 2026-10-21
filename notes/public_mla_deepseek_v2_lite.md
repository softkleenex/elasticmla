# Risk-capacity spectrum and basis ablation on DeepSeek-V2-Lite (2026-10-21)

## 2026-10-21 - Public MLA replication
### Summary: the Section 5.1 spectrum and cancellation signature replicate on a production MLA model; the ~50 pp basis effect shrinks to ~11 pp.

### What and why

Every Section 5.1 number and the basis ablation (prediction 7) came from three TinyStories models
trained here. The GPT-2 control (`low_rank_does_not_mean_compressible.md`) moved to public
checkpoints but measured per-head KV, not an MLA latent. DeepSeek-V2-Lite caches exactly our object:
a 512-wide latent (= our 250M `d_c`) plus a 64-wide RoPE key, 27 layers, 5.7T training tokens.

`experiments/analyze_public_mla_spectrum.py` hooks `kv_a_layernorm` (whose output is the cached
latent) at every layer and runs the Section 5.1 probe unchanged: suffix-safe r*, H=32, eps=0.10,
`default_rank_grid(512)`, tail counts {32,16,8,4,2,1}, the four bases of `analyze_basis_dependence`.
WikiText-103-raw-v1: calibration 32 train sequences, evaluation 24 test sequences x 16 positions
(384 probes) of BOS + 512 tokens; BOS is never probed. float16 on Kaggle T4 x2 with accelerate CPU
offload; HF transformers 5.17 native DeepseekV2, model commit `604d566`.

No router, no retraining, nothing tuned after seeing results (grid, eps, H, bases all fixed from
the in-project runs before launch).

### Runs

| Kaggle kernel | bases | wall time | full-rank delta | identical-row spread |
|---|---|---:|---:|---:|
| `softkleenex/elasticmla-review-publicmla-a` | coord, pca | ~2.3 h | 0.0 | 0.0 |
| `softkleenex/elasticmla-review-publicmla-b` | grad, model_metric | ~2.3 h | 0.0 | 0.0 |

Both ran the scripts of commit `03ea9b5` (analysis source sha256 `12580424...` matches). Baseline mean
token loss 2.1383 in both. Evaluation tokens, probe positions, rank grid and baseline losses are
bit-identical across runs; the saliency hash differs (fp16 GPU backward is not deterministic), so
each basis uses its own run's calibration. `experiments/combine_public_mla_runs.py` checks all of
this (and each `.raw.npz` against its JSON hash) before computing the cross-run paired comparisons.

Artifacts: `experiments/public_mla/deepseek_v2_lite_{coord_pca,grad_model_metric,combined}.json`;
raw `.npz` (1.5 MB each) under `kaggle_output/review_reruns/publicmla_{a,b}/` (gitignored,
authenticated by `raw_npz_sha256`).

### Results

Normalized r* (/512):

| k | coord | pca | grad | model_metric | (our 250M coord) |
|---:|---:|---:|---:|---:|---:|
| 32 (mean) | 8.22% | 5.06% | 5.94% | 5.19% | 7.15% |
| 16 | 22.92% | 12.72% | 16.58% | 13.79% | 22.12% |
| 8 | 40.14% | 25.68% | 31.18% | 26.68% | 38.60% |
| 4 | 56.86% | 42.36% | 49.51% | 43.47% | 54.16% |
| 2 | 70.69% | 56.90% | 64.97% | 57.44% | 64.95% |
| 1 (max) | 78.65% | 67.34% | 73.47% | 67.58% | 71.78% |
| TCP | 0.704 [0.676, 0.731] | 0.623 | 0.675 | 0.624 | 0.646 |

Cancellation at the mean-safe rank (coord): positive-part / signed mean 2.10x; 96.6% of probes have
an offset above eps (ours: 2.05-2.31x, 93.1-96.1%).

Paired max-criterion differences (pp of d_c): pca-coord -11.31 [-13.54, -9.12]; model_metric-coord
-11.08; grad-coord -5.18; model_metric-pca +0.24 [-1.29, +1.90]; pca-grad -6.13. Ours: about -50 for
pca-coord at every scale. Probes needing all 512 channels: coord 106/384, pca 59/384 (ours 250M: 77/768
vs 0/768).

### Reading

- The spectrum's shape and magnitude replicate; the value does not exactly (TCP CI above ours).
- The basis effect is small on a production model, as the GPT-2 KV control predicted -- now shown on
  the actual MLA latent. The TinyStories models are unrepresentatively compressible in a rotated basis.
- Basis ranking (pca ~ model_metric < grad < coord) is the same as in our models.

### Issues

- 275/384 coord probes are floor-censored at rank 16 under the mean criterion: 8.22% is an upper bound,
  TCP a lower bound.
- Kaggle setup problems (old dataset mounted, GPU0 OOM from onloaded offloaded experts under plain
  autograd -> layer gradient checkpointing, grouped_mm lacking fp64 -> eager experts) are in the
  commit message of `03ea9b5` and the smoke runs; none affected the full runs.
