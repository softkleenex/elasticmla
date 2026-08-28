# Validated risk-capacity spectrum and cancellation mechanism (30M/122M/250M)

## Status

All three scales completed on Lightning T4 GPUs (`elasticmla-risk-v5-30m-0819-v2`,
`elasticmla-risk-v5-122m-0819-v2`, `elasticmla-risk-v5-250m-0826`). Checkpoint/data SHA-256 all
verified against locally recomputed hashes. Records (per-position raw deltas) were written to
job-local Studio storage that is not retrievable after job completion; only the aggregate summary
JSON (printed to stdout and independently hash-checked against checkpoint/data) was recovered.
`experiments/contextual_router_{30m,122m,250m}/risk_capacity_v5_summary.json` are the artifacts.

The 250M model (249.27M unique parameters, d_model=1024, 16 layers, d_c=512, trained 3,000 steps
on Kaggle P100, `softkleenex/elasticmla-exp2-scale-up-training-250m`, final val loss 1.6588) used
the same TinyStories corpus construction as the 122M model, producing a byte-identical held-out
validation stream (verified: both have SHA-256 `03246b57...073ce`). It has not been used for
router training, oracle generation, or fresh-window confirmation; it contributes only to this
risk-capacity spectrum result.

## Full monotone risk-capacity spectrum across an 8x parameter range

| Tail count k (of 32) | alpha=1-k/H | 30M norm. | 122M norm. | 250M norm. |
|---:|---:|---:|---:|---:|
| 32 (mean) | 0.000 | 9.00% | 8.22% | 7.15% |
| 16 | 0.500 | 21.47% | 23.00% | 22.53% |
| 8 | 0.750 | 39.65% | 39.52% | 40.14% |
| 4 | 0.875 | 55.75% | 54.22% | 57.23% |
| 2 | 0.938 | 66.94% | 65.56% | 69.18% |
| 1 (max) | 0.969 | 74.07% | 72.92% | 76.03% |

Every step is monotone nondecreasing at all three scales (guaranteed by Proposition 2). The
spectra visually nearly overlap across an 8x parameter range (Figure 2a).

## Tail-capacity premium is nearly scale-invariant across three scales

Normalized premium `E[r*_max - r*_mean]/d_c`:

- 30M: 0.6507 (95% CI [0.6335, 0.6673])
- 122M: 0.6470 (95% CI [0.6283, 0.6641])
- 250M: 0.6888 (95% CI [0.6647, 0.7129])

The 30M/122M intervals overlap almost completely; the 250M interval is adjacent/slightly higher
but still within ~0.02-0.04 of the other two, and all three are far tighter together than the
spread of absolute required ranks (36.6 to 389.3 across scales). This is the strongest
scale-consistency result in the paper, now confirmed at a third, substantially larger scale.

## The mean/max separation is NOT explained by rare catastrophic spikes, at any scale

At the rank that is *safe on signed mean*:

| Scale | signed mean | positive-part mean | max | fraction of records with >=1 offset above epsilon |
|---|---:|---:|---:|---:|
| 30M | 0.0269 | 0.0550 (2.05x) | 0.836 | 93.1% |
| 122M | 0.0279 | 0.0645 (2.31x) | 0.911 | 94.1% |
| 250M | 0.0307 | 0.0659 (2.15x) | 0.948 | 96.1% |

The cancellation mechanism (mean-safe rank still exceeds epsilon at >=1 offset for the large
majority of records, and the positive-part mean is roughly double the signed mean) replicates at
250M, if anything slightly more strongly (96.1% vs 93.1%/94.1%).

## Implication for the paper

- The tail-capacity premium and cancellation mechanism are now three-scale, not two-scale,
  findings, substantially strengthening the scale-consistency claim.
- The 250M point does not participate in router training/confirmation/causal-heuristic results in
  this version; extending those experiments to 250M is future work (requires generating a
  contextual oracle and repeating the full router/heuristic pipeline at this scale).
