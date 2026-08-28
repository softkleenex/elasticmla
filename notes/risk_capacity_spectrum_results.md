# Validated risk-capacity spectrum and cancellation mechanism (30M/122M)

## Status

Both scales completed on Lightning T4 (`elasticmla-risk-v5-30m-0819-v2`, `elasticmla-risk-v5-122m-0819-v2`).
Checkpoint/data SHA-256 match the frozen v4 artifacts exactly. Records (per-position raw deltas) were
written to job-local Studio storage that is not retrievable after job completion; only the aggregate
summary JSON (printed to stdout and independently hash-checked against checkpoint/data) was recovered.
Therefore this note reports validated **aggregate** spectrum/diagnostics, not a re-auditable per-record
file. `experiments/contextual_router_{30m,122m}/risk_capacity_v5_summary.json` are the artifacts.

## Full monotone risk-capacity spectrum (not just mean/max)

| Tail count k (of 32) | alpha=1-k/H | 30M mean r* | 30M norm. | 122M mean r* | 122M norm. |
|---:|---:|---:|---:|---:|---:|
| 32 (mean) | 0.000 | 23.04 | 9.00% | 31.56 | 8.22% |
| 16 | 0.500 | 54.96 | 21.47% | 88.33 | 23.00% |
| 8 | 0.750 | 101.50 | 39.65% | 151.75 | 39.52% |
| 4 | 0.875 | 142.71 | 55.75% | 208.19 | 54.22% |
| 2 | 0.938 | 171.35 | 66.94% | 251.75 | 65.56% |
| 1 (max) | 0.969 | 189.62 | 74.07% | 280.00 | 72.92% |

Every step is monotone nondecreasing at both scales, exactly as guaranteed by Proposition 2
(risk-capacity ordering) in `notes/theory_contextual_tail_rate.md`. This is the first empirical
demonstration that the mean/max separation reported earlier is one slice of a smooth, monotone risk
spectrum rather than an artifact of comparing only two endpoints.

## Tail-capacity premium is nearly scale-invariant

Normalized premium `E[r*_max - r*_mean]/d_c`:

- 30M: 0.6507 (95% CI [0.6335, 0.6673])
- 122M: 0.6470 (95% CI [0.6283, 0.6641])

These two intervals overlap almost completely. This is the strongest scale-consistency result in
the paper: the *relative* extra rate required to protect the worst reuse offset, versus the average
offset, is essentially unchanged from 30M to 122M even though absolute required rank grows.

## The mean/max separation is NOT explained by rare catastrophic spikes

This refutes the intuitive "sparse harmful token" story. At the rank that is *safe on signed mean*:

| Scale | signed mean | positive-part mean | max | fraction of records with >=1 offset above epsilon |
|---|---:|---:|---:|---:|
| 30M | 0.0269 | 0.0550 (2.05x) | 0.836 | 93.1% |
| 122M | 0.0279 | 0.0645 (2.31x) | 0.911 | 94.1% |

If harm were concentrated in a few rare spikes, most mean-safe positions would have zero offsets
above epsilon. Instead, over 93% of records have **at least one** future offset whose loss increase
exceeds epsilon even when the signed mean is within budget, and the positive-part mean is roughly
double the signed mean. The mean criterion is therefore satisfied mostly by **cancellation between
many small positive and negative excursions across the 32-token horizon**, not by suppressing a few
outliers. This means a max/CVaR-style criterion is protecting against pervasive, not rare, risk.

## Implication for the paper

- Replace any "rare tail-risk token" narrative with "distributed reuse risk that a per-token mean
  criterion masks through cancellation."
- The scale-invariant tail-capacity premium (~0.65 at both scales) is a defensible, novel,
  quantitative claim suitable as a headline empirical result independent of the router work.
- This spectrum should replace the two-point mean/max table in the manuscript's Results section.
