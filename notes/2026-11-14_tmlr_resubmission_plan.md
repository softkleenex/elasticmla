# 2026-11-14 - TMLR resubmission plan and pre-registered robustness analyses

## Decision

Venue debate (three subagents: ICML advocate, journal advocate, skeptical reviewer) on 2026-11-14.
Chosen route: TMLR resubmission in the first week of January 2027 (the author's 2026 TMLR quota,
N1 = 2 solo papers, is used by #13801 and the 2026-11-11 OOD submission), plus a non-archival
ICLR 2027 workshop in February. ICML 2027 rejected as the main target (estimated 8-15% after fixes;
same January window, no dual submission). Rolling journals rejected for now (slow, block other venues).

Main reviewer objection to answer first: the mean-vs-worst gap could be mechanical, since the top-k
mean of any signed 32-vector falls with k, and per-token cross-entropy swings by more than 0.1 nat
under any small perturbation.

## Schedule

| Phase | Dates | Owner | Content |
|---|---|---|---|
| 1 | 11/15-11/28 | Claude | analyses R1 and R2 below, reported as they come out |
| 2 | 11/29-12/05 | Claude | restructure: remove status block, router to a short case study, state that fp16/int8 quantization dominates on bytes, latency to appendix, fold in R1/R2 |
| 3 | 12/06-12/27 | Claude + author | clarity pass on the prose (author reads and owns every sentence) |
| 4 | 12/28-01/03 | both | TMLR anonymized build (drop repo link), supplement, resubmission form with link to #13801 and change list; check quota after 1/1; author submits |
| 5 | Feb 2027 | both | ICLR 2027 workshop version |

## Pre-registered analyses (committed before running)

Nothing below may be retuned after its result is seen. Results are reported whatever their sign.

### R1. Budget sensitivity (offline, existing raw records)

- Inputs:
  - 250M per-offset deltas `kaggle_output/review_reruns/spectrum250/spectrum250_v5/risk_capacity_v5_records.json` (768 probes, H = 32, the Table 2 run).
  - DeepSeek-V2-Lite coordinate-basis deltas `kaggle_output/review_reruns/publicmla_{a,b}/*.raw.npz`.
  - All inputs are SHA-256 hashed into the output.
- Grid: epsilon in {0.05, 0.10, 0.20, 0.50} nat; tail counts k in {32, 16, 8, 4, 2, 1}; H = 32. The suffix-safe r* is the same definition as in Section 3.3.
- Sanity check: at epsilon = 0.10, the recomputed per-probe r* must equal the stored `r_star_by_tail_count` (250M) and reproduce the published DeepSeek means. Otherwise the analysis stops.
- Pre-registered reading: the "large gap" claim stands at a given epsilon if the mean normalized r* at k = 1 is at least 3x that at k = H. We report the epsilon values where it fails.

### R2. Equal-norm random-perturbation null (new forward passes)

- Question: is the worst-offset requirement specific to removing the low-saliency latent channels, or does any perturbation of the same size give the same spectrum?
- Construction: for each probe token p, layer and rank r, truncation removes a residual e_r (the zeroed channels).
  - The null instead adds a random vector u_r with ||u_r|| = ||e_r|| to the full latent, in all layers at once, as in the truncation probe.
  - u_r is drawn as an isotropic Gaussian direction from seed 60521 + probe index. One draw per probe.
  - The null "rate" axis is r: same norm schedule as truncation.
  - The spectrum, the TCP and the cancellation statistic are then computed exactly as for truncation.
- Models: 30M first (local MPS), then 250M (local or Kaggle T4) on the same 768 probes and the same windows as Table 2. DeepSeek-V2-Lite only if compute allows; if not, we say so.
- Pre-registered reading, at k = 1 and at k = H separately:
  - If the null's mean normalized r* is within 0.10 (absolute) of truncation's, the paper must state that the spectrum reflects loss sensitivity to perturbations of that norm, not the information in the dropped channels.
  - If the null requires more rate (larger r*), truncation is gentler than random noise of equal norm. That supports the saliency order carrying information.
  - If the null requires less, report it as is.
