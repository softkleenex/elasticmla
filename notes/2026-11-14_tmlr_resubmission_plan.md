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

## R1 result (2026-11-14, `experiments/budget_sensitivity.json`)

Sanity checks passed: all 250M stored r* values were reproduced at epsilon = 0.10 (0 mismatches), and the
published DeepSeek coordinate means were reproduced to 1e-9. Mean normalized r*, H = 32:

| epsilon | 250M mean (k=32) | 250M max (k=1) | ratio | DSV2 mean | DSV2 max | ratio |
|---|---|---|---|---|---|---|
| 0.05 | 0.133 | 0.866 | 6.5 | 0.139 | 0.935 | 6.8 |
| 0.10 | 0.071 | 0.718 | 10.1 | 0.082 | 0.787 | 9.6 |
| 0.20 | 0.040 | 0.514 | 12.8 | 0.048 | 0.549 | 11.5 |
| 0.50 | 0.031 | 0.263 | 8.4 | 0.032 | 0.276 | 8.7 |

The pre-registered large-gap criterion (ratio >= 3) holds at every epsilon on both models. Caveat: at
epsilon = 0.50 the mean criterion sits at the grid floor (16/512 = 0.031), so the ratio there is a lower
bound on the mean side, not a measured value. The absolute TCP shrinks as the budget loosens (0.73 -> 0.23
at 250M), so "the premium is ~0.65" is specific to epsilon = 0.10 and must be stated that way.
DeepSeek uses shard a only, the only shard with the coordinate basis.

## R2 implementation details (2026-11-15, fixed before the full runs)

Script `experiments/analyze_perturbation_null.py`, tests `tests/test_perturbation_null.py`.

- The null direction u is drawn once per (probe, layer) and kept fixed across ranks, so only its norm
  follows the truncation schedule ||c * (1 - m_r)||. Probe index = position in the published record order.
- Channel orders are read from the published summary (`layer_channel_orders`), not recalibrated, so
  truncation and null share exactly the Table 2 orders. Windows and positions are regenerated from the
  evaluation seed and, at 250M, checked against the published records (the script stops on a mismatch).
- epsilon = 0.10 and H = 32, as in Table 2. Truncation is recomputed in the same run, and the
  comparison is paired per probe (sequence-cluster bootstrap of null - truncation).
- Disclosure: a 2-sequence smoke run of the 30M model was run once on MPS to debug the script and its
  output was seen. Nothing above was changed after it.

### Corrections after an adversarial code review (2026-11-15, before any full-run result was seen)

A reviewer subagent found that the first version matched norms on the null's own (already perturbed)
stream, which equals the pre-registered ||e_r|| only at layer 0. Fixed: the truncation forward records
||e_r|| per layer with a read-only hook, and the null adds exactly those norms. Also fixed:
- Window check is mandatory: exact match to the published records (250M); for 30M, whose summary
  predates stored records, per-probe agreement of the mean-safe rank with `mean_safe_rate_diagnostics_by_record`.
- Stopping rule: the run fails if recomputed truncation r* agrees with the published per-probe r* on
  fewer than 99% of probes. Baselines are batched exactly as in the reference script.
- Integer r* tables are kept, not reconstructed from floats.
- Added, explicitly **exploratory** (not pre-registered): a covariance-shaped null, u ~ N(0, Sigma_l) with
  Sigma_l the layer latent covariance on the published calibration windows, seed 62521 + probe index.
  It addresses the objection that isotropic noise falls in directions the up-projections ignore.
The partial 30M run started with the first version was stopped before finishing; its output was never read.
