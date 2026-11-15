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

## Second debate (2026-11-15): restructure and the remaining reviewer objections

Three subagents (TMLR action editor, hostile reviewer, author advocate) read the draft. Agreed:
Section 5 is reorganized around the objection "is the gap mechanical, or just perturbation size?"
(Table 2 -> R1 -> R2 -> cancellation -> basis); the router becomes a short case study (Tables 4/5 kept);
latency moves to the appendix but the abstract keeps "no peak-memory or throughput gain, 2.4x slower
decode"; the status block goes to the resubmission form. Disagreement resolved: we do **not** write that
quantization "dominates" without measuring it; R4 below measures it. "Scale-invariant" becomes "similar"
(DeepSeek's CI does not overlap). Already answered from existing data: the positive-part-mean criterion
(no cancellation of harm by gain) needs 0.103 at 250M vs 0.071 (mean) and 0.718 (max), so the gap is
not an artifact of the signed mean (`horizon_capacity_spectrum.positive_mean`, Table 2 run).

### R3. Order-statistic null (offline, 250M Table 2 records)

- Question: is the gap just "max of 32 noisy values vs their mean"?
- Construction: for each probe and rank, replace the 32 offset deltas by 32 i.i.d. draws from N(mu, sigma^2)
  with that row's own mean and standard deviation (seed 63521 + probe index), and separately flip each
  offset's sign of (delta - mu) at random (sign-randomized, keeps |deviations|). Recompute r* at k = H, 1.
- Reading: if the i.i.d. Gaussian null reproduces the k = 1 mean normalized r* within 0.10, the paper says
  the worst-offset requirement is what a dispersed, near-zero-mean loss change implies by order statistics,
  and drops any claim that offsets are structured. If the observed k = 1 rate exceeds the null by > 0.10,
  the worst offsets are heavier-tailed than Gaussian dispersion. Either way it is reported.

### R4. Latent quantization baseline (30M local, 250M Kaggle)

- Per-token symmetric absmax int8 and int4 quantization of the MLA latent at every position and every
  layer (decoupled RoPE key left in full precision, as in the packed cache), on the 24 Table 2 evaluation
  windows. Bytes relative to an fp16 latent: int8 = 0.5 + 2/d_c (one fp16 scale per token-layer), int4 = 0.25 + 2/d_c.
- Report: mean per-token loss change over all positions, and the bytes. Compared, without new tuning, to the
  router's published (bytes, loss) points. No threshold: the numbers go in Section 6 as measured.

### R5. Joint application of per-token safe rates (250M Kaggle)

- Question: does per-token mean safety compose when many tokens are truncated together, and what does
  the worst-offset rule buy at sequence level?
- For each of the 24 sequences, truncate all 32 probe positions jointly (all layers), each at its published
  r* for k = H (mean-safe), then for k = 1 (max-safe). Measure the loss change at every position after the
  first probe: sequence mean, fraction of positions above epsilon = 0.10, and the maximum; plus the rank sum.
- Reading: if the joint mean-safe sequence-mean change exceeds 2 epsilon, per-token mean safety does not
  compose and the paper says so. The comparison of the fraction above epsilon between the two allocations,
  at their rank sums, is reported as measured.
- Also checked (unit test, no reading needed): truncating position p in the full-attention probe equals
  masking p's latent in incremental cached decoding.

## Results so far (2026-11-15)

### R2 at 30M (`experiments/perturbation_null/null_30m.json`, MPS, windows verified 768/768)

| k | truncation | isotropic null (pre-reg.) | null - trunc [95% CI] | covariance null (exploratory) | cov - trunc [95% CI] |
|---|---|---|---|---|---|
| 32 (mean) | 0.090 | 0.062 | -0.028 [-0.032, -0.023] | 0.067 | -0.023 [-0.028, -0.018] |
| 1 (max) | 0.741 | 0.566 | -0.175 [-0.202, -0.148] | 0.737 | -0.003 [-0.027, 0.019] |

Pre-registered reading: at k = H the null is within 0.10, so the paper must say the mean-criterion rate
reflects loss sensitivity to perturbations of that norm. At k = 1 isotropic noise of equal norm needs 0.175
less rate than truncation. The exploratory covariance-shaped null reproduces truncation at every k (TCP 0.671
vs 0.651): realistic perturbations of the same norm are as harmful as removing the low-saliency channels.
So the spectrum is a property of how the loss responds to latent perturbations of a given size, not
evidence that the dropped channels carry special information. 250M pending (Kaggle).

### R3 (`experiments/order_statistic_null.json`, 250M, offline)

Observed / Gaussian / sign-randomized mean normalized r*: k = 32: 0.071 / 0.108 / 0.108; k = 1: 0.718 / 0.663 /
0.758. Pre-registered reading: the Gaussian null is within 0.10 at k = 1 (0.055), so the worst-offset
requirement is what dispersed loss changes imply by order statistics; we drop any claim that the worst
offsets are structured. (The mean-side null is higher than observed because the observed rows are not
Gaussian around their mean; reported as is.)

### R4 at 30M (`experiments/latent_quantization/quant_30m.json`)

Per-token int8 latent: 0.508 of fp16 bytes, mean loss change -0.00002 nat [-0.0001, 0.0001], no position
above epsilon. int4: 0.258 of fp16 bytes, +0.0007 nat [-0.0010, 0.0022], 5.6% of positions above epsilon.
The router's published points (0.60-0.69 of bytes, +0.10-0.18 nat) are dominated on bytes by int8, now as
a measurement. 250M (and R5) pending.

### Consequence for the framing (to apply in Phase 2)

The paper becomes a measurement paper: per-token latent truncation produces dispersed, near-zero-mean
future-loss changes, so worst-offset safety needs ~10x the mean-safe rate at every budget (R1); this cost is
shared by realistic equal-norm perturbations (R2) and predicted by the order statistics of the dispersion
(R3). The lesson is about evaluation: mean-loss evaluations of token-wise compression understate the
worst-offset cost by an order of magnitude. The router and the packed cache are a case study, dominated on
bytes by int8 quantization (R4).

### R4 and R5 at 250M (`experiments/latent_quantization/quant_250m.json`, Kaggle T4, windows identical to records)

R4: int8 latent 0.504 of fp16 bytes, +0.00005 nat [-0.00001, 0.00011], no position above epsilon; int4 0.254,
+0.0018 nat [0.0001, 0.0035], 5.5% of positions above epsilon.

R5 (all 32 probes per sequence truncated jointly, 24 sequences):

| allocation | mean probe rate | sequence-mean loss change | positions > epsilon | max position change |
|---|---|---|---|---|
| mean-safe (k = 32) | 0.071 d_c | +0.102 [0.093, 0.111] | 28.6% [27.6, 29.7] | +4.64 [3.90, 5.62] |
| max-safe (k = 1) | 0.718 d_c | -0.005 [-0.007, -0.004] | 4.4% [3.8, 5.0] | +0.41 [0.32, 0.52] |

Pre-registered reading: mean-safe rates compose at the sequence-mean level (0.102 <= 2 epsilon). But a
mean-loss evaluation of that allocation would report about epsilon while 29% of positions exceed the budget
and the worst position loses 4.6 nat; the max-safe allocation cuts that to 4.4% and 0.41 nat at 10x the rate.
This is the paper's answer to "why does mean-safe vs tail-safe matter".

### R2 at 250M (`experiments/perturbation_null/null_250m.json`, Kaggle T4)

Windows identical to the Table 2 records; recomputed truncation r* agrees with the published values on
100% of probes at every k. Isotropic null (pre-registered): k = 32 0.031 (grid floor) vs 0.071, -0.040
[-0.048, -0.033], within 0.10; k = 1 0.434 vs 0.718, -0.284 [-0.304, -0.264], null needs less. Covariance
null (exploratory): k = 32 0.033; k = 1 0.703, -0.015 [-0.034, 0.005]; TCP 0.670 vs 0.646. Same reading as
30M, now at both scales. Folded into Section 5.2 / Table 4 of the draft.

## Phase 4 preparation (2026-11-15)

- Anonymized build: `uv run python manuscript/convert_to_latex.py --style tmlr` now replaces the repository
  sentence with a pointer to the supplement and stops if the author's handle survives. Built 16 pages; no
  author name, handle, affiliation or e-mail in main.tex or the PDF text. Supplement
  (`build_supplement.py`): 292 files, no identity strings found by grep.
- Still for the author before submitting (early January): reread every sentence (TMLR: authors own the
  text), check the 2027 quota on the TMLR checker after 1/1, fill the form, submit.

### Draft change list for the resubmission form (link to #13801)

This is a resubmission of #13801 (desk-rejected 2026-11-08 for clarity of writing). Changes:
1. Reframed as a measurement paper. The main claim is the gap between mean-safe and worst-offset-safe
   latent rates; the learned router is now a short case study.
2. New pre-registered controls for the gap (plans committed before running, reported whatever the sign):
   budget sensitivity (ratio 6.5-12.8 for epsilon 0.05-0.5), an equal-norm random-perturbation null at
   30M and 250M, an order-statistic null, and a joint application of per-token safe rates (mean-safe
   allocation keeps mean loss near the budget while 28.6% of positions exceed it).
3. Measured latent int8/int4 quantization baseline: int8 dominates the packed cache on bytes; stated plainly.
4. Unit test showing the full-attention truncation probe equals incremental cached decoding.
5. Latency and router training details moved to the appendix; status/process history removed; prose
   rewritten for clarity, with terms defined before use.

## Third round (2026-11-16): pre-registered analyses R6-R9 (committed before running)

Proposed by a reviewer subagent after the author's full read of the restructured draft. Order: R6 and R7 at
30M locally, then 250M (Kaggle), then R8, then R9. Same rules: nothing retuned after its result is seen.

### R6. The spectrum under a nonnegative metric (KL and greedy flips)
- Question: does the ~10x gap survive a metric that cannot cancel? Signed target-token cross-entropy can be
  negative at some offsets; KL(full || truncated) of the next-token distribution cannot.
- Construction: truncation only, same probes, windows, channel orders and stopping rule as R2 (>= 99% of
  loss-based r* reproduced in the same pass). At each probe, rank and offset record the KL and whether the
  greedy (argmax) token differs from the full model's. r*_k(delta) uses the same top-k mean and suffix rule
  on the KL vectors. Full-rank KL must be < 1e-4 nat at every offset or the run stops.
- Reading: primary budget delta = 0.05 nat (also reported: 0.01, 0.10). The gap claim holds under KL if the
  mean normalized r* at k = 1 is at least 3x that at k = H; if not, the abstract must say the
  order-of-magnitude gap is a property of the signed target loss. Greedy flip rates at the loss-based
  mean-safe and worst-offset-safe rates are reported as measured.

### R7. Quantization bit-width as the rate axis
- Question: is the mean-vs-worst gap specific to truncation, or does it also hit per-token quantization?
- Construction: quantize only the probe token's latent, in all layers at once (R4's symmetric absmax
  quantizer, fp16 scale), at b in {1, 2, 3, 4, 5, 6, 8} bits (b = 1 means sign times scale); full precision
  is the top of the axis. b*_k is the smallest width that is safe under the same suffix rule (epsilon = 0.10,
  loss deltas, k = H and k = 1); a probe unsafe even at 8 bits gets b* = 16.
- Reading: gap present if mean b*(k = 1) - mean b*(k = H) >= 2 bits; absent if <= 1 bit; in between, reported
  as inconclusive.

### R8. R2 and R5 on DeepSeek-V2-Lite (Kaggle T4 x2, compute permitting)
- Coordinate basis, the 384 shard-a probes, seeds 60521 / 62521 + probe index; covariance from the script's
  calibration sequences; R5 from the shard-a r*. Sanity: truncation r* matches shard-a raw deltas on >= 99%
  of probes. R2 and R5 readings unchanged, except that k = H is reported but not read for R2 because 275 of
  384 probes sit at the grid floor there.

### R9. Per-token low-rank truncation of GPT-2's per-head KV cache
- Question: is the gap specific to MLA or to our training data?
- Construction: per-head principal directions as in `analyze_public_kv_basis.py`; project K and V at the probe
  position only, in all layers; 24 sequences x 32 probes, wikitext-2, block 512, H = 32, epsilon = 0.10.
- Reading: the R1 criterion (ratio >= 3) for gap present.

R7 implementation detail fixed before running: b = 1 is sign(x) * mean|x| per row (L1-optimal binary scale,
fp16); b >= 2 use R4's absmax quantizer with qmax = 2^(b-1) - 1. Script `experiments/analyze_metric_and_bitwidth.py`.

### R6 and R7 at 30M (`experiments/metric_bitwidth/metric_bitwidth_30m.json`, MPS)

Sanity: mean-safe r* reproduced on 768/768 probes; max full-rank KL 4.7e-7.
R6, mean normalized r* under KL(full || truncated), k = 32 / k = 1 / ratio: delta 0.01: 0.296 / 0.763 / 2.58;
**delta 0.05 (primary): 0.109 [0.103, 0.115] / 0.490 [0.469, 0.511] / 4.50**; delta 0.10: 0.079 / 0.360 / 4.53.
Reading: the gap holds under a metric that cannot cancel, but it is about half as large as under the signed
target loss (ratio ~4.5 vs ~8). Greedy flips: at the loss-based mean-safe rate, 4.4% of offsets
[4.0, 4.8] flip and 74.6% of probes flip at least once within 32 offsets; at the worst-offset-safe rate,
0.72% and 19.1%.
R7 bit-width: mean b* k = 32: 1.01, k = 16: 1.18, k = 8: 1.61, k = 4: 2.12, k = 2: 2.54, k = 1: 2.85.
Gap 1.84 bits: pre-registered reading "inconclusive". The k = 32 value sits at the axis floor (1 bit), so the
gap is a lower bound. Reported as is.

### R8 implementation details (fixed before running; `experiments/analyze_public_mla_null.py`)

- Channel order: shard a stored no order, so calibration is rerun with the published settings. The stop
  rule is r* agreement >= 99% at k = H and k = 1 against the shard-a raw deltas; window, position and grid
  must match exactly. The saliency hash match is recorded as a diagnostic.
- Covariance null uses the centered Sigma = E[cc^T] - mu mu^T (mu collected read-only during calibration).
- R5 baseline is an unmodified forward with the same batch shape; metrics run from the first probe on.
- R2 at k = H is reported, not read (275 of 384 probes at the grid floor).
