# Pair-interaction residuals: theory prediction 5 (2026-09-16)

## What prediction 5 says

From `notes/theory_contextual_tail_rate.md`, Section 7:

> 5. Pairwise intervention residuals should be nonzero if isolated-label supervision is
>    structurally misaligned with joint routing.

Section 4 of the same memo introduces the additive special case `D(r, Z) = sum_t d_t(r_t, Z)`,
under which the Lagrangian reduces to a per-token marginal-value rule, and says explicitly that the
method does not assume it. The oracle labels, however, do lean on it: capacity is measured by
truncating **one position at a time**, while the deployed router truncates **every position at
once**.

## The test

The second-order mixed difference. Writing `L_S` for the per-token loss vector when the positions
in `S` are truncated,

    residual = L_{t1,t2} - L_{t1} - L_{t2} + L_{}

is identically zero for any additive distortion, whatever the individual effects are. Three
choices make this a measurement rather than a formality:

- It is summed over the **whole sequence**, not a horizon window, so the two positions' reuse
  windows need not be aligned.
- It is reported **relative to the size of the individual effects** (`|d1| + |d2|`), so "nonzero"
  means "not negligible" rather than "not exactly zero in floating point".
- **Separation is swept.** Additivity should hold trivially once the positions are far enough
  apart; the interesting regime is nearby positions, and a residual that does not decay with
  distance would indicate a measurement artifact rather than an interaction.

`experiments/analyze_pair_interaction.py`, 1008 pairs per scale (8 evaluation sequences x 7
separations x 6 pairs x 3 truncation ranks), using the published sampling protocol so the channel
order and evaluation sequences are the ones every other result was measured on.

`forward_with_layer_masks` was generalized from `(batch,)` to `(batch, n_positions)` probe
positions rather than duplicating the model forward. The single-position path is bit-identical
afterwards (verified: the 1-D and `(B,1)` forms agree exactly, a duplicated position collapses to
the single-position result, out-of-range positions are rejected), and
`tests/test_risk_capacity_spectrum.py` pins that.

## Result: confirmed at all three scales, and the violation grows with scale

Mean residual as a fraction of the individual effects (p90 in parentheses):

| separation | 30M | 122M | 250M |
|---:|---:|---:|---:|
| 1 | **9.09%** (20.52%) | **12.17%** (28.71%) | **13.17%** (26.70%) |
| 2 | 5.63% (11.61%) | 8.05% (17.53%) | 8.61% (17.62%) |
| 4 | 3.77% (7.65%) | 4.68% (9.67%) | 4.95% (10.14%) |
| 8 | 1.88% (3.46%) | 3.09% (7.37%) | 3.65% (7.33%) |
| 16 | 1.32% (2.56%) | 1.81% (3.72%) | 2.08% (3.83%) |
| 32 | 0.67% (1.46%) | 1.13% (2.27%) | 1.25% (2.05%) |
| 64 | 0.44% (0.73%) | 0.56% (1.21%) | 0.67% (1.23%) |

By truncation severity (mean over all separations):

| scale | heaviest (r=16) | middle | lightest |
|---|---:|---:|---:|
| 30M (d_c=256) | 5.69% | 3.01% (r=128) | 1.07% (r=224) |
| 122M (d_c=384) | 8.41% | 4.06% (r=192) | 1.03% (r=352) |
| 250M (d_c=512) | 7.98% | 4.63% (r=192) | 2.13% (r=384) |

Three consistent patterns:

1. **Residuals are nonzero**, so prediction 5 holds and the additive special case is falsified.
2. **They decay monotonically with separation** at every scale, falling below ~1% by separation 32.
   The interaction is local, which is what a reuse-horizon mechanism predicts.
3. **They grow with scale and with truncation severity.** Adjacent-position residuals rise from
   9.09% at 30M to 13.17% at 250M, and the overall mean from 3.26% to 4.91%.

## What this means for the method

Read plainly, this both confirms the prediction and partially defends the current oracle: the
isolated-position labels are structurally misaligned with joint routing, but the misalignment is
*local and modest* -- roughly 9-13% of the individual effects for adjacent positions under the
heaviest truncation, under 1% beyond separation 32. It converts Section 6's "Oracle scope"
limitation from a qualitative worry into a measured quantity.

The scale trend is the part that matters for future work: the misalignment gets **worse**, not
better, as the model grows. Joint-rollout training (already used for the deployed policy) is
therefore more, not less, necessary at larger scale, and an oracle built from isolated labels
should be expected to degrade with scale.

For calibration against the other mechanism result in this repository: basis choice moves the
required rate by ~50 percentage points (`notes/basis_dependence_prediction7.md`), while
isolated-versus-joint labelling costs single-digit percentages of the effect size. Basis design is
by far the larger of the two problems.

## Device cross-check

The 30M analysis was run both locally on Apple MPS and on a Kaggle P100 (CUDA). The two agree to
within 0.001 percentage points at every separation, on identical checkpoint hashes, confirming the
same device independence established for the basis ablation.

## Artifacts

- `experiments/analyze_pair_interaction.py`, `tests/test_pair_interaction.py`
- `experiments/contextual_router_{30m,122m,250m}/pair_interaction.json` (records included)
- `experiments/pair_interaction_kernel.log`; kernel `softkleenex/elasticmla-pair-interaction` v1
