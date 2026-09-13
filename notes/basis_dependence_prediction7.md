# Basis dependence of the nested latent code: theory prediction 7 (2026-09-13)

## What prediction 7 says

From `notes/theory_contextual_tail_rate.md`, Section 7:

> 7. A learned orthogonal nested basis should reduce safe rates or nonmonotonicity relative to a
>    raw coordinate permutation if basis design is a material bottleneck.

Every rank this project reports is measured in the model's own latent coordinates, ordered by a
gradient-times-activation saliency. That is a *permutation* -- the most restricted possible nested
basis. The question is whether the reported rates are an artifact of that restriction.

## Method: rotate the model, not the probe

`c = W_DKV x` and `k, v = W_UK c, W_UV c` are all bias-free, so substituting
`W_DKV <- U^T W_DKV` and `W_UK, W_UV <- W_UK U, W_UV U` leaves the full-rank function unchanged
while making "mask the first `k` coordinates" identical to "project onto `span(U[:, :k])`" in the
original latent space. This reuses the already-validated coordinate-masking probe verbatim instead
of writing a second forward pass, and it gives a free correctness check: at full rank each rotated
model must reproduce the original's logits (measured deviation ~2.4e-05, the accelerator's matmul
noise). `experiments/analyze_basis_dependence.py`; the rotation algebra is pinned by
`tests/test_basis_dependence.py`.

Three bases, all calibrated in one pass over the same tokens so the comparison cannot be confounded
by calibration data:

- `coord` -- the published saliency permutation (`U` is a permutation matrix)
- `pca` -- eigenvectors of the uncentred second moment `E[c c^T]`, by eigenvalue: reconstruction of
  the latent itself, ignoring the loss
- `grad` -- eigenvectors of `sym(E[g c^T])`, by |eigenvalue|. To first order, dropping direction `u`
  changes the loss by `-(g.u)(u.c)`, so `u^T sym(E[g c^T]) u` is exactly the mean first-order loss
  effect along `u`. Its diagonal in the coordinate basis is the signed version of the published
  saliency, which makes `coord` the axis-aligned special case of the same criterion.

## Validation, and the bug it caught

The `coord` basis runs the published procedure, so it must reproduce the published spectrum. The
first version **did not**: 8.89% / 73.16% against the published 9.00% / 74.07% at 30M.

The cause was mine, and it was silent -- no crash, no warning, plausible-looking numbers. The
published analysis draws sequence starts with a rejection sampler (`sample_starts`) that forces
nonoverlapping windows and excludes starts across calibration repeats; this script used a plain
`rng.integers`. It was therefore measuring a *different probe population*. Two checks separated the
possibilities: the published script rerun on this machine's MPS reproduced the published T4 numbers
to the digit (device effect exactly 0.00pp), and the pre-fix script rerun on a CUDA P100 reproduced
the *wrong* numbers exactly (8.894857% / 73.160807%, bit-identical to MPS). So the discrepancy was
purely the sampling protocol, and this analysis is device-independent.

After the fix, `coord` reproduces the published `mean_r_star` and `max` exactly (23.041666666666668
and 189.625 at 30M; likewise at 122M). `tests/test_basis_dependence.py` now pins the protocol,
including a test that asserts plain `rng.integers` does *not* reproduce the published starts.

## Result: confirmed, and it is a negative result for the reported rates

768 probes per scale, H=32, epsilon=0.1, normalized mean `r*`:

| scale | basis | mean r* | worst-offset r* | probes with a monotonicity violation |
|---|---|---:|---:|---:|
| 30M | coord | 9.00% [8.54, 9.52] | **74.07%** [72.10, 75.72] | 59.4% |
| 30M | pca | 6.49% [6.38, 6.62] | **23.56%** [22.76, 24.39] | 82.0% |
| 30M | grad | 6.54% [6.44, 6.67] | **34.29%** [32.90, 35.78] | 81.9% |
| 122M | coord | 8.22% [7.52, 8.90] | **72.92%** [70.96, 74.95] | 69.7% |
| 122M | pca | 4.63% [4.50, 4.76] | **23.25%** [21.55, 24.95] | 93.4% |
| 122M | grad | 4.75% [4.60, 4.93] | **35.75%** [33.88, 37.69] | 89.8% |

The worst-offset rate falls by ~50 pp at both scales (a 3.1x reduction), with bootstrap intervals
nowhere near overlapping. The `r*` histograms make the mechanism plain: under `coord` at 30M, 410 of
768 probes need at least 224 of 256 channels (207 at 224, 203 at 256); under `pca` only 3 probes
need 224 or more and none need the full width.

**So a large part of the paper's headline "high tail rate" is a cost of axis alignment, not an
intrinsic property of the latent code.** This is deployable rather than merely diagnostic: the
rotation is a fixed change of variables that can be folded permanently into
`W_DKV`/`W_UK`/`W_UV`, and the packed cache stores the same number of floats either way.

Two honest qualifications:

- **The second half of prediction 7 fails.** Rotated bases do not reduce nonmonotonicity; they
  *increase* it (59% -> 82% of probes at 30M, 70% -> 93% at 122M). Only the rate half holds.
- **The mean-criterion numbers for `pca`/`grad` are floor-censored.** They sit at or just above the
  rank grid's smallest entry (16 = 6.25% at 30M, 4.17% at 122M): 750 of 768 probes are at the floor
  for `pca` at 30M. Read them as upper bounds. The worst-offset numbers are not censored and are
  real measurements.

That `pca`, which never looks at the loss, beats `grad`, which is built from the first-order loss
effect, is worth noting rather than explaining away. A plausible reading is that at the rates where
the worst-offset criterion binds, truncation is far outside the first-order regime that `grad`
optimises for, so reconstruction fidelity is the better proxy. Testing that would need a basis
fitted at the operating rate rather than from a first-order expansion at full rank.

## Scope

30M and 122M only. 250M is not included: its checkpoint is 2.8 GB and the Lightning account ran out
of credits mid-session (see below), so it was not worth a third upload before deciding what to do
with this result.

## Compute provenance

Lightning became unusable during this work -- `lightning job run` failed at cost 0.0 even for
`echo hello` on CPU, so the free credits are exhausted; `notes/compute_fallback_policy.md`'s
primary route (Kaggle) was used instead. Kaggle assigns a Tesla P100 (sm_60) on this account, and
the preinstalled PyTorch is built for sm_70+, so every CUDA launch failed with "no kernel image is
available for execution on the device"; the kernel now installs a cu118 build first. Inputs are
split across two datasets -- `elasticmla-basis-ablation-inputs` (checkpoints and data) and
`elasticmla-analysis-code` (15 KB) -- so a code fix does not mean re-uploading ~600 MB.
Kernel: `softkleenex/elasticmla-basis-ablation` version 7; log at
`experiments/basis_dependence_kernel.log`.
