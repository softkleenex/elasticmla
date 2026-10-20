# Basis dependence of the nested latent code: theory prediction 7 (2026-09-13)

> **Correction (2026-10-20, `notes/code_review_2026-10_corrections.md`).** The 250M row below used a rank grid that jumped 384 -> 512. Corrected 250M: coord max 71.78%, pca 21.89%, model_metric 21.33%; paired pca-coord -49.9 pp [-51.3, -48.5]; model_metric-pca -0.57 [-1.03, -0.12]; probes needing all 512 channels 77 (coord) vs 0 (pca); any-rise nonmonotonicity 78.0% -> 98.4%. Also: 'increases nonmonotonicity' means more probes with any rise -- by mean violation magnitude rotated bases are lower at 30M and 122M; the full-rank rotation check is <= 3.1e-5 over all runs (not 2.4e-5); the coord arm reproduces the published endpoints and histograms, not every intermediate tail count. Scope: the effect does not transfer to public GPT-2 checkpoints (low_rank_does_not_mean_compressible.md).


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
- `model_metric` -- the latent only reaches the model through `W_UK c` and `W_UV c`, so what a
  truncation costs is reconstruction error *as the model sees it*. With
  `A = W_UK^T W_UK + W_UV^T W_UV` and `U` the eigenvectors of `A^{1/2} E[cc^T] A^{1/2}`, storing
  `T c` for `T = U^T A^{1/2}` makes the cost of dropping the last `d_c - k` coordinates exactly the
  tail sum of those eigenvalues -- the minimum achievable at that rank. Unlike the other three,
  `T` is not orthogonal: it is a change of frame rather than a rotation, so it preserves the packed
  cache's float count but not vector norms (which would matter if the cache were ever quantized).
  This goes beyond prediction 7's "orthogonal basis" and is reported as such.

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

768 probes per scale, H=32, epsilon=0.1. Worst-offset `r*` as a fraction of `d_c`:

| basis | 30M | 122M | 250M |
|---|---:|---:|---:|
| `coord` (published) | 74.07% [72.10, 75.72] | 72.92% [70.96, 74.95] | 76.03% [73.42, 78.72] |
| `grad` | 34.29% [32.90, 35.79] | 35.75% [33.88, 37.69] | 32.49% [30.51, 34.69] |
| `pca` | 23.56% [22.76, 24.39] | 23.25% [21.55, 24.95] | 22.15% [20.48, 23.74] |
| `model_metric` | **22.62%** [21.70, 23.53] | 23.68% [21.91, 25.48] | **21.46%** [19.83, 22.97] |

Every basis is measured on the *same* probes, so the comparison that the design supports is the
paired per-probe difference, not the overlap of two marginal intervals. Differencing the margins
would throw the pairing away, and probe-to-probe variation is far larger than the gaps between
rotations. Paired differences in percentage points of `d_c` (`*` = 95% sequence-cluster bootstrap
interval excludes zero):

| comparison | 30M | 122M | 250M |
|---|---|---|---|
| `pca` - `coord` | -50.51 [-51.97, -48.90] * | -49.66 [-50.88, -48.26] * | -53.88 [-55.74, -52.11] * |
| `grad` - `coord` | -39.78 [-41.76, -37.74] * | -37.17 [-38.53, -35.93] * | -43.54 [-45.42, -41.69] * |
| `model_metric` - `coord` | -51.45 [-52.91, -49.73] * | -49.24 [-50.74, -47.69] * | -54.57 [-56.36, -52.80] * |
| `pca` - `grad` | -10.73 [-12.13, -9.44] * | -12.49 [-13.34, -11.64] * | -10.34 [-11.76, -9.07] * |
| `model_metric` - `pca` | -0.94 [-1.60, -0.18] * | **+0.42 [-0.31, +1.13]** | -0.70 [-1.25, -0.16] * |

Three findings, in descending order of how solid they are.

**1. Rotating at all is what matters, and the effect is large and consistent.** Every rotation beats
the coordinate permutation by 37-55 pp at every scale, with intervals far from zero. The `r*`
histograms show the mechanism: under `coord` at 30M, 410 of 768 probes need at least 224 of 256
channels, while under `pca` only 3 do; at 250M `coord` needs the entire 512-wide latent for 389 of
768 probes against 13 for the rotated frame.

**2. Reconstruction criteria beat the first-order loss criterion, consistently.** `pca` is 10-12 pp
below `grad` at every scale. This is informative rather than incidental: `grad` orders directions by
the *first-order* loss effect measured at full rank, but the worst-offset criterion binds at roughly
a quarter of full width, far outside the regime where that expansion holds. A criterion that ignores
the loss entirely and just reconstructs the latent does better than one built from a linearisation
evaluated in the wrong place.

**3. The theoretically optimal frame gives no reliable practical gain over plain PCA.**
`model_metric` provably minimises output-space reconstruction error at every rank -- this is checked
directly in `tests/test_basis_dependence.py`, where the measured error equals the predicted
eigenvalue tail -- and it does win at 30M and 250M. But at 122M the paired difference is +0.42 pp
with an interval covering zero, i.e. no better than `pca` and nominally slightly worse. Roughly half
the probes tie at every scale (54.8%/52.3%/48.6%). The honest reading is that being optimal for L2
reconstruction in the model's metric does not transfer into a reliable advantage under a
loss-threshold criterion, which is a different objective. It is not a reason to prefer the more
complicated frame.

Two honest qualifications:

- **The second half of prediction 7 fails.** Rotated bases do not reduce nonmonotonicity; they
  *increase* it (59% -> 82% of probes at 30M, 70% -> 93% at 122M, 72% -> 97% at 250M), and the
  increase itself grows with scale. Only the rate half holds.
- **The mean-criterion numbers for the rotated bases are floor-censored.** They sit at or just
  above the rank grid's smallest entry (16, i.e. 6.25% at 30M, 4.17% at 122M, 3.12% at 250M): 750
  of 768 probes are at the floor for `pca` at 30M. Read them as upper bounds. The worst-offset
  numbers are not censored and are real measurements.

The earlier version of this note proposed testing the `pca`-beats-`grad` result with "a basis fitted
at the operating rate rather than from a first-order expansion at full rank". `model_metric` is that
test, in its cleanest available form: it is fitted to the metric the model actually applies to the
latent, and is provably optimal for reconstruction under that metric at every rank. It confirms the
diagnosis of `grad` -- both reconstruction criteria beat it by 10-12 pp -- but it does *not* deliver
a further gain over plain `pca`. Whatever separates a good rotation from a mediocre one here, it is
not captured by refining the reconstruction metric.

## Scope

All three scales (30M, 122M, 250M), matching the rest of the paper's headline results.

## Compute provenance

Lightning became unusable during this work -- `lightning job run` failed at cost 0.0 even for
`echo hello` on CPU, so the free credits are exhausted; `notes/compute_fallback_policy.md`'s
primary route (Kaggle) was used instead. Kaggle assigns a Tesla P100 (sm_60) on this account, and
the preinstalled PyTorch is built for sm_70+, so every CUDA launch failed with "no kernel image is
available for execution on the device"; the kernel now installs a cu118 build first. Inputs are
split across two datasets -- `elasticmla-basis-ablation-inputs` (checkpoints and data) and
`elasticmla-analysis-code` (15 KB) -- so a code fix does not mean re-uploading ~600 MB.
The four-basis re-run adds paired per-probe comparisons; a bootstrap seed derived from a
comparison's position in a list would have changed with the `--bases` subset, so seeds are derived
from the comparison's own name and the artifacts were regenerated so their `analysis_source_sha256`
matches the committed script. The 30M analysis was run on both Apple MPS and a Kaggle P100 and the
two agree to 0.0000 pp for three of the four bases (0.016 pp for `grad`).

The 250M checkpoint lives in its own dataset (`elasticmla-250m-checkpoint`, 2.8 GB) because
uploading a single top-level directory makes Kaggle extract its *contents* to the dataset root, so
that scale arrives as `<root>/ckpt/latest.pt` rather than `<root>/exp2_250m/ckpt/latest.pt`; the
kernel accepts both layouts. Kernels: `softkleenex/elasticmla-basis-ablation` version 7 (30M, 122M)
and version 10 (250M); 30M/122M log at `experiments/basis_dependence_kernel.log`.
