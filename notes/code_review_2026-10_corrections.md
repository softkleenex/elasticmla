# Pre-submission code review and the corrections it forced (2026-10-17 to 2026-10-20)

Before any submission, the whole codebase was reviewed by five independent reviewers with no prior
context (library; data/training/measurement; router pipeline and pre-registration; systems,
pair-interaction and provenance; basis-dependence). Each finding below was re-verified by hand
before acting on it. This note supersedes the affected numbers in earlier notes; each of those
carries a dated pointer here.

## What was wrong, and what it changed

### 1. The 250M rank grid jumped 384 -> 512 (changed headline numbers)

`DEFAULT_RANK_GRID` stopped at 384 and `normalize_rank_grid` only appended `d_c`, so at 250M
(`d_c`=512) every requirement in (384, 512] was rounded up to full rank. The v4 source-of-truth run
had used the full grid explicitly; the v5 spectrum, the horizon runs and the basis ablation used
the default. Fixed with `default_rank_grid(d_c)` (steps of 32 up to `d_c`; identical for 30M/122M,
so their numbers are unchanged) and a test. 250M rerun on Kaggle T4
(`softkleenex/elasticmla-review-spectrum250`, `-basis250`), same seeds, checkpoint, data and channel
orders (verified equal):

| 250M | published | corrected |
|---|---:|---:|
| k=16 / 8 / 4 / 2 | 22.53 / 40.14 / 57.23 / 69.18% | 22.12 / 38.60 / 54.16 / 64.95% |
| max (k=1) | 76.03% | **71.78%** |
| tail-capacity premium | 0.6888 [0.6647, 0.7129] | **0.6464 [0.6257, 0.6665]** |
| mean (k=32), positive-part diagnostic | 7.15%, 2.15x / 96.1% | unchanged |
| basis: pca max / paired fall | 22.15% / 53.9 pp | 21.89% / 49.9 pp |
| basis: probes needing all 512 (coord / pca) | 389 / 13 | 77 / 0 |
| horizon h=128 max (A) | 78.95% | 74.28% |

The premium is now 0.651/0.647/0.646 across scales -- more scale-consistent than reported, not
less. Prediction 3 still holds at every scale; the position control still moves the spectrum by at
most 1.73 pp and the premium by at most 0.009. The corrected 250M raw records are kept under
`kaggle_output/review_reruns/` (gitignored) and authenticated by `records_sha256`.

### 2. The causal-heuristic comparison was not byte-matched (reversed Section 5.4)

`evaluate_causal_heuristic_routers.py` ran each rule at its own validation-selected bias. Its
bytes differed from the router's by -6.5% to +13.0%, and every published win/loss lined up with
that difference (the three router wins were all against rules spending fewer bytes, all nine
losses against rules spending more). New modes, run locally on MPS with the router loss recomputed
on the same device (drift vs stored <= 7.2e-7):

- `--placement router-bytes`: the rule gets only the router's per-sequence rank total and chooses
  its own integer histogram. Router **6 wins / 8 ties / 2 losses**; both losses at 250M-coarse.
- `--placement router-histogram`: the rule places the router's own tier multiset. Router 14 / 2 / 0.

Lexical identity ties the router at 30M, 122M and 250M-fine and beats it at 250M-coarse, so the
router is not shown to use context beyond the current token. Sanity checks: placing by the router's
own ranks reproduces its loss exactly; reversing the order raises loss 2.18 -> 2.37.

### 3. The peak-memory reduction was a harness artifact (Section 5.5)

`run_full`/`run_packed_uniform` kept the (B,S,V) prefill logits referenced into decode
(`_, cache = ...`) while `run_router` rebound `_` and freed them. Fixed (all paths keep only the
cache; checkpoint loaded to CPU; no host sync in the router's timed loop). T4 rerun, 3 repeats
(`softkleenex/elasticmla-review-benchmark`): router peak 99.7% (30M) / 99.3% (122M) of full MLA,
not 96.5% / 95.8%. Cache-byte ratios (67.4% / 59.6%) and the ~2.4x decode regression reproduce.

### 4. Pair-interaction: mislabeled headline and an unsupported scale trend

9.09/12.17/13.17% are separation-1 means pooled over the three ranks, not "the heaviest
truncation" (rank 16 alone: 16.79/23.28/22.78%). The ranks are fixed absolute values, so they are a
different fraction of `d_c` per scale; at rank 16, 250M (7.98%) is below 122M (8.41%). The "grows
with scale" claim is withdrawn.

### 5. Disclosures the manuscript was missing

- 250M saw 55.3M training tokens (3000 x 48 x 384), not 288M; fewer than 122M (147.5M).
- The 250M-fine confirmation reused the 250M-coarse fresh windows (same seed and exclusions).
- At 122M the coarse grid produced no validation-feasible candidate; the fine grid replaced it.
- For 30M/122M the pre-registration was the `a4bcc7f` protocol note plus policy files; the JSON
  manifest arrived with the results in `080af82`.
- The fresh-confirmation audit re-checks arithmetic but does not rerun the model; the replay
  audit's retrained policy files are not committed.
- The public-checkpoint bound on the basis result (1.06-1.30x on GPT-2 vs 1.65-3.22x here,
  like-for-like coord/pca over 0.05-0.5 nats) was absent from the manuscript.

### 6. Code and test fixes without number changes

- `analyze_public_kv_basis.py` recorded no provenance; it now stores model commit hash, a
  state-dict SHA-256, dataset fingerprints, token-id hashes, source hash, seed and versions, plus
  the top-8 key-energy share over every head (the published 94.9% was layer 0 head 0 only).
  Rerun on Kaggle T4 (`softkleenex/elasticmla-review-publickv`): every required-rank figure
  reproduces the earlier MPS run to 0.01; over all heads the top-8 key-energy share is mean
  77.3% / 76.6% (GPT-2 / GPT-2-medium), range 40-99%. The committed JSONs now carry provenance.
- The STE test passed with the estimator removed (the rank penalty alone reaches the router); it
  now backprops the LM term only, with a control test.
- New tests: truncated packed decode vs the analysis rank-mask path (prefill + decode, free
  per-token per-layer ranks, 1e-10 in float64); exact bytes after decode appends; grid coverage;
  `match_total`.
- The library itself (MLA math, RoPE, packed/dense equivalence, Proposition 1 byte count, router
  causality, STE gradient) was verified correct numerically; no bug found there.

## What was verified correct

Horizon indexing (window `[p, p+32)`, zero delta before p), data disjointness (val is the last 2%
of the stream; 32-gram train overlap 0.29%/0.02%), sampler reconstruction at every scale, the main
router table and its pass/fail labels, exact-byte static and shuffle controls, manifest-before-
result order for both 250M configurations, fresh-window separation, policy selection on
validation only, every 30M/122M spectrum number, all figures regenerate pixel-identically.

## Reruns

`code/kaggle_review_reruns/` (runner template + push script) is the committed harness; the code
dataset is `softkleenex/elasticmla-review-reruns-code`. Every Kaggle job ran the scripts of commit
`28e44ab` (byte-compared after download).
