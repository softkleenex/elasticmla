# Submission-readiness roadmap (updated)

> **2026-10-21 update.** Content strengthened before choosing a venue: the Section 5.1 spectrum and
> the basis ablation were rerun on the public, production-trained DeepSeek-V2-Lite MLA latent
> (`notes/public_mla_deepseek_v2_lite.md`). The spectrum and cancellation signature replicate
> (premium 0.704 vs 0.646-0.651 here); the basis effect is ~11 pp there vs ~50 pp here. The
> manuscript (abstract, 4.1, 5.1 new table, 6, 7) now leads with this and demotes the GPT-2
> control. This partially answers Package C's ">=1B checkpoint / second domain" for the spectrum
> claim only; the router is still TinyStories-only. Next decision is still the venue.

> **2026-10-20 update -- read before the rest.** A pre-submission code review
> (`notes/code_review_2026-10_corrections.md`) found real errors that the earlier "DONE" items below
> did not catch: the 250M rank grid jumped 384 -> 512 (250M max rate 76.03% -> 71.78%, premium
> 0.689 -> 0.646, basis fall 53.9 -> 49.9 pp), the causal-heuristic comparison was not byte-matched
> (router now 6 wins / 8 ties / 2 losses; ties token identity), and the peak-memory reduction was a
> harness artifact (now 0.3-0.7%). All were rerun and the manuscript (now 16 pages), notes and
> Figure 2 updated; numbers below that conflict with that note are superseded. The basis material
> has now had its independent review. Still open from the review: nothing that changes a number;
> Package B/C/D items below are unchanged. Next decision is the venue (a workshop, per the tier
> assessment below), which sets the page limit -- the draft must shrink from 16 pages.

## Honest current tier

With three scales (30M/122M/250M), a formal rate-allocation theory (now including a basis-
dependence ablation that qualifies the headline tail-rate numbers and was stress-tested, and found
not to generalize, on public GPT-2 checkpoints), a fully pre-registered routing evaluation at four
scale/tier-grid configurations, causal-heuristic baselines, measured GPU memory/latency (decode
latency fixed from 168-201x down to ~2.4x), an independent fresh-eyes review pass (bugs found and
fixed), and a compiling LaTeX package, this is now solidly at **rigorous efficient-ML workshop /
mechanistic-study** quality -- arguably stronger than that given the basis-dependence self-
critique. It is still not a main-track MLSys/NeurIPS/ICLR submission: the main remaining blockers
are realistic domain/scale coverage beyond TinyStories/250M, multiple seeds per configuration, and
a still-unremoved dense-reconstruction gather that keeps even the improved ~2.4x decode-latency
regression from being a clean systems/serving claim.

## Claim architecture (status)

1. **Theory -- DONE.** Nested latent prefixes as a basis-dependent operational rate code
   (Section 3.2); exact cache-byte affinity (Proposition 1); risk-capacity ordering across
   upper-tail loss criteria (Proposition 2); joint rate-distortion framing of the router objective;
   scope-limited straight-through-estimator statement. See `notes/theory_contextual_tail_rate.md`.
2. **Mechanism -- DONE at 3 scales, including the falsifiable predictions from the theory memo.**
   Tail-capacity premium (~0.65-0.69, nearly scale-invariant); positive-part/cancellation
   diagnostic (mean-safe positions still exceed epsilon at 93-96% of records, refuting a "rare
   spike" story); horizon-length saturation (prediction 3); pair-interaction residual audit
   (prediction 5, additivity falsified but the violation is local/modest and grows with scale);
   and the basis-dependence ablation (prediction 7, see item 1 and Package A/F below).
3. **Empirics -- DONE, mixed result honestly reported.** Contextual routing beats random and
   shuffled placement at 30M/122M/250M-fine but fails its own pre-registered criterion at
   250M-coarse; against causal heuristics (position, lexical, frequency, type) the router wins only
   3 of 16 scale/tier-grid comparisons. Global-budget (not just router-conditioned-budget) static
   policies and a learned orthogonal (PCA/SVD) nested basis are still not compared (Package B).
4. **Systems -- mostly done.** Persistent cache-byte reduction and a small (3.5-4.2%) peak-memory
   reduction are measured on a real T4 GPU (not just derived). Decode latency, originally 168-201x
   worse (traced to three unvectorized Python loop sites), is now ~2.4x worse after vectorizing
   those loops and switching to an in-place packed-cache append; the remaining gap is the dense-
   reconstruction gather before every attention call, a design question rather than an optimization
   (Package D).

## Remaining work packages, in priority order

### A. Mechanism completion (moderate effort, high value)
- DONE: the matched-prefix horizon sweep (prediction **3**, not 2 -- the earlier entry here cited
  the wrong number; prediction 2 is the cancellation diagnostic, already done and in Section 5.1).
  It also turned out not to need running: `analyze_risk_capacity_spectrum.py` has emitted
  `horizon_capacity_spectrum` since `ae187c2` and all three published summaries carry it, so the
  prediction was testable from already-authenticated artifacts with no GPU. Confirmed at all three
  scales -- worst-offset `r*` strictly nondecreasing and saturating in `h`, signed mean
  non-monotone. See `notes/horizon_capacity_prediction3.md` and
  `experiments/analyze_horizon_capacity.py`.
- DONE: extended to h=128 at all three scales on T4 (`elasticmla-horizon-{30m,122m,250m}-0912b`).
  Saturation is now reached, not merely setting in: the final doubling moves +0.43/+0.93/+1.30 pp
  against +2.07/+2.78/+2.95 pp for h=16->32, still strictly nondecreasing. A position-matched H=32
  control shows the published tail-count table shifts by at most 1.73 pp with inconsistent sign and
  the tail-capacity premium by at most 0.009, so the published spectrum is not an artifact of which
  positions were eligible to be probed. One honest caveat recorded: past h=32 the signed-mean curve
  is censored at the rank grid's smallest entry (16) and its long-horizon values are bounds, not
  estimates.
- DONE: pairwise-interaction residual audit (prediction 5) at all three scales, 1008 pairs each.
  The second-order mixed difference is nonzero, so the additive special case is falsified, but the
  violation is local and modest (9.09%/12.17%/13.17% of the individual effects for adjacent
  positions at 30M/122M/250M, under 1.3% beyond separation 32) -- which converts the "Oracle scope"
  limitation from a qualitative worry into a measured quantity. It grows with scale and with
  truncation severity, so joint-rollout training becomes more necessary at larger scale, not less.
  An order of magnitude smaller than the basis effect. See `notes/pair_interaction_prediction5.md`.
- DONE, and it is a negative result for the reported rates: the basis-dependence ablation
  (prediction 7) at all three scales. Rotating the latent basis cuts the worst-offset required rate
  to 23.56%/23.25%/22.15% at 30M/122M/250M -- a fall of 50-54 pp, bootstrap intervals nowhere
  near overlapping. A large part of the paper's headline tail rate is therefore a cost of axis
  alignment, not a property of the latent code, and the fix is deployable (a fixed change of
  variables folded into W_DKV/W_UK/W_UV, same cache bytes). Prediction 7's second half fails:
  rotated bases increase nonmonotonicity rather than reducing it. The ablation's `coord` arm
  reproduces the published spectrum exactly, which is what caught a silent sampling bug in the
  first version. See `notes/basis_dependence_prediction7.md`.
- DONE: the basis fitted to the model's own metric (`model_metric`: principal directions of
  `A^{1/2} E[cc^T] A^{1/2}` for `A = W_UK^T W_UK + W_UV^T W_UV`), which provably minimises
  output-space reconstruction error at every rank. It confirms the diagnosis of `grad` -- both
  reconstruction criteria beat the first-order loss criterion by 10-12 pp at every scale -- but
  gives no reliable gain over plain PCA (wins by <1 pp at 30M/250M, indistinguishable at 122M, about
  half of probes tied). Which rotation is used matters far less than rotating at all. Comparisons
  are paired per-probe, since all four bases are measured on identical probes.
- The manuscript reports this in Section 5.1, a Section 6 "Basis scope" limitation, and the
  abstract (which states the coordinate-basis 74.07%/72.92%/76.03% figures *and* the rotated-basis
  22-24% figures together, explicitly framed as "a basis ablation shows this tail requirement is
  substantially a cost of that coordinate alignment"), keeping the coordinate-basis numbers as the
  deployed ones throughout.

### B. Stronger controls (mostly done)
- DONE: exact-histogram position, lexical-identity, token-frequency, and token-type controls at all
  four scale/tier-grid configurations (`experiments/evaluate_causal_heuristic_routers.py`).
- REMAINING: a separately-tuned *global*-budget static policy (one fixed rank for the whole
  dataset, not conditioned on the router's per-sequence realized budget) and a learned nested-basis
  comparison are still missing.

### C. Replication (largest remaining gap)
- Still one router seed and one base checkpoint per scale/tier-grid configuration.
- Still one domain (TinyStories) and short contexts (256-384 tokens).
- 250M is the largest scale tested; no >=1B checkpoint. This is the single largest blocker to a
  main-track submission and requires a genuinely larger compute budget (a multi-day, not
  multi-hour, Kaggle/Lightning campaign) to address properly.

### D. Systems -- root cause diagnosed AND fixed; residual gap identified
- DONE: `code/elastic_mla/elastic_cache.py`'s `pack_latents`/`unpack_latents`/`append_packed_latents`
  and `code/elastic_mla/mla.py`'s `forward_cached_packed` rank-mask construction were rewritten as
  vectorized PyTorch operations (no algorithm/representation change; validated by 500 random-trial
  cross-checks vs. the originals plus the full unit suite). This cut the measured decode-latency
  regression from 168-201x to ~3x at both benchmarked scales (T4 GPU, see
  `notes/measured_cache_memory_latency.md`).
- DONE (implementation, pending T4 re-measurement): `append_packed_latents` now appends in place
  instead of unpacking and repacking the whole history, enabled by switching the flat buffer to a
  sequence-major `(sequence, batch)` layout. Profiling then showed the real dominant cost was
  per-call validation synchronisation rather than the repacking, so the guards were moved to where
  their guarantee is established (full permutation validation on the prefill that creates the
  cache; equality against the cache's stored order on every decode step) and the bounds/offset
  audits were fused. No guard was removed. Verified bit-identical: 54/54 unit tests, a 500-trial
  cross-check against the previous layout, and identical 30M router decode logits/ranks/bytes.
  Measured on MPS the packed step is ~1.6x faster (regression versus full MLA 9.0x -> 5.7x
  uniform, 9.9x -> 6.5x router); see `notes/measured_cache_memory_latency.md` Update 3.
- DONE: measured on T4 with a paired design (both source versions run alternately, 3 repeats each,
  inside one job per scale). Regression versus full MLA 3.09x -> 2.42x at 30M and 2.99x -> 2.37x at
  122M; the untouched dense control moved <=1% within each job, and the pre-fix column reproduced
  the published 3.00x/2.96x. Manuscript abstract, Section 5.5, Section 6 system-scope paragraph and
  conclusion updated from "roughly 3x" to 2.4x; LaTeX regenerated and compiles clean (14 pages).
- REMAINING: the packed path still reconstructs the dense latent history via a gather before every
  attention call -- now the dominant per-step cost (45% of the packed step on MPS). Removing it is
  a design/claim question, not an optimization: either a dense working set stays resident (which
  makes the persistent-byte saving irrelevant at decode time) or `W_UK`/`W_UV` are computed
  directly against the ragged packed buffer.
- Still no comparison against optimized MHA/GQA/FlashMLA baselines, and peak memory/latency are
  only benchmarked at 30M/122M (not 250M).

### E. Submission package -- DONE, pending re-verification after the basis-dependence additions
- `manuscript/draft.md` (source of truth) + `manuscript/latex/main.tex`/`main.pdf` (compiled with
  no LaTeX errors via `manuscript/convert_to_latex.py`; page count has grown with the Section 5.1
  and Section 6 basis-dependence material added since the last confirmed 13/14-page build -- rerun
  `tectonic main.tex` and re-check for table/figure overflow before citing a specific page count).
- Full reproducibility appendix: every result JSON carries checkpoint/data/policy/oracle SHA-256
  hashes; `audit_fresh_confirmation.py` and `audit_joint_training_replay.py` independently
  recompute every reported statistic and verify bit-exact deterministic retrains.
- Repository cleanup: `README.md`, `notes/README.md`, `archive/README.md` all current;
  superseded/withdrawn work moved to `archive/` and confirmed unreferenced.
- An independent "naive fresh reviewer" pass (no prior context) re-verified 15+ numbers from raw
  JSON, found one real P0 arithmetic-tally bug and one real P1 CI-rounding mismatch in the
  manuscript, both fixed (commit `c9de709`), plus two smaller code/doc issues (also fixed,
  `046818a`).

**Package A is complete.** Predictions 3, 5 and 7 are all measured at three scales, and the
operating-rate basis question is answered. The mechanism findings are of very different size: basis
choice moves the required rate by ~50 pp, the first-order-versus-reconstruction criterion by
10-12 pp, isolated-versus-joint labelling by single-digit percentages of the effect, and refining
the reconstruction metric by under a point.

### F. Scope control on the basis result (done, and it bites)

The basis finding was tested on public pretrained checkpoints and does **not** generalise:
GPT-2 and GPT-2 medium give a 1.09-1.14x coordinate-basis overstatement against the 1.65-3.2x
measured here. Chasing the reason produced a stronger result than the one that failed --
no fixed linear projection (coordinate, principal, or the score-optimal oblique frame)
compresses those caches much below 50 of 64 dimensions, even though GPT-2 medium holds 94.9%
of its key energy in 8 directions. Low-rank second moments do not imply a compressible KV
cache. See `notes/low_rank_does_not_mean_compressible.md`.

Two consequences for the manuscript. The Section 5.1/Section 6 basis material must state that
the effect is measured on this project's models and does not transfer to public checkpoints.
And the most economical explanation -- that 3,000-step TinyStories models are unrepresentatively
compressible -- is a caution about the whole empirical base, not only about this measurement.

## Recommended next action

Package D is now done through the append-in-place fix: 168-201x -> ~3x (vectorization) -> 2.4x
(in-place append plus hot-path guard hoisting), all measured on T4 and reflected in the manuscript.
Package A is complete (predictions 3, 5, 7 all measured at three scales, bounded on public
checkpoints). The next highest-value, still-bounded steps are: (1) extending the benchmark to 250M
and adding a first MHA/GQA baseline comparison, or (2) removing the dense-reconstruction gather by
computing the per-head key/value projections directly against the ragged packed buffer -- now the
single largest remaining latency item (45% of the packed step on MPS per Package D's measurement),
but a design change that should be scoped before it is started, since it trades the persistent-byte
saving's relevance at decode time against further speed. Package C (more seeds, a second domain, a
>=1B checkpoint) requires a resourcing decision (days of paid GPU time) rather than more
engineering effort and should be scoped explicitly before starting. The commits from `f465c34`
to `10f1580` were pushed to `origin/master` on 2026-10-17 after a fresh 15-page `tectonic`
rebuild (no errors, no overfull boxes). **Still outstanding before any of the above: a second
independent review pass on the basis-dependence material, given its significance to the paper's
headline claim -- those commits have not yet had a fresh-reviewer pass.**
