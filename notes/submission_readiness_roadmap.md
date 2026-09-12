# Submission-readiness roadmap (updated)

## Honest current tier

With three scales (30M/122M/250M), a formal rate-allocation theory, a fully pre-registered
routing evaluation at four scale/tier-grid configurations, causal-heuristic baselines, measured
GPU memory/latency, an independent fresh-eyes review pass (bugs found and fixed), and a compiling
LaTeX package, this is now solidly at **rigorous efficient-ML workshop / mechanistic-study**
quality. It is still not a main-track MLSys/NeurIPS/ICLR submission: the main remaining blockers
are realistic domain/scale coverage beyond TinyStories/250M, multiple seeds per configuration, and
an unfixed 168-201x decode-latency regression that blocks any systems/serving claim.

## Claim architecture (status)

1. **Theory -- DONE.** Nested latent prefixes as a basis-dependent operational rate code
   (Section 3.2); exact cache-byte affinity (Proposition 1); risk-capacity ordering across
   upper-tail loss criteria (Proposition 2); joint rate-distortion framing of the router objective;
   scope-limited straight-through-estimator statement. See `notes/theory_contextual_tail_rate.md`.
2. **Mechanism -- DONE at 3 scales.** Tail-capacity premium (~0.65-0.69, nearly scale-invariant);
   positive-part/cancellation diagnostic (mean-safe positions still exceed epsilon at 93-96% of
   records, refuting a "rare spike" story). Horizon-law and pair-interaction residual audits from
   the theory memo remain unimplemented (see Package A below).
3. **Empirics -- DONE, mixed result honestly reported.** Contextual routing beats random and
   shuffled placement at 30M/122M/250M-fine but fails its own pre-registered criterion at
   250M-coarse; against causal heuristics (position, lexical, frequency, type) the router wins only
   3 of 16 scale/tier-grid comparisons. Global-budget (not just router-conditioned-budget) static
   policies and a learned orthogonal (PCA/SVD) nested basis are still not compared (Package B).
4. **Systems -- partially done.** Persistent cache-byte reduction and a small (3.5-4.2%) peak-memory
   reduction are measured on a real T4 GPU (not just derived), but decode latency is 168-201x worse,
   traced to two specific unvectorized Python loop sites. No serving-benefit claim is possible until
   these are fixed (Package D).

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
- REMAINING: pairwise-interaction residual audit (prediction **5**) -- still designed but not run.
- REMAINING: basis-dependence ablation (prediction 7): compare the current
  gradient-times-activation channel order against a PCA/SVD or learned orthogonal nested basis, to
  test whether the reported ranks are an artifact of channel-ordering choice.

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

### E. Submission package -- DONE
- `manuscript/draft.md` (source of truth) + `manuscript/latex/main.tex`/`main.pdf` (compiled,
  13 pages, no LaTeX errors) via `manuscript/convert_to_latex.py`.
- Full reproducibility appendix: every result JSON carries checkpoint/data/policy/oracle SHA-256
  hashes; `audit_fresh_confirmation.py` and `audit_joint_training_replay.py` independently
  recompute every reported statistic and verify bit-exact deterministic retrains.
- Repository cleanup: `README.md`, `notes/README.md`, `archive/README.md` all current;
  superseded/withdrawn work moved to `archive/` and confirmed unreferenced.
- An independent "naive fresh reviewer" pass (no prior context) re-verified 15+ numbers from raw
  JSON, found one real P0 arithmetic-tally bug and one real P1 CI-rounding mismatch in the
  manuscript, both fixed (commit `c9de709`), plus two smaller code/doc issues (also fixed,
  `046818a`).

## Recommended next action

Package D is now done through the append-in-place fix: 168-201x -> ~3x (vectorization) -> 2.4x
(in-place append plus hot-path guard hoisting), all measured on T4 and reflected in the manuscript.
The next highest-value, still-bounded steps are: (1) **Package A**'s horizon-length sweep (reuses
already-authenticated checkpoints/data, needs no new training), (2) extending the benchmark to 250M
and adding a first MHA/GQA baseline comparison, or (3) removing the dense-reconstruction gather by
computing the per-head key/value projections directly against the ragged packed buffer -- the
largest remaining latency item, but a design change that should be scoped before it is started.
Package C (more seeds, a second domain, a >=1B checkpoint) requires a resourcing decision (days of
paid GPU time) rather than more engineering effort and should be scoped explicitly before starting.
