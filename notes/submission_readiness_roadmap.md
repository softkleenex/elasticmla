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
- Horizon-length sweep and pairwise-interaction residual audits proposed in
  `notes/theory_contextual_tail_rate.md` (falsifiable predictions 2 and 5) are designed but not run.
- Basis-dependence ablation: compare the current gradient-times-activation channel order against a
  PCA/SVD or learned orthogonal nested basis, to test whether the reported ranks are an artifact of
  channel-ordering choice.

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
- REMAINING: `append_packed_latents` still unpacks and repacks the *entire* cache history on every
  single-token decode step rather than appending only the new token in place -- an algorithmic
  limitation, not a vectorization one, and the next concrete target for closing the residual ~3x
  gap toward true O(1)-per-step incremental packing.
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

Package D's vectorization is now done and measured (168-201x -> ~3x). The next highest-value,
still-bounded steps are: (1) the append-in-place incremental packing fix to close the residual ~3x
gap, (2) **Package A**'s horizon-length sweep (reuses already-authenticated checkpoints/data, needs
no new training), or (3) extending the 250M benchmark and a first MHA/GQA baseline comparison.
Package C (more seeds, a second domain, a >=1B checkpoint) requires a resourcing decision (days of
paid GPU time) rather than more engineering effort and should be scoped explicitly before starting.
