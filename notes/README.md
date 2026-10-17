# Research log index

This is a dated lab notebook, not a curated results folder. Read chronologically; later notes
correct earlier ones. `manuscript/draft.md` is the curated, authoritative summary -- when in
doubt, trust the manuscript over any single note here.

## Superseded / historical (kept for provenance, not cited as current evidence)

- `exp0_findings.md`, `exp0_layerwise_comparison.md`, `exp0_v2_corrected_methodology.md` --
  early rank-variance methodology, superseded by v4.
- `exp0_v3_corrected_methodology.md` -- documents the off-by-one horizon bug found and fixed in v4.
  Its own v3 numbers were withdrawn.
- `contextual_router_30m_poc.md`, `contextual_router_122m_poc.md`, `global_router_poc.md` --
  pre-registration-era PoC router evaluations, superseded by the joint-rollout pipeline and the
  pre-registered fresh-window confirmation.
- `codex_review2_findings.md`, `cached_decode_prototype.md`, `packed_tiered_cache.md` --
  early implementation notes.
- `causal_heuristic_baseline_results.md` -- covers only 30M/122M causal-heuristic results; its
  own trailing "Update" section flags this and points to the complete 4-configuration tally
  (below). Read `contextual_router_250m_tier_granularity_diagnostic.md` for the full picture.

## Current, cited by the manuscript

- `exp0_v4_scale_comparison.md`, `exp1_122m_v4_results.md` -- corrected v4 mean/max endpoints.
- `risk_capacity_spectrum_results.md` -- the full three-scale risk-capacity spectrum (Section 5.1).
- `theory_contextual_tail_rate.md` -- the formal rate-allocation theory (Propositions 1-2).
- `fresh_confirmation_protocol.md`, `fresh_confirmation_results.md` -- 30M/122M pre-registered
  confirmation.
- `contextual_router_250m_results.md` -- the initial 250M coarse-tier confirmation (fails its own
  criterion) and its causal-heuristic table. Historically confounded with scale; see next entry.
- `contextual_router_250m_tier_granularity_diagnostic.md` -- adds the 250M-fine confirmation
  (succeeds), separates the tier-granularity confound from a pure scale effect, and gives the
  complete, correct 3-of-16 causal-heuristic tally across all four scale/tier configurations.
  **This is the authoritative source for the 250M and cross-scale causal-heuristic story.**
- `measured_cache_memory_latency.md` -- three sections, read all of them: the original T4
  benchmark (168-201x decode slowdown, root-caused to three Python loop sites); the
  post-vectorization T4 benchmark (~3x, same cache-byte/peak-memory numbers, from a different
  Lightning job); and Update 3, the in-place incremental append plus hot-path validation-sync
  removal, with both an MPS development measurement and a later **paired T4 measurement**
  (~3x -> ~2.4x at both 30M/122M, same GPU instance for before/after, bit-identical decode output)
  that **is** reflected in the manuscript.
- `horizon_capacity_prediction3.md` -- theory prediction 3 (matched-prefix horizon sweep), confirmed
  at three scales from already-published artifacts plus a T4 extension to h=128.
- `low_rank_does_not_mean_compressible.md` -- **bounds the basis result.** On public
  pretrained checkpoints (GPT-2, GPT-2 medium) the coordinate-basis overstatement is only
  1.1x, not the 1.65-3.2x seen here, and no fixed linear projection compresses the cache much
  below 50 of 64 dimensions even when 95% of the key energy sits in 8 directions. Read it
  alongside the two basis notes below; it is the control that keeps them honest.
- `basis_dependence_prediction7.md` -- theory prediction 7 (basis dependence). **A negative result
  for Section 5.1's headline rates**: rotating the latent basis cuts the worst-offset required rate
  by 50-54 pp at all three scales.
- `pair_interaction_prediction5.md` -- theory prediction 5 (pair-interaction residuals). Additivity
  is falsified but the violation is local and modest, and grows with scale.
- `submission_readiness_roadmap.md` -- current status per claim/work-package, and the recommended
  next action. **Read this first when resuming work in a new session.**
- `literature_review.md`, `compute_fallback_policy.md`, `resources_check.md` -- process/reference
  notes still in force (credentials/compute-provider status, literature survey).
- `exp1_v6_completion.md`, `exp2_250m_completion.md`, `lightning_exp1_v3_run.md`,
  `lightning_exp1_v4_run.md` -- training-run completion records (provenance, not findings).
