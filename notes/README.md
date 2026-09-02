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

## Current, cited by the manuscript

- `exp0_v4_scale_comparison.md`, `exp1_122m_v4_results.md` -- corrected v4 mean/max endpoints.
- `risk_capacity_spectrum_results.md` -- the full three-scale risk-capacity spectrum (Section 5.1).
- `theory_contextual_tail_rate.md` -- the formal rate-allocation theory (Propositions 1-2).
- `fresh_confirmation_protocol.md`, `fresh_confirmation_results.md` -- 30M/122M pre-registered
  confirmation.
- `contextual_router_250m_results.md` -- the initial (confounded) 250M coarse-tier result.
- `contextual_router_250m_tier_granularity_diagnostic.md` -- the corrected 250M analysis
  (coarse vs fine tier grids); **supersedes the confounded framing in the previous note**.
- `causal_heuristic_baseline_results.md` -- the causal-heuristic comparison at 30M/122M.
- `measured_cache_memory_latency.md` -- measured T4 GPU peak memory/latency (Section 5.5).
- `submission_readiness_roadmap.md` -- outstanding work for a stronger venue.
- `literature_review.md`, `compute_fallback_policy.md` -- process/reference notes still in force.
- `exp1_v6_completion.md`, `exp2_250m_completion.md`, `lightning_exp1_v3_run.md`,
  `lightning_exp1_v4_run.md` -- training-run completion records (provenance, not findings).
