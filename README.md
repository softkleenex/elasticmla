# ElasticMLA: Contextual Tail-Rate Allocation for Multi-Head Latent Attention

Working repository for the ElasticMLA project: token-wise variable-width latent caching for
Multi-Head Latent Attention (MLA), a formal rate-allocation theory for the resulting
risk-capacity spectrum, and a rigorously audited, pre-registered evaluation of a learned
contextual router against random, shuffled, and simple causal-heuristic baselines at three model
scales (30M, 122M, 250M parameters).

**Start here:** `manuscript/draft.md` is the paper. Its headline claims are, in order of how
robust they are:

1. **Risk-capacity spectrum (robust, three scales, does not involve the router).** Required
   latent rate rises monotonically from ~8-9% of full width at a mean-loss criterion to ~73-76% at
   a worst-offset criterion, with a nearly scale-invariant tail-capacity premium (~0.65-0.69). The
   mean/tail gap comes from pervasive cancellation across the reuse horizon, not rare spikes.
2. **Router beats random matched-budget allocation (real, but tier-grid-dependent).** True at
   30M and 122M; fails with a coarse tier grid at 250M and succeeds again with a finer grid at
   250M -- diagnosed and reported as a tier-resolution confound, not spun as a clean scale trend.
3. **Router does not beat simple causal heuristics (the main negative finding).** Across four
   tested scale/tier-grid configurations, the router shows a confident win in only 2 of 16
   scale-heuristic comparisons, and loses more decisively at finer tier resolution.
4. **Decode latency is 168-201x slower for the packed path (measured, unfixed).** Persistent cache
   bytes and a small peak-memory reduction are real and measured on a T4 GPU; decode speed is not,
   because of an unvectorized Python-loop cache reconstruction.

## Repository layout

- `manuscript/draft.md` -- the paper (theory, method, results, limitations, references).
- `code/elastic_mla/` -- the MLA / packed-cache / contextual-router implementation.
  - `mla.py` (dense + packed cached attention), `model.py` (`MLAGPT`), `elastic_cache.py` (packed
    prefix storage), `router.py` (`TieredRankRouter`, `ElasticMLAGPT`, `ContextualElasticMLAGPT`).
- `experiments/` -- the current, citable analysis/training/evaluation pipeline (see below) plus
  one results subdirectory per scale (`exp0_rank_variance/`, `exp1_rank_variance_122m/`,
  `exp2_rank_variance_250m/`, `contextual_router_{30m,122m,250m}/`).
- `notes/` -- a dated research log. Read chronologically; later notes correct or supersede earlier
  ones (e.g. the v4 methodology note supersedes v2/v3; the 250M tier-granularity diagnostic
  supersedes the first, confounded 250M result note). This is a lab notebook, not a curated
  "results" folder -- `manuscript/draft.md` is the curated, authoritative summary.
- `figures/` -- the two figures actually used in the manuscript
  (`elasticmla_main_results.*`, `elasticmla_risk_spectrum.*`) plus their generating scripts.
- `tests/` -- unit tests for the packed cache, routers, analysis helpers, and audit scripts.
  Run with `uv run python -m unittest discover -s tests`.
- `archive/` -- superseded or withdrawn exploratory work, kept for provenance only. Not cited by
  the manuscript. See `archive/README.md`.
- `papers/` -- the literature-review arXiv index used while writing the related-work section.

## The current pipeline, in dependency order

1. `experiments/train_exp0.py` -- trains the 30M base MLA checkpoint (30M is the only checkpoint
   trained locally on Apple Silicon MPS; 122M and 250M were trained on Kaggle via
   `code/kaggle_notebook/` and `code/kaggle_notebook_exp2_250m/`).
2. `experiments/analyze_rank_variance_v4.py` -- corrected per-token future-loss rank intervention
   (fixes the off-by-one horizon bug documented in `notes/exp0_v3_corrected_methodology.md`).
   Produces the `exp{0,1,2}_..._122m/250m}/results_v4/exp0_v4_{summary,records}.json` pair used
   as the seeded-window source of truth for every downstream step.
3. `experiments/analyze_risk_capacity_spectrum.py` -- the full upper-tail risk-capacity spectrum
   (six tail levels, not just mean/max), tail-capacity premium, and the positive-part/cancellation
   diagnostic. Independent of steps 4+; this is what backs Section 5.1 of the paper.
4. `experiments/generate_contextual_oracle_v1.py` -- layer-0-full, shared-downstream-tier oracle
   labels aligned to the contextual router's actual intervention scope.
5. `experiments/train_contextual_router.py` -- supervised `router_max.pt` on isolated-position
   oracle labels; also fixes the reproducible 16/4/4 train/val/test sequence split.
6. `experiments/train_joint_rollout_router.py` -- straight-through, joint-rollout training of the
   deployed hard-tier router under a rank-penalized Lagrangian; sweeps `--rank-lambda`.
7. `experiments/evaluate_joint_rollout_sweep.py` -- selects one policy per scale using only the
   original 16 training / 4 validation sequences (never the frozen fresh-confirmation windows).
8. `experiments/confirm_fresh_contextual_router.py` -- the one-shot, pre-registered confirmation
   on 24 new nonoverlapping windows, run only after `experiments/fresh_confirmation_manifest.json`
   is committed with that scale's frozen policy/oracle hashes.
9. `experiments/audit_fresh_confirmation.py` and `experiments/audit_joint_training_replay.py` --
   independent recomputation of every reported statistic and a from-scratch bit-exact retrain
   check, respectively. Both must report `"status": "passed"` for a result to be cited.
10. `experiments/evaluate_causal_heuristic_routers.py` -- position/lexical/rarity/type causal
    baselines fit only on the training/validation split, evaluated once on the same frozen fresh
    windows at the router's own byte budget.
11. `experiments/benchmark_cache_memory_latency.py` -- measured T4 GPU peak memory and decode
    latency for full/packed-uniform/packed-router configurations.

Every script in this list authenticates its inputs by SHA-256 against the files that produced
them and refuses to run (or the corresponding audit script refuses to pass) if provenance does
not match. Cite `experiments/fresh_confirmation_manifest.json` and the per-scale
`*_audit.json` / `*_replay_audit*.json` files as evidence that a reported number is what it claims
to be, not just the raw result JSON.

## Environment

```bash
uv sync
uv run python -m unittest discover -s tests
```

GPU-heavy steps (steps 2-4, 8, 11 above) were run on Kaggle (P100, `code/kaggle_notebook*`) and
Lightning AI (T4, via `lightning job run` / `lightning studio ssh`); see
`notes/compute_fallback_policy.md` for the compute-provider fallback policy used throughout.
