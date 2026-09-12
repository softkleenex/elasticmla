# ElasticMLA: Contextual Tail-Rate Allocation for Multi-Head Latent Attention

Working repository for the ElasticMLA project: token-wise variable-width latent caching for
Multi-Head Latent Attention (MLA), a formal rate-allocation theory for the resulting
risk-capacity spectrum, and a rigorously audited, pre-registered evaluation of a learned
contextual router against random, shuffled, and simple causal-heuristic baselines at three model
scales (30M, 122M, 250M parameters).

**Start here:** `manuscript/draft.md` is the paper (source of truth); `manuscript/latex/main.pdf`
is a compiled LaTeX version (regenerate via `uv run python manuscript/convert_to_latex.py &&
cd manuscript/latex && tectonic main.tex` after editing the markdown). Its headline claims, in
order of how robust they are:

1. **Risk-capacity spectrum (robust, three scales, does not involve the router).** Required
   latent rate rises monotonically from ~8-9% of full width at a mean-loss criterion to ~73-76% at
   a worst-offset criterion, with a nearly scale-invariant tail-capacity premium (~0.65-0.69). The
   mean/tail gap comes from pervasive cancellation across the reuse horizon, not rare spikes.
2. **Router beats random matched-budget allocation (real, but tier-grid-dependent).** True at
   30M and 122M; fails with a coarse tier grid at 250M and succeeds again with a finer grid at
   250M -- diagnosed and reported as a tier-resolution confound, not spun as a clean scale trend.
3. **Router does not beat simple causal heuristics (the main negative finding).** Across four
   tested scale/tier-grid configurations, the router shows a confident win in only 3 of 16
   scale-heuristic comparisons (all at coarse tier grids: 2 at 30M, 1 at 250M-coarse), and loses
   more decisively at finer tier resolution.
4. **Decode latency was 168-201x slower for the packed path; vectorizing three Python loop sites
   cut this to ~3x (measured, fixed).** Persistent cache bytes and a small peak-memory reduction
   were always real; the latency regression was root-caused to three unvectorized per-token Python
   loops and mostly (not fully) fixed -- see `notes/measured_cache_memory_latency.md` for the
   before/after numbers and the remaining ~3x gap's cause.

## Current status snapshot

As of commit `6294db8` (see `git log` for the latest): working tree clean, all commits pushed to
`origin/master`, 49/49 unit tests pass, the LaTeX PDF compiles with no errors. No cloud resources
(Lightning studios/jobs, Kaggle kernels) are left running. An independent "naive fresh reviewer"
pass (no prior context, spawned as a sub-agent) re-derived 15+ numbers from raw JSON and found one
real P0 tally bug and one real P1 CI-rounding mismatch in the manuscript; both are fixed. See
`notes/submission_readiness_roadmap.md` for exactly what is done, what remains, and the
recommended next action -- read that file first if resuming work in a new session.

## Repository layout

- `manuscript/draft.md` -- the paper (theory, method, results, limitations, references).
  `manuscript/convert_to_latex.py` regenerates `manuscript/latex/main.tex` from it (see docstring
  and `manuscript/latex/README.md` for the converter's scope/limitations); do not hand-edit
  `main.tex`.
- `code/elastic_mla/` -- the MLA / packed-cache / contextual-router implementation.
  - `mla.py` (dense + packed cached attention, now vectorized), `model.py` (`MLAGPT`),
    `elastic_cache.py` (packed prefix storage, now vectorized), `router.py`
    (`TieredRankRouter`, `ContextualElasticMLAGPT` -- the one actually used for every reported
    result; `ElasticMLAGPT`/`GlobalElasticMLAGPT` are earlier, non-pipeline variants, marked as
    such in their docstrings).
- `experiments/` -- the current, citable analysis/training/evaluation pipeline (see below) plus
  one results subdirectory per scale (`exp0_rank_variance/`, `exp1_rank_variance_122m/`,
  `exp2_rank_variance_250m/`, `contextual_router_{30m,122m,250m}/`).
- `notes/` -- a dated research log; see `notes/README.md` for an index of which notes are current
  vs. superseded. This is a lab notebook, not a curated "results" folder --
  `manuscript/draft.md` is the curated, authoritative summary.
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
   deployed hard-tier router under a rank-penalized Lagrangian; sweeps `--rank-lambda`. Two
   independent policies are frozen per scale where tier granularity is varied (see 250M coarse vs.
   fine in the results).
7. `experiments/evaluate_joint_rollout_sweep.py` -- selects one policy per scale/tier-grid using
   only the original 16 training / 4 validation sequences (never the frozen fresh-confirmation
   windows).
8. `experiments/confirm_fresh_contextual_router.py` -- the one-shot, pre-registered confirmation
   on 24 new nonoverlapping windows, run only after `experiments/fresh_confirmation_manifest.json`
   is committed with that configuration's frozen policy/oracle hashes. Four configurations are
   frozen and confirmed: `30m`, `122m`, `250m` (coarse tiers), `250m_fine`.
9. `experiments/audit_fresh_confirmation.py` and `experiments/audit_joint_training_replay.py` --
   independent recomputation of every reported statistic and a from-scratch bit-exact retrain
   check, respectively. Both must report `"status": "passed"` for a result to be cited.
10. `experiments/evaluate_causal_heuristic_routers.py` -- position/lexical/rarity/type causal
    baselines fit only on the training/validation split, evaluated once per configuration on the
    same frozen fresh windows at the router's own byte budget.
11. `experiments/benchmark_cache_memory_latency.py` -- measured T4 GPU peak memory and decode
    latency for full/packed-uniform/packed-router configurations, run once before and once after
    vectorizing `code/elastic_mla/{elastic_cache,mla}.py` (see
    `notes/measured_cache_memory_latency.md` for both sets of numbers).

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
`notes/compute_fallback_policy.md` for the compute-provider fallback policy used throughout. No
cloud resources should ever be left running between sessions -- always check
`lightning studio list` / `lightning job list` / `kaggle kernels status <name>` and stop/verify
completion before ending a session.
