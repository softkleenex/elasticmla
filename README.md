# Mean-Safe Is Not Tail-Safe: Token-Wise Latent Capacity in Multi-Head Latent Attention

Working repository for a measurement paper on token-wise latent capacity in Multi-Head Latent Attention
(MLA). The paper asks how much of a token's cached latent is needed for its future losses to stay
within budget, on average versus at every future position. It also includes a pre-registered,
byte-exact case study of a learned contextual router over a variable-width packed cache, at three
model scales (30M, 122M, 250M parameters) plus the public DeepSeek-V2-Lite.

**Start here:** `manuscript/draft.md` is the paper (source of truth). `manuscript/latex/main.pdf` is
the compiled version and `manuscript/latex_tmlr/main.pdf` the anonymized TMLR version. Regenerate
them with `uv run python manuscript/convert_to_latex.py [--style tmlr]` and then `tectonic main.tex`.

Headline claims (epsilon = 0.1 nat, horizon H = 32, saliency-ordered coordinate basis), each with
its main caveat:

1. **The gap.**
   - Keeping each of a token's next 32 losses within budget needs 72-79% of the latent.
   - Keeping their mean within budget needs almost nothing: on a grid extended to rate 0, 82% (30M)
     and 78% (250M) of tokens are mean-safe with all latent content removed (pre-registered R10b).
   - The ratio is at least 6.5 at every budget from 0.05 to 0.5 nat; about 4.5 under KL at 0.05 nat;
     6.2 in GPT-2's per-head KV cache. At the mean-safe rate, about 75% of tokens flip a greedy
     prediction.
   - The ratio depends on H: 2.3 at H = 4, 23 at H = 128. The worst-offset rate is stable (0.55-0.74).
2. **What it reflects.**
   - Per-offset loss changes are dispersed around a near-zero mean, heavy-tailed, and largest just
     after the token.
   - Gaussian draws matched to each row's mean and spread give 0.62-0.66 against the observed 0.72.
     A sign-randomized null with one draw shared across ranks matches within 0.03 at every tail level.
   - Equal-norm noise shaped like the latent (exploratory) costs as much on average, but not per token.
     Isotropic noise of equal norm costs 0.18-0.28 less.
3. **The consequence.** Applied jointly at 250M:
   - Mean-safe rates pass a mean check (+0.10 nat) while 28.6% of positions exceed the budget. That is
     as many as uniform truncation at equal bytes (29.3%), so the fraction reflects total rate, not
     the criterion.
   - Worst-offset-safe rates leave 4.4%, against 9.7% for uniform truncation at their bytes.
4. **Basis caveat.** Rotating to principal directions removes about 50 points of the worst-offset
   requirement in our small models but only 11 in DeepSeek-V2-Lite.
5. **Router case study.**
   - The router beats exact-byte random placement in 3 of 4 pre-registered configurations, but does
     no better than a per-token lookup table.
   - int8 latent quantization (0.50 of the bytes, about zero loss) dominates it on bytes.
   - The packed cache gives no peak-memory or throughput gain and decodes 2.4x slower.

## Current status snapshot

**2026-11-20.**
- The paper was desk-rejected by TMLR (#13801, 2026-11-08) for clarity of writing. It has been
  rewritten as a measurement paper for resubmission in January 2027.
- Pre-registered analyses R1-R10 and an internal four-reviewer critique are in
  `notes/2026-11-14_tmlr_resubmission_plan.md`. A verified related-work sweep is in
  `notes/2026-11-16_related_work_sweep.md`.
- The DeepSeek-V2-Lite nulls and joint test (R8) failed their reproduction check and are reported as
  exploratory.
- The earlier October 2026 code-review corrections are in `notes/code_review_2026-10_corrections.md`.

Nothing has been submitted anywhere except the desk-rejected TMLR submission. No cloud resources
are left running.

## Repository layout

- `manuscript/draft.md` -- the paper (theory, method, results, limitations, references).
  `manuscript/convert_to_latex.py` regenerates `manuscript/latex/main.tex` from it (see docstring
  and `manuscript/latex/README.md` for the converter's scope/limitations); do not hand-edit
  `main.tex`.
- `code/elastic_mla/` -- the MLA / packed-cache / contextual-router implementation.
  - `mla.py` (dense + packed cached attention, vectorized), `model.py` (`MLAGPT`),
    `elastic_cache.py` (packed prefix storage; sequence-major flat layout, appended in place on
    each decode step), `router.py`
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
   `code/kaggle_notebook/` and `code/kaggle_notebook_exp2_250m/`). Reproducing the checkpoints:
   the 122M checkpoint (step 8000, no stored `train_config`) came from
   `code/kaggle_notebook/train_kaggle.py` as of commit `aa5caee`, where `MAX_STEPS = 8000` was
   hard-coded; the current file defaults to 1500, so set `MAX_STEPS=8000`. The 250M checkpoint
   stores its `train_config` (3000 steps x batch 4 x grad-accum 12 x 384 tokens = 55.3M tokens).
   `train_exp0.py` seeds torch but not numpy's batch sampler, so the 30M run is pinned by its
   checkpoint hash, not bit-reproducible.
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

## Resubmission analyses (R1-R10, `notes/2026-11-14_tmlr_resubmission_plan.md`)

Pre-registered unless marked post hoc. Each script hashes its inputs and checks that it reproduces
the published per-probe records before reporting anything.

- `analyze_perturbation_null.py` (R2): equal-norm isotropic and covariance-shaped nulls.
  Results: `experiments/perturbation_null/`.
- `analyze_order_statistic_null.py` (R3): Gaussian and sign-randomized order-statistic nulls.
  Result: `order_statistic_null.json`.
- `analyze_latent_quantization.py` (R4/R5): int8/int4 latent quantization and the joint safe-rate
  allocation. Results: `experiments/latent_quantization/`. The byte fractions stored there use 2/d_c
  for the fp16 scale; the correct value is 1/d_c (fixed in the script, 2026-11-20).
- `analyze_metric_and_bitwidth.py` (R6/R7): KL and greedy-flip spectra; bit width as the rate axis.
  Results: `experiments/metric_bitwidth/`.
- `analyze_public_mla_null.py` (R8, exploratory): the R2/R5 analyses on DeepSeek-V2-Lite.
- `analyze_gpt2_kv_spectrum.py` (R9): GPT-2 per-head KV spectrum.
- `analyze_low_rank_and_matched_joint.py` (R10b): rate grid extended to 0, rate-matched joint
  references, quantization flips. Results: `experiments/low_rank_joint/`.
- `analyze_review_descriptives.py` (R10a, post hoc): grid-floor shares, horizon dependence,
  location of the worst offset, shared-draw nulls. Result: `review_descriptives.json`.

The 250M runs go through Kaggle via `code/kaggle_review_reruns/push.py`.

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
