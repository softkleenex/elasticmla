# Archive: superseded / withdrawn exploratory work

Everything here predates the corrected, currently-cited pipeline and is kept only for provenance
and history. **Nothing in this directory is cited as evidence in `manuscript/draft.md`.** If a
script or result here conflicts with a current note under `notes/` or a current script under
`experiments/`, the current one is authoritative.

## legacy_analysis_v1_v2/

Early (v1), corrected-but-superseded (v2), and layer-wise variants of the token-wise rank-variance
intervention, plus their plotting scripts and the two figures they produced
(`exp0_rank_variance.png`, `exp0_uniform_vs_layerwise.png`). All three analysis versions had
methodological issues fixed only in `experiments/analyze_rank_variance_v4.py` (see
`notes/exp0_v3_corrected_methodology.md` for the specific off-by-one horizon bug found in the
intermediate v3 iteration, which was never checked in as a standalone script). The current,
citable analysis is v4, plus the upper-tail-spectrum extension in
`experiments/analyze_risk_capacity_spectrum.py`.

## legacy_lexical_router_poc/

The original lexical/global token-identity router proof-of-concept (`train_tier_router.py`,
`eval_tier_router.py`, `exp_router/`), and the early layer0-full contextual router evaluator
(`eval_contextual_router.py`) used only for the exploratory `contextual_router_{30m,122m}_poc.md`
notes. All quantitative results from this era were superseded (and, for the lexical/global router,
explicitly withdrawn after the v3 label bug was found) by the pre-registered joint-rollout pipeline:
`generate_contextual_oracle_v1.py` -> `train_contextual_router.py` -> `train_joint_rollout_router.py`
-> `evaluate_joint_rollout_sweep.py` -> `confirm_fresh_contextual_router.py` ->
`audit_fresh_confirmation.py` -> `evaluate_causal_heuristic_routers.py`.

## legacy_benchmarks/

Early cached-decode and packed-cache benchmark prototypes, superseded by
`experiments/benchmark_cache_memory_latency.py`, which is the script actually used for the
measured GPU memory/latency numbers in the manuscript (Section 5.5).

## legacy_scratch/

One-off checkpoint/sanity verification scripts used during development. Not part of any
reproducible pipeline.

## legacy_kaggle_kernels/

An early P100 diagnostic kernel (`kaggle_diag/`) and an early exploratory analysis notebook
(`kaggle_exp1_v3_analysis/`) tied to the withdrawn Exp0-v3 methodology. The current Kaggle
training kernels are `code/kaggle_notebook/` (122M) and `code/kaggle_notebook_exp2_250m/` (250M).
