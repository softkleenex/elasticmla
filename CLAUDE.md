# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research repository (not a shipping library) for **ElasticMLA**: token-wise variable-width
latent caching for Multi-Head Latent Attention, a rate-allocation theory for the resulting
risk-capacity spectrum, and a **pre-registered** evaluation of a learned contextual router at
three model scales (30M/122M/250M) and four scale/tier-grid configurations
(`30m`, `122m`, `250m`, `250m_fine`).

The deliverable is `manuscript/draft.md`. Code exists to produce numbers that go into it.
`README.md` has the full claim inventory and pipeline description; `notes/submission_readiness_roadmap.md`
has current status and the recommended next action — **read that first when resuming work.**

## Commands

```bash
uv sync                                             # environment (Python 3.11)
uv run python -m unittest discover -s tests         # full suite (108 tests, ~10s, CPU only;
                                                    # test_token_type_uses_only_current_id needs
                                                    # network on first run for the tiktoken vocab)
uv run python -m unittest tests.test_tiered_router  # one module
uv run python -m unittest tests.test_tiered_router.TieredRouterTest.test_name   # one test

uv run python manuscript/convert_to_latex.py        # regenerate manuscript/latex/main.tex
cd manuscript/latex && tectonic main.tex            # compile the PDF
```

Experiment scripts are argparse CLIs run individually; every one takes explicit
`--checkpoint/--data/--device` style paths rather than reading config. Example:

```bash
uv run python experiments/confirm_fresh_contextual_router.py \
  --checkpoint <base.pt> --data <val.bin> \
  --oracle-summary ... --oracle-records ... --policy ... \
  --protocol-manifest experiments/fresh_confirmation_manifest.json \
  --contextual-summary ... --scale 250m_fine --output ... --device cuda
```

The unit tests run on CPU and are the only thing you can verify end-to-end locally without
checkpoints. Steps 2-4, 8 and 11 of the pipeline need a GPU.

## Architecture

### The library (`code/elastic_mla/`)

Not installed as a package. Every consumer does
`sys.path.insert(0, <repo>/code)` before `from elastic_mla import ...`. Keep that pattern.

- `mla.py` — `MultiHeadLatentAttention`: dense forward, `forward_cached` (dense KV cache), and
  `forward_cached_packed` (variable-width packed cache). Rank-mask construction here is
  vectorized; do not reintroduce per-token Python loops.
- `elastic_cache.py` — the packed representation: `values/offsets/ranks/channel_order/k_rope`.
  Each token stores only its own saliency-ordered latent prefix; dense `(B,S,d_c)` tensors exist
  only as temporary reconstruction buffers. `pack_latents`/`unpack_latents`/`append_packed_latents`
  are vectorized via `repeat_interleave`/`gather`.
- `model.py` — `MLAGPT`, the base transformer.
- `router.py` — three wrappers with very different status, which is the main trap in this repo:
  - **`ContextualElasticMLAGPT` is the one used for every reported result.** One router reads the
    layer-0-full hidden state and picks a single tier shared by all downstream layers.
  - `ElasticMLAGPT` (per-layer independent routers) — earlier variant, exercised only by tests.
  - `GlobalElasticMLAGPT` — dead code, no script or test imports it.
  Their docstrings say so; do not cite the wrong one as "the paper's architecture".

### The pipeline (`experiments/`)

Eleven scripts in strict dependency order, enumerated in `README.md`. The shape to internalize:

`train_exp0` → `analyze_rank_variance_v4` (the seeded-window source of truth) →
{`analyze_risk_capacity_spectrum` (Section 5.1, router-independent)
| `generate_contextual_oracle_v1` → `train_contextual_router` → `train_joint_rollout_router`
→ `evaluate_joint_rollout_sweep` → `confirm_fresh_contextual_router` →
`audit_*` + `evaluate_causal_heuristic_routers`} and `benchmark_cache_memory_latency` on the side.

A second group of scripts, outside the eleven-step list, tests the theory memo's falsifiable
predictions (`notes/theory_contextual_tail_rate.md`): `analyze_horizon_capacity` (prediction 3),
`analyze_pair_interaction` (5), `analyze_basis_dependence` (7), then `analyze_kv_basis` and
`analyze_public_kv_basis` (does the basis effect hit a plain per-head KV cache, and public GPT-2
checkpoints). Most of them load `analyze_risk_capacity_spectrum.py` as a module via `importlib`
(registered as `risk_v5`) and reuse its functions, so a change there changes them too. The
basis result is a **negative result for Section 5.1's headline rates and does not transfer to
public checkpoints** (`notes/low_rank_does_not_mean_compressible.md`); keep both halves together
whenever it is quoted.

Results live in one directory per configuration: `experiments/exp{0,1,2}_rank_variance*/` and
`experiments/contextual_router_{30m,122m,250m}/`. Within a router directory, `joint_lambda_*`
files are the **coarse** tier grid and `fine_lambda_*` / `*_fine.json` are the **fine** grid —
`250m` and `250m_fine` are two frozen configurations sharing one directory.

## Non-negotiable working rules

These exist because the paper's credibility rests on them.

1. **Provenance chaining.** Every script authenticates its inputs by SHA-256 against the artifacts
   that produced them and refuses to run on a mismatch. When you add or change a script, preserve
   this: hash checkpoint, data, policy, oracle summary/records, and store the hashes in the output
   JSON. A result whose corresponding `*_audit.json` / `*_replay_audit*.json` does not report
   `"status": "passed"` is not citable.
2. **Pre-registration.** `experiments/fresh_confirmation_manifest.json` freezes the policy/oracle
   hashes, seed (91827), window count (24) and control repeats for each configuration, and must be
   **committed before** the confirmation runs. Never retune a policy, tier grid, λ, or threshold
   after seeing fresh-window results. The 24 fresh windows are confirmatory and single-use; the
   older 4 `test` windows are exploratory and must be labeled as such.
3. **Report negative results as negative.** The router winning only 3 of 16 causal-heuristic
   comparisons, and the 250M-coarse confirmation failing its own criterion, are findings, not
   problems to engineer away. Don't reframe a tier-resolution confound as a scale trend.
4. **`manuscript/draft.md` is the source of truth.** Never hand-edit `manuscript/latex/main.tex` —
   it is generated. The converter is purpose-built for this document's markdown conventions
   (see `manuscript/latex/README.md`); if you add a new construct to the draft, extend the
   converter rather than working around it.
5. **`notes/` is a dated lab notebook, not curated results.** Later notes correct earlier ones and
   several are explicitly withdrawn — consult `notes/README.md` for what is current before quoting
   any number from a note. `archive/` is superseded work kept for provenance only; nothing there
   is cited.

## Conventions

- **Two code styles, both intentional.** `code/` and `tests/` are ordinary PEP 8 with type hints
  and docstrings. The original pipeline scripts in `experiments/` are deliberately compressed
  (one-space indent, semicolon-joined statements, packed lines); the newer prediction/basis
  `analyze_*` scripts are PEP 8 with long module docstrings. Match the file you are editing; do
  not reformat either kind.
- Device selection is explicit everywhere (`--device cuda|mps|cpu`), never inferred silently in
  confirmatory scripts.
- Seeds are fixed and recorded in output JSON; determinism is checked by
  `audit_joint_training_replay.py` (bit-exact retrain).

## Data, checkpoints, compute

`experiments/*/data/*.bin`, `experiments/*/ckpt/*.pt` and `kaggle_output/` are gitignored.
Regenerate the 30M data with `experiments/prepare_data.py` (TinyStories, GPT-2 BPE, 40k docs);
122M/250M checkpoints come from Kaggle (`code/kaggle_notebook*/`) and must be re-downloaded.

30M trains locally on Apple Silicon MPS; everything larger ran on Kaggle P100 or Lightning AI T4.
`notes/compute_fallback_policy.md` is the operating policy — notably: verify a launched cloud job
is genuinely running five minutes after launch (scheduler `Running` alone is not a health check),
and **never leave cloud resources running between sessions** (check `lightning studio list`,
`lightning job list`, `kaggle kernels status <name>` before finishing).

`master` has been ahead of `origin/master` with unpushed commits (a past network outage blocked
the push and a fresh `tectonic` rebuild). Check `git status -sb`; push only with the user's
approval, and rebuild the PDF before quoting a page count.
