# 250M contextual router: full pipeline results (oracle, training, confirmation, heuristics)

## Status

Full pipeline replicated at 250M, mirroring 30M/122M exactly: Exp0-v4 rank-variance analysis ->
contextual oracle (layer-0-full, shared downstream tier) -> supervised router_max.pt -> joint-
rollout lambda sweep -> validation-selected policy -> pre-registered fresh-window confirmation ->
independent audit -> causal-heuristic baseline comparison -> deterministic replay audit.

Computed via Kaggle (250M base-model training) + Lightning AI (`lightning studio ssh` into a
started T4 Studio for the GPU-heavy Exp0-v4 + oracle generation, and `lightning job run` for the
fresh-window confirmation), with all provenance hashes independently verified locally.

## Setup

- Rank grid: 16,32,...,512 (18 points). Deployment tiers: {32, 128, 320, 512} (same proportional
  spacing as the 30M/122M coarse-tier scheme: 6.25%/25%/62.5%/100% of d_c).
- `router_max.pt`: supervised on isolated-position oracle labels, val macro-F1 0.412, test accuracy
  52.3%, predicted mean rank 414.5 vs target 399.5.
- Joint-rollout sweep (lambda in {0.05,0.1,0.2,0.4,0.8}, 15 epochs, router_max-weights
  initialization -- **not** random init, unlike the 122M fine-grid policy): lambda=0.4 selected
  as the only validation-feasible candidate (minimum bytes under +0.15-nat budget, beats exact-byte
  static on validation).

## Pre-registered fresh-window confirmation: FAILED

Following the exact frozen protocol (commit `a4bcc7f`, manifest extended with a `250m` entry before
viewing any fresh-window result), 24 new nonoverlapping windows were evaluated once:

- `mean_router_minus_exact_static`: **+0.0019** nat (router is *worse* on average)
- 95% paired sequence-bootstrap CI: **[-0.0022, +0.0061]** (includes zero)
- Sequence wins: 13/24 (barely better than chance)
- Exact one-sided sign-flip p: 0.806 (not significant)
- **`success: false`** under the pre-specified criterion (upper 95% CI bound < 0)

This is the first scale at which the router does **not** meet its own pre-registered bar against a
random, position-independent, exact-byte-matched control. It still beats the matched-tier-histogram
shuffle control (mean -0.0176 nat, CI [-0.0231,-0.0123], 21/24 wins, p<1e-5), so within-sequence
placement of the router's chosen tier multiset still helps, but the overall magnitude/rate
allocation is statistically indistinguishable from random at 250M.

Audit (`experiments/contextual_router_250m/fresh_confirmation_audit.json`) independently
reconstructed the seeded sampler, verified nonoverlap (min fresh/prior distance 2797, min
fresh/fresh distance 968), authenticated the checkpoint config (d_model=1024, 16 layers, d_c=512),
and recomputed every reported statistic; `status: passed`.

## Causal heuristic baselines: router loses to 3 of 4

Following the same procedure as 30M/122M (scores fit on the 16 training sequences, additive bias
selected on the 4 validation sequences under the same budget, evaluated once on the fresh windows
at the router's own byte budget):

| Control | Router - control (nat) | 95% CI | Result |
|---|---:|---|---|
| position | +0.00549 | [0.00108, 0.00969] | **router loses** |
| lexical identity | -0.01547 | [-0.02056, -0.01057] | router wins |
| token rarity | +0.00549 | [0.00103, 0.00978] | **router loses** |
| token type | +0.00549 | [0.00102, 0.00973] | **router loses** |

(Position, rarity, and type collapse to numerically identical results because, with only 4 coarse
tiers, their validation-selected bias pushes nearly the entire sequence into tier 320, i.e. they
degenerate to a near-uniform allocation at this coarse tier count.) Three of four causal heuristics
now beat the router with confidence intervals excluding zero, consistent with the failed
pre-registered confirmation above.

## Deterministic replay: passed

`experiments/contextual_router_250m/joint_training_replay_audit.json`: a from-scratch retrain
reproduces the frozen router tensors, training history, best score, splits, hyperparameters, and
initializer state bit-for-bit. Full rigor is preserved even though the headline result is negative.

## Interpretation: an honest scale-dependent trend

| Scale | Fresh confirmation vs exact-byte static | Causal heuristics beaten |
|---|---|---:|
| 30M | success, -0.0196 nat (CI excludes 0) | 2/4 |
| 122M | success, -0.0325 nat (CI excludes 0) | 0/4 (loses to 1, ties 3) |
| 250M | **failure**, +0.0019 nat (CI includes 0) | 1/4 |

Contextual placement clearly helps at 30M, is more marginal at 122M once compared against causal
heuristics, and provides **no measurable benefit** over even a random matched-budget allocation at
250M. This is a coherent, honest trend across three scales rather than an isolated failure, and it
is the paper's most important finding for calibrating claims about the current method's viability:
the isolated-position oracle labels and the straight-through joint-rollout surrogate this router
uses do not currently scale to larger, more expressive base models. This does not affect the
separate risk-capacity spectrum result (Section 5.1), which does not depend on the router at all.

## Provenance

- `experiments/exp2_rank_variance_250m/results_v4/{exp0_v4_summary,exp0_v4_records}.json`
- `experiments/contextual_router_250m/contextual_oracle_v1/{contextual_oracle_v1_summary,contextual_oracle_v1_records}.json`
- `experiments/contextual_router_250m/{router_max.pt, joint_lambda_0p4.pt, joint_rollout_selection.json}`
- `experiments/contextual_router_250m/{fresh_confirmation.json, fresh_confirmation_audit.json}`
- `experiments/contextual_router_250m/causal_heuristic_controls.json`
- `experiments/contextual_router_250m/joint_training_replay_audit.json`
- `experiments/fresh_confirmation_manifest.json` (`250m` entry)
