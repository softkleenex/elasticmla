# Causal heuristic baselines beat or match the learned contextual router

## Setup

Position, lexical-identity, token-rarity, and token-type scores are estimated **only** from the
16 original router-training sequences (never fresh windows). An additive rate bias is selected on
the 4 original validation sequences under the +0.15-nat budget (minimum-bytes-under-budget rule,
matching the router selection protocol). The chosen rule is then evaluated once on the same 24
frozen fresh windows used for router confirmation, at **exactly the router's own per-sequence
byte budget** (same total downstream rank sum, upward-quantized to the tier grid).

This is a strictly causal control: score at position t depends only on the absolute position t
and/or the current token x[t], never on future tokens or the sequence-level router output.

## Results (fresh windows, router loss minus causal-control loss; negative = router better)

| Scale | Control | Mean (router − control) | 95% CI | Interpretation |
|---|---|---:|---|---|
| 30M | position | +0.01375 | [0.00245, 0.02537] | **router worse** |
| 30M | lexical_identity | -0.02533 | [-0.03990, -0.01135] | router better |
| 30M | token_rarity | -0.02675 | [-0.04124, -0.01242] | router better |
| 30M | token_type | +0.00653 | [-0.00417, 0.01736] | tie (CI includes 0) |
| 122M | position | +0.00404 | [-0.00977, 0.01706] | tie (CI includes 0) |
| 122M | lexical_identity | +0.00674 | [-0.00439, 0.01774] | tie (CI includes 0) |
| 122M | token_rarity | +0.04678 | [0.03401, 0.05925] | **router clearly worse** |
| 122M | token_type | +0.01174 | [-0.00280, 0.02581] | tie (CI includes 0) |

## Interpretation

The learned contextual router beats the earlier random-placement and matched-tier-shuffle
controls at both scales (see `notes/fresh_confirmation_results.md`), but it does **not** reliably
beat simple, cheap, strictly causal heuristics evaluated at the identical per-sequence byte
budget:

- At 30M, a trivial absolute-position rule (rate increases with position in-sequence, biased/
  quantized to the tier grid on validation data only) **beats the router** with a CI that excludes
  zero.
- At 122M, a trivial inverse-token-frequency rule **clearly beats the router** (CI [0.034, 0.059]),
  and the position and lexical-identity rules are statistically tied with the router.
- Only the lexical-identity and token-rarity rules at 30M are clearly beaten by the router.

This substantially narrows the paper's central claim. The correct, defensible statement is:

**Contextual placement is better than random or shuffled placement at the same byte budget, but
current evidence does not show it is better than the strongest simple causal heuristic at that
budget; in several cases a trivial heuristic wins.** The router's practical value therefore lies
in producing a reasonable *and automatic* allocation without hand-designed features, not (yet) in
demonstrated superiority over hand-designed causal heuristics.

## What this changes for the paper

1. The abstract/conclusion may no longer claim "contextual routing beats noncontextual
   allocation" without qualifying "than random/shuffled placement, but not always than simple
   causal heuristics."
2. This is a genuinely useful negative result: it argues that isolated-position future-loss
   sensitivity is a fundamentally weak training signal relative to what a designed heuristic can
   achieve at the same budget, motivating joint-rollout objectives with stronger regularization,
   larger/better calibration data, or hybrid heuristic-plus-learned architectures.
3. Any resubmission must report these heuristic baselines alongside random/shuffle controls,
   consistent with the reviewer's explicit prior request for "position-only and lexical-identity"
   baselines.

## Provenance

- `experiments/evaluate_causal_heuristic_routers.py`
- `experiments/contextual_router_30m/causal_heuristic_controls.json`
- `experiments/contextual_router_122m/causal_heuristic_controls.json`
- Selection uses only the original 16 train / 4 validation sequences; fresh windows are the same
  24 pre-registered, audited windows used for the router confirmation. All checkpoint/data/policy/
  oracle/audit hashes are verified before scoring (see script provenance checks).

## Update: this note covers only 30M/122M

This note predates the 250M causal-heuristic runs and reports the correct 30M/122M-only tally at
the time it was written (2 wins: 30M-coarse/lexical, 30M-coarse/rarity). It is now superseded as a
*complete* tally by `notes/contextual_router_250m_results.md` (250M-coarse: router also beats
position/rarity/type but *loses* to lexical identity there) and
`notes/contextual_router_250m_tier_granularity_diagnostic.md` (250M-fine: router loses all four,
including lexical identity). The correct all-scale tally, used in `manuscript/draft.md`, is
**3 of 16** scale-heuristic comparisons won by the router (30M-coarse/lexical, 30M-coarse/rarity,
250M-coarse/lexical), not 2. An earlier version of the manuscript itself under-counted this as
"2 of 16, both at 30M" before an independent fresh review caught the discrepancy; see commit
`c9de709`.
