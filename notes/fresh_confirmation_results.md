# Fresh-window contextual routing confirmation

## Status

The one-shot, pre-specified confirmation completed at both scales. The routing policies and
protocol were frozen in commit `a4bcc7f` before the new windows were evaluated. Each run used
24 newly sampled full-context sequences separated from every authenticated prior oracle/router
window and from one another by at least `block_size + 1` tokens.

The narrow primary conclusion is supported: **a frozen contextual router has lower next-token
cross-entropy than a content-independent static allocation using exactly the same per-sequence
persistent packed-cache bytes at both evaluated scales.**

## Confirmatory results

| Scale | Mean downstream rank | Packed bytes / fixed MLA | Δloss vs full | Router − exact-byte static | Paired bootstrap 95% CI | Wins | Exact one-sided sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30M | 145.78 / 256 | 68.80% | +0.1001 | **−0.01959** | [−0.02911, −0.00938] | 20/24 | 0.000655 |
| 122M | 206.93 / 384 | 61.46% | +0.1823 | **−0.03251** | [−0.04575, −0.01958] | 21/24 | 0.0000493 |

The pre-specified success rule was an upper 95% paired sequence-bootstrap bound below zero,
assessed separately at each scale. Both scales passed. The intervals use two-sided 95% bounds;
therefore the joint two-scale primary statement is also compatible with Bonferroni control based
on the two upper 97.5th-percentile bounds.

Supporting exact-histogram shuffle controls also favored contextual allocation:

| Scale | Router − matched-tier shuffle | Paired bootstrap 95% CI | Wins |
|---|---:|---:|---:|
| 30M | −0.07989 | [−0.08999, −0.06956] | 24/24 |
| 122M | −0.13907 | [−0.15439, −0.12408] | 24/24 |

For the static control, each sequence's routed rank sum was decomposed into integer ranks `q`
and `q+1`; content-independent locations were randomized 20 times. For the shuffle control, the
router's exact tier histogram was permuted 20 times. Every control repeat was asserted to have
exactly the router cache's byte count.

## Interpretation and limitations

- The 30M point stays within the validation-time +0.15 nat quality constraint on fresh data.
- The 122M point does **not**: its fresh Δloss is +0.1823 despite +0.1408 on the four validation
  sequences. Thus 122M confirms contextual allocation efficiency at its realized budget, but not
  reliable satisfaction of the +0.15 quality constraint. A quality-constrained deployment needs
  a more conservative policy or calibration on a larger validation set.
- Results establish persistent packed-cache storage reduction, not peak-memory or latency gains;
  the correctness-first implementation reconstructs dense temporary latents.
- Evidence covers two checkpoints and their validation distribution, not arbitrary model scales,
  architectures, datasets, or out-of-distribution prompts.
- The full model was frozen and only the router was trained. The 30M and 122M policies use
  different tier grids and initialization, so cross-scale numerical differences are descriptive.

## Provenance and audit

- 30M Lightning job: `elasticmla-confirm-30m-0819` — Completed, reported cost 0.14672777.
- 122M Lightning job: `elasticmla-confirm-122m-v2-0819` — Completed, reported cost 0.2254.
- A preceding 122M launch `elasticmla-confirm-122m-0819` failed before loading data because of a source filename
  mismatch. It exposed no sequence/result and changed no policy; reported cost 0.11433333.
- Five-minute checks observed `Running` plus accrued cost, but the Lightning SDK cannot stream
  logs for running jobs, so these were partial rather than full log-confirmed health passes.
- The original one-shot job commands predated the manifest-enforcement arguments now required
  by the runner. This is explicit: the original evidence is bound post hoc by the already-frozen
  protocol commit, immutable policy/result hashes, complete job logs, and the audit; future runs
  are rejected unless they match the machine-readable manifest.
- `audit_fresh_confirmation.py` authenticated the checkpoint configuration, data, policy, v4
  summary/records, and contextual-oracle hashes; exactly reconstructed the seeded start sampler;
  checked all prior/fresh spans; and independently recomputed row arithmetic, bootstrap intervals,
  exact sign-flip statistics, rank-derived packed bytes, and success decisions.
- Deterministic training replays reproduced both frozen policies bit-for-bit at the final router-
  tensor level, with exactly equal histories, best scores, splits, and training hyperparameters.
  At 30M, `router_max.pt` supplied the initial weights. At 122M, the router used a recorded seeded-
  random initial-state hash; `router_max.pt` supplied only authenticated splits/provenance, not
  weights. Replay audit files enforce this distinction and record the oracle hashes.
- Minimum fresh-to-prior start distances: 278 tokens at 30M (required 257), 2,797 at 122M
  (required 385). Minimum fresh-to-fresh distances: 416 and 968 respectively.

Artifacts:

- `experiments/fresh_confirmation_manifest.json`
- `experiments/fresh_confirmation_comparison.json`
- `experiments/contextual_router_30m/fresh_confirmation.json`
- `experiments/contextual_router_30m/fresh_confirmation_audit.json`
- `experiments/contextual_router_122m/fresh_confirmation.json`
- `experiments/contextual_router_122m/fresh_confirmation_audit.json`
- `experiments/contextual_router_{30m,122m}/joint_training_replay_audit.json`
