# Contextual ElasticMLA 122M replication

## Oracle and setup

- Layer 0 full rank; shared contextual tier for layers 1–11
- Proportional tiers `{24,96,240,384}`
- 768 positions over 24 nonoverlapping sequences
- Mean raw/tier rank: 23.8333 / 35.3438
- Max raw/tier rank: 247.8542 / 297.1875
- Lightning job `elasticmla-context-oracle-122m-0818`, Completed, cost 0.19899444
- Checkpoint/data/source provenance and token alignment validated locally

## Held-out joint packed evaluation

Four test sequences `{9,12,19,21}`:

| Policy | Mean downstream rank | Fixed-MLA bytes | Delta loss |
|---|---:|---:|---:|
| fixed 240 | 240 | 68.75% | +0.0920 |
| contextual max | 297.52 | 81.42% | +0.0584 |
| full 384 | 384 | 100.48% | 0 |

The contextual max policy beats its exact matched-budget shuffled placement by 0.0877 nats;
all four sequence differences are negative. This replicates the contextual placement signal
observed at 30M.

However, uniform downstream rank 298 at nearly identical bytes achieves loss 1.9168 versus
1.9566 for the router. The current router therefore does not beat the continuous same-budget
fixed-rank baseline. The coarse isolated-intervention tier distribution is not yet rollout-Pareto
optimal.

The mean policy also has unacceptable quality loss (+1.5295 nats), despite beating its random
and static rank-30 controls. Results remain a four-sequence PoC rather than population evidence.

## Cross-scale conclusion

At both 30M and 122M, contextual ordering improves over random placement at an identical tier
histogram, demonstrating a reproducible content-aware signal. At both scales the max router
fails to dominate the same-byte uniform rank. Next work must calibrate router risk bias/floors on
validation joint-rollout loss and evaluate once on the untouched test split.

## Validation-only rollout calibration

A 32-policy risk-bias/floor grid found zero validation candidates that both stayed within +0.15 loss and beat the same-rank static baseline. The unmodified policy remained selected and was worse than static rank 298 on test by 0.0398 nats. Post-hoc calibration is insufficient; direct joint-rollout optimization is required.
