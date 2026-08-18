# Contextual ElasticMLA 30M PoC v1

## Aligned method

Layer 0 is always full rank. Its contextual output is normalized at the layer-1
pre-attention boundary and fed to one router. The selected token tier is shared by
layers 1–5. Oracle labels use the same intervention scope: at one source position,
layer 0 remains full and layers 1+ share the tested rank over the immediate next-32
loss window.

Deployment tiers are `{16,64,160,256}`. Raw oracle ranks are retained and upward-
quantized tier labels are stored separately. Checkpoint/data hashes, seeded evaluation
starts/positions, token IDs, channel orders, and record counts are validated before use.

## Oracle

- 768 positions, 24 nonoverlapping sequences
- Mean raw r*: 17.4167; mean tier 18.125
- Max raw r*: 141.4583; mean tier 165.875
- Mean tier histogram: 16=748, 64=13, 160=7, 256=0
- Max tier histogram: 16=164, 64=50, 160=211, 256=343

The oracle is an isolated-position intervention and is not a joint rollout quality guarantee.

## Held-out simultaneous packed evaluation

Four nonoverlapping held-out sequences: 9, 12, 19, 21.

| Policy | downstream mean rank | fixed-MLA bytes | delta loss |
|---|---:|---:|---:|
| fixed 16 | 16 | 31.25% | +1.0140 |
| fixed 64 | 64 | 45.14% | +0.5591 |
| fixed 160 | 160 | 72.92% | +0.1195 |
| contextual max router | 171.91 | 76.36% | +0.1006 |
| full 256 | 256 | 100.70% (packed metadata) | 0 |

The max router's exact per-sequence tier histogram was shuffled across token positions
20 times. At identical packed bytes, the contextual ordering improved loss by 0.0529 nats.
All four sequence-level paired differences were negative; the exploratory four-cluster
bootstrap interval was [-0.0738, -0.0338]. With only four clusters this is PoC evidence,
not a population-level significance claim.

The mean-label router largely collapsed to rank 16 and failed under simultaneous compression
(delta loss +1.0138); high isolated-label accuracy therefore does not imply rollout quality.
This negative result motivates rollout-aware training/objectives.

## Defensible conclusion

On this small held-out PoC, contextual token placement improves quality over a noncontextual
matched-budget allocation for the conservative max-risk tier distribution. It does not yet
establish generalization, joint-oracle optimality, 122M replication, peak-memory reduction,
or latency gains.

## Execution

- Oracle Lightning job: `elasticmla-context-oracle-30m-0818`, T4, Completed
- Reported cost: 0.11297222
- Five-minute scheduler check: Running
- All 35 unit tests passed before oracle generation
