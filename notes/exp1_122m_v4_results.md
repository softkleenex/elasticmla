# Exp1 122M Exp0-v4 completion

## Execution and integrity

- Lightning job: `elasticmla-exp1-v4-122m-0817`
- Status: Completed
- Machine/device: T4 / CUDA
- Total reported job cost: 0.2022611
- Checkpoint step: 8000 (122.14M model)
- Method: v4 immediate-next-token horizon with nonoverlapping calibration/evaluation windows
- Probes: 24 sequences × 32 positions = 768
- Exact future horizon: 32 losses
- Rank grid: `{16,32,48,64,96,128,160,192,224,256,288,320,352,384}`
- Epsilon: 0.10 nats

Downloaded results passed local structural validation: 768 records, 12 valid 384-channel
permutations, valid rank values/horizons, 32 records per sequence, summary means and histograms
exactly recomputed from records. Evaluation start separation was at least 1,270 tokens; reproduced
calibration minimum separation was 425 tokens (required 385), with a 24,083-token realized gap
between the last calibration and first evaluation window.

## Results

- Future-mean r*: **31.5625**, sequence-cluster bootstrap 95% CI **[28.8948, 34.2500]**
- Future-max r*: **280.0000**, sequence-cluster bootstrap 95% CI **[272.4161, 287.4797]**
- Mean normalized by d_c: 8.22%
- Max normalized by d_c: 72.92%
- Nonmonotonic curve frequency: mean 89.32%, max 69.66%

This supports a strong separation between average and tail-risk capacity requirements on this
checkpoint. It does not yet establish scale replication against the 30M model because the old
30M v3 result was invalidated; the 30M checkpoint must be rerun with v4.

## Artifact hashes

- `analysis.log`: `498fd11c9efae7ba521b00b0bd1bb6a235b0798856b85bb604c6951bfc2699d4`
- `exp0_v4_records.json`: `0ed88e820a06e76bba62fa7fcb9eaa09bdcff874a1134ebaa25db5564705231d`
- `exp0_v4_summary.json`: `f49c9f966f3910fd682e1937f9d694f978428f44483fa77196372149065e5cb1`

Scope remains a one-position full-attention truncation simulation, not joint packed-cache rollout,
peak-memory validation, or latency evidence.
