# Exp0-v4 30M ↔ 122M scale comparison

## Validated inputs

Both runs use the corrected immediate-next-token horizon, nonoverlapping calibration/evaluation
windows, 24 sequences × 32 probes, horizon 32, epsilon 0.10 nats, and per-layer
gradient×activation channel orders.

| Model | d_c | mean r* | mean/d_c | max r* | max/d_c |
|---|---:|---:|---:|---:|---:|
| 30M (30.6M) | 256 | 23.0417 | 9.00% | 189.6250 | 74.07% |
| 122M (122.14M) | 384 | 31.5625 | 8.22% | 280.0000 | 72.92% |

Sequence-cluster bootstrap comparison of normalized rank, 122M minus 30M:

- future mean: -0.781 percentage points, 95% CI [-1.617, +0.076]
- future max: -1.156 percentage points, 95% CI [-3.849, +1.611]

Both intervals include zero. The defensible interpretation is that the large separation between
average capacity (~8–9%) and tail-risk capacity (~73–74%) replicates descriptively at both scales,
and this experiment finds no clear normalized-rank shift. This is **not** an equivalence test and
does not prove scale invariance beyond these two checkpoints/datasets.

The absolute r* rises with d_c, so router tiers should be parameterized by rank fraction or calibrated
per model rather than copying the same absolute tier set across scales.

## 30M Lightning execution

- Job: `elasticmla-exp0-v4-30m-0817`
- Status: Completed; T4/CUDA; reported cost 0.08738333
- Five-minute scheduler check: Running at 305 seconds, cost 0.038383335
- Checkpoint SHA-256: `18646f1fdd2d97396e687aaf5e60900b0f825ce96a0d11ffe9b6ac600acc8dc1`
- Validation SHA-256: `5e146271a0bbaf5581b9e71490b1c5b12910b6a59062307ab81206acec8615be`
- Result hashes:
  - analysis log: `bdd6ead0df68b1cc70fae4f3357bb9ce177930cff321d6b6b62dfdfed6310887`
  - records: `acca0aea5be67f7c2f0f90d2905504c787d7adf5306ea17ceb1f74b81a750b38`
  - summary: `0c9b32f049ca92e6b8110d3a9812353ca7caf06b4427517fb48fa9b7c0e6ddc4`

Results passed local checks: 768 records, six valid 256-channel permutations, summary means and
histograms recomputed exactly, and valid rank/horizon values. The temporary retrieval Studio was stopped.
