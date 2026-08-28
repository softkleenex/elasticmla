# Measured GPU cache memory and decode latency (T4, real CUDA)

## Setup

`experiments/benchmark_cache_memory_latency.py` runs a real incremental prefill+decode loop on a
Lightning T4 GPU with the authenticated 30M and 122M checkpoints and their frozen router policies,
using real held-out data tokens (not synthetic random tokens). Three configurations are compared,
each starting from the same prefill: full-width dense MLA (`forward_cached`), a uniform fixed-rank
packed cache matched to the router's realized average rank (`forward_cached_packed` with constant
rank), and the frozen contextual router's packed cache. `torch.cuda.max_memory_allocated` is reset
before each run and read after prefill+128 decode steps; `resident_cache_bytes` is the exact
tensor-payload size of the cache object at the end, using the same `cache_num_bytes` /
`packed_cache_num_bytes` helpers used throughout the repository.

## Results (Tesla T4, batch=8)

| Scale | Metric | Full MLA | Packed (uniform matched-rank) | Packed (router) |
|---|---|---:|---:|---:|
| 30M | Cache bytes | 17,694,720 | 11,891,736 (67.2%) | 11,926,296 (67.4%) |
| 30M | Peak allocated | 598,753,792 | 591,000,064 (98.7%) | 577,955,328 (96.5%) |
| 30M | Mean decode step | 8.41 ms | 1450.9 ms (**172.5x**) | 1412.6 ms (**167.9x**) |
| 122M | Cache bytes | 61,341,696 | 33,260,592 (54.2%) | 36,541,424 (59.6%) |
| 122M | Peak allocated | 1,537,902,080 | 1,507,376,128 (98.0%) | 1,472,963,072 (95.8%) |
| 122M | Mean decode step | 17.87 ms | 3592.7 ms (**201.0x**) | 3592.9 ms (**201.1x**) |

## Interpretation

1. **Persistent cache byte reduction is real and measured, not just formula-derived.** Measured
   ratios (67.2-67.4% at 30M, 54.2-59.6% at 122M) are consistent with the byte-formula predictions
   used throughout the paper (68.80%/61.46% for the router at 30M/122M in the earlier confirmation
   experiments; small differences reflect different sampled sequences and decode-step content).
2. **Peak GPU memory is modestly, not dramatically, lower for packed** (1.3-4.2% reduction across
   scales/configs) rather than unchanged as our prior (unmeasured) limitation language assumed.
   This is a genuine small positive finding, not previously claimed.
3. **Decode latency is dramatically worse for packed: 168-201x slower per step.** This is because
   `pack_latents`/`unpack_latents` (`code/elastic_mla/elastic_cache.py`) reconstruct the entire
   cached history with a per-token Python loop on every single decode step (`O(T)` Python-level
   work per step, growing with sequence length), whereas the dense full-MLA path only recomputes
   K/V from a single cached tensor without a Python loop. This is a decisive, previously
   unquantified systems limitation: **the current implementation cannot be used for real-time
   decoding**, regardless of its true persistent-memory savings.

## What this changes for the paper

- Replace vague "we do not claim latency or peak-memory improvements" language with the concrete
  measured numbers above: a modest measured peak-memory win, and a severe measured latency loss
  with a diagnosed root cause (Python-loop pack/unpack).
- This sharpens future work: a vectorized (no Python loop) or fused packed-attention kernel is
  necessary before any serving-latency claim is possible, and is now a quantified target (need at
  least ~200x speedup on the packed path to match dense MLA at 122M).

## Provenance

- `experiments/benchmark_cache_memory_latency.py`
- `experiments/contextual_router_30m/measured_cache_memory_latency.json`
- `experiments/contextual_router_122m/measured_cache_memory_latency.json`
- Lightning jobs: `elasticmla-bench-30m-0826`, `elasticmla-bench-122m-0826` (T4, both Completed)
- Checkpoint/data/policy SHA-256 verified against the same values used in the fresh-window
  confirmation results before running.
