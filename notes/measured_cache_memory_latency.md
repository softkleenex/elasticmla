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


---

# Measured GPU cache memory and decode latency -- before and after vectorization

## Update: the latency regression is now mostly fixed

The original benchmark (`notes/measured_cache_memory_latency.md`) found the packed decode path
168-201x slower than full-width dense MLA, and traced this to three unvectorized per-token Python
loop sites: `pack_latents`/`unpack_latents`/`append_packed_latents`
(`code/elastic_mla/elastic_cache.py`) and the rank-mask construction in
`MultiHeadLatentAttention.forward_cached_packed` (`code/elastic_mla/mla.py`). All three were
rewritten to pure vectorized PyTorch operations (`torch.repeat_interleave` plus a single gather for
pack/unpack; a broadcasted inverse-permutation comparison for the rank mask) with **no change to
the packed representation, byte accounting, or numerical results** -- validated by 500 random-trial
cross-checks against the original loop-based implementations (`torch.equal`, zero mismatches) and
the full 49-test unit suite before and after (both pass).

## Results (Tesla T4, batch=8, same benchmark script/config as the original measurement)

| Scale | Metric | Full MLA | Packed (router) -- before | Packed (router) -- after vectorization |
|---|---|---:|---:|---:|
| 30M | Cache bytes (router/full) | -- | 67.4% | 67.4% (unchanged) |
| 30M | Peak allocated (router/full) | -- | 96.5% | 96.5% (unchanged) |
| 30M | Mean decode step | 8.50 ms | 1412.6 ms (167.9x) | **25.48 ms (3.00x)** |
| 122M | Cache bytes (router/full) | -- | 59.6% | 59.6% (unchanged) |
| 122M | Peak allocated (router/full) | -- | 95.8% | 95.8% (unchanged) |
| 122M | Mean decode step | 20.99 ms | 3592.9 ms (201.1x) | **62.13 ms (2.96x)** |

Vectorizing the three Python loop sites gives a **~55-58x decode-step speedup** on the packed
path at both scales, cutting the latency regression from two orders of magnitude to roughly 3x.
Cache-byte and peak-memory ratios are numerically unchanged (as expected -- these fixes only
change how the same computation is expressed, not what is computed).

## Remaining gap

A ~3x per-step latency cost remains, from real, expected sources that vectorizing pack/unpack does
not remove: (1) the packed path still reconstructs a dense `(B, T, d_c)` latent tensor via a gather
before every attention call, an unavoidable consequence of the correctness-first design described
in the paper; (2) `append_packed_latents` still fully unpacks and repacks the entire cache history
on every single-token decode step, rather than appending only the new token in place -- this is an
algorithmic (not just a vectorization) limitation and is the next target for a genuine O(1)-per-step
incremental packed update. We do not claim latency parity with full MLA; we do claim the regression
is now small enough (~3x) to be a plausible target for further, more modest optimization, rather
than a two-orders-of-magnitude blocker.

## Provenance

- `code/elastic_mla/elastic_cache.py`, `code/elastic_mla/mla.py` (vectorized; commit `169c286`)
- `experiments/contextual_router_{30m,122m}/measured_cache_memory_latency_vectorized.json`
- Lightning jobs: `elasticmla-bench-30m-vec-0902`, `elasticmla-bench-122m-vec-0902` (T4, Completed)
- Checkpoint/data SHA-256 verified identical to the original (pre-vectorization) benchmark run.
