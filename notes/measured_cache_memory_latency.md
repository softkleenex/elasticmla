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

---

# Update 3: in-place incremental append (2026-09-12)

## What changed

The "remaining gap" section above names two residual costs. The second one --
`append_packed_latents` fully unpacking and repacking the entire cache history on every
single-token decode step -- is now fixed, along with a third cost that the earlier diagnosis
missed.

**1. Sequence-major flat layout enables a true in-place append.** The packed `values`/`offsets`
buffers were laid out row-major `(batch, sequence)`, so a new token for batch row 0 landed in the
*middle* of the buffer and every later batch row's offsets shifted -- a full rebuild was
unavoidable. Switching the flat ordering to sequence-major `(sequence, batch)` makes every new
token sort after every stored token, so appending is a plain concatenation of the packed buffers.
`ranks` keeps its `(B, S)` shape, the dense reconstruction is bit-identical, and the byte
accounting is unchanged, so the layout is invisible outside `elastic_cache.py`.

**2. The dominant cost was actually per-call validation, not the repacking.** Profiling the fixed
append showed it still costing ~200x the actual data movement (1.91 ms vs. 0.007 ms for the
`values` concatenation, one 30M layer, B=8, S=320, MPS). The cause is accelerator
synchronisation in the defensive guards, which ran on every decode step for every layer:
`validate_channel_order` copies the order to host and sorts it; `validate_ranks` issued two
separate device-side reductions; `pack_latents` read the final offset twice via `.item()`;
`_ragged_prefix_positions` recomputed `int(flat_ranks.sum())` that `repeat_interleave` had
already determined; and `unpack_latents` audited its offsets with two more `.item()` calls.
Every guard is retained -- none was deleted -- but each now runs where its guarantee is
actually established rather than on every step: full permutation validation happens on the
prefill call that creates the cache's coordinate system, and the decode path checks the order
by equality against the cache's own stored (already validated) order, which is itself a proof
of permutation-ness. The bounds and offset audits were folded into single fused comparisons.

## Measured effect on T4 (paired, 3 repeats per version, one job per scale)

Measured as a *paired* benchmark: the unmodified benchmark script run alternately against the
pre-fix (`c3bed39`) and post-fix (`71634ad`) sources inside one Lightning T4 job per scale, so both
versions share a GPU instance and thermal state. Medians of 3; individual repeats are in the result
JSONs.

| Scale | Config | Before (stage 2) | After (stage 3) | Speedup | Regression vs full MLA |
|---|---|---:|---:|---:|---|
| 30M | Full MLA (untouched control) | 7.88 ms | 7.79 ms | 1.01x | -- |
| 30M | Packed uniform, matched rank | 24.12 ms | 18.71 ms | 1.29x | 3.06x -> 2.40x |
| 30M | Packed router | 24.36 ms | 18.83 ms | 1.29x | **3.09x -> 2.42x** |
| 122M | Full MLA (untouched control) | 17.81 ms | 17.74 ms | 1.00x | -- |
| 122M | Packed uniform, matched rank | 54.31 ms | 41.75 ms | 1.30x | 3.05x -> 2.35x |
| 122M | Packed router | 53.24 ms | 42.11 ms | 1.26x | **2.99x -> 2.37x** |

Two checks that make this comparison trustworthy. First, the untouched dense control moved by 1% or
less within each job, whereas across the *separate* jobs of 2026-09-02 and 2026-09-12 it moved by
4-14% -- which is why the paired design was used rather than differencing two independent runs.
Second, the before column reproduces the separately measured 2026-09-02 result (3.09x vs. the
published 3.00x at 30M; 2.99x vs. 2.96x at 122M), so the paired run is measuring the same thing the
manuscript already reports.

Invariants held exactly on T4 at both scales: `resident_cache_bytes` identical before and after
(30M 11,926,296; 122M 36,541,424), `mean_downstream_rank` identical to four decimals (140.7439;
195.9401 -- i.e. the router made the same decisions), and peak allocated memory unchanged apart
from allocator-level jitter (30M router identical at 577,955,328 B; 122M router identical at
1,472,741,376 B).

- `experiments/contextual_router_30m/measured_cache_memory_latency_append_paired.{json,log}`
  (Lightning job `elasticmla-paired-30m-0912`, T4, Completed)
- `experiments/contextual_router_122m/measured_cache_memory_latency_append_paired.{json,log}`
  (Lightning job `elasticmla-paired-122m-0912`, T4, Completed)

## Measured effect during development (Apple M-series MPS, 30M, batch=8, prefill=192, decode=128, 4 runs, median)

| Config | Before | After | Speedup |
|---|---:|---:|---:|
| Full MLA (dense, untouched control) | 5.54 ms | 5.47 ms | 1.01x |
| Packed uniform, matched rank | 49.67 ms | 31.10 ms | **1.60x** |
| Packed router | 54.82 ms | 35.43 ms | **1.55x** |

Latency regression versus full MLA on this device: **8.96x -> 5.69x** (uniform) and
**9.89x -> 6.48x** (router). A same-process steady-state profile attributes the remaining packed
step as: unpack-for-attention 45.5%, append 27.5%, the dense attention work itself 21.6%.

## Correctness

- 54/54 unit tests pass (five new `IncrementalAppendTest` cases lock in incremental-equals-one-shot
  packing, append-does-not-rewrite-the-existing-prefix, and the relocated int32 overflow guard).
- 500 random-trial cross-check against the previous row-major implementation: dense round-trip,
  incremental append versus one-shot pack, and `packed_cache_num_bytes` all match exactly.
- End-to-end: 30M contextual-router decode (prefill 64 + 24 steps, B=3) produces **bit-identical
  logits** (max absolute difference 0.0), identical chosen ranks, and identical cache bytes
  before and after. No reported number in the manuscript changes.

## Status and what is still open

The MPS numbers above are development measurements on a device with a very different
synchronisation cost profile (its own pre-fix regression is ~9x, not ~3x); the T4 paired
measurement is the one the manuscript cites. The manuscript's Section 5.5, abstract, Section 6
system-scope paragraph and conclusion have been updated from "roughly 3x" to 2.4x accordingly.

The first residual cost named above is untouched and is now the dominant one: the packed path
still reconstructs the dense latent history via a gather before every attention call. That is
inherent to the correctness-first design -- the persistent representation is packed, so a decoder
that wants dense `c_kv` must rebuild it. Removing it means either keeping a dense working set
resident (which would make the persistent-byte saving irrelevant at decode time, and is a claim
change, not an optimization) or computing `W_UK`/`W_UV` directly against the ragged packed buffer.
