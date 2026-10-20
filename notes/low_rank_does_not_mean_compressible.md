# Low-rank second moments do not make a KV cache compressible (2026-09-18)

> **Correction (2026-10-20, `notes/code_review_2026-10_corrections.md`).** The '1.09-1.14x vs 1.65-3.2x' comparison mixes denominators (coord/task_aware at 0.05-0.10 nats vs coord/pca at 0.05-0.5 nats). Like-for-like coord/pca over 0.05-0.5 nats: GPT-2 1.06-1.21x, GPT-2-medium 1.09-1.30x, vs 1.65-3.22x on our models. '94.9% of key energy in 8 directions' is layer 0 head 0 of GPT-2-medium only; over all 384 heads the top-8 share averages 76.6% (range 40-99%). 'The most anisotropic model had the least benefit' is reversed: GPT-2-medium gains more than GPT-2. The 1.88 vs 1.43 pp energy-capture figure has no artifact in the repository.


## How this started, and what it turned into

`notes/kv_cache_basis_generalization.md` found that measuring KV-cache
compressibility in a model's coordinate axes overstates the requirement by 1.65-3.2x
in this project's three models, and proposed that as a claim about the field's
measurement convention. The obvious test was whether it holds on public pretrained
checkpoints. It does not: on GPT-2 and GPT-2 medium the same measurement gives
1.09-1.14x, which is not worth reporting as a correction to anything.

Chasing *why* produced a better result than the one that failed.

## Four attempts, one wall

| attempt | prediction | outcome |
|---|---|---|
| energy-capture law with one fitted constant | predicts `r*` from the covariance spectrum | worse than a two-constant null (1.88pp vs 1.43pp) |
| `model_metric` frame (`notes/basis_dependence_prediction7.md`) | optimal for output-space reconstruction | no reliable gain over plain PCA; covers zero at 122M |
| generalisation to public models | same 2-3x effect | 1.09-1.14x; the *most* anisotropic model had the *least* benefit |
| `task_aware` frame (below) | optimal for score error, so it should recover the gap | **1.01-1.03x over plain PCA** |

The fourth is the decisive one. Dropping direction `u` from a cached key perturbs an
attention score by `-(q.u)(u.k)`, so a criterion built only from `E[kk^T]` is blind to
what the queries actually probe. Treating queries and keys as independent, the
expected squared score error of truncating to a subspace is
`tr((I-P) Sigma_k (I-P) Sigma_q)`, and storing `T k` for `T = U^T Sigma_q^{1/2}` -- `U`
the eigenvectors of `Sigma_q^{1/2} Sigma_k Sigma_q^{1/2}` -- makes the truncation cost
exactly the tail sum of those eigenvalues, the minimum at that rank. Values get the
same treatment with `W_O`'s per-head Gram as the metric, since a value's effect is
what `W_O` writes into the residual stream. The frames are oblique rather than
orthogonal; the derivation and the idempotence of the resulting projections are
checked in `tests/test_public_kv_basis.py`, and on synthetic data the frame cuts score
error roughly fourfold against plain PCA.

On real models it buys 1.7 coordinates out of 64.

## The result

Rank needed per head to stay within a loss budget, WikiText-2:

| model | budget | coord | pca | task_aware |
|---|---|---:|---:|---:|
| gpt2 | 0.10 nats | 57.1 | 52.5 | **50.8** |
| gpt2 | 0.50 nats | 46.1 | 38.1 | **37.0** |
| gpt2-medium | 0.10 nats | 56.5 | 50.6 | **49.6** |
| gpt2-medium | 0.50 nats | 46.3 | 35.5 | **34.7** |

Set against the anisotropy of the very same cached vectors:

| model | key top-8 energy (of 64 dims) | best basis still needs |
|---|---:|---:|
| gpt2 | 71.2% | 50.8 / 64 |
| gpt2-medium | **94.9%** | 49.6 / 64 |

GPT-2 medium concentrates 94.9% of its cached key energy in 8 of 64 directions and
still needs roughly 50 of them. That is the claim:

> **A low-rank second moment does not imply a compressible KV cache.** Directions
> holding a few percent of the energy are not optional; discarding them costs loss
> out of proportion to the energy they carry.

A plausible mechanism, not tested here: attention scores pass through a softmax, so a
small score perturbation is amplified, and squared reconstruction error is simply the
wrong currency. The three failed criteria above are all L2-flavoured, including the
one built from the score error itself -- which suggests the gap is not in the choice
of quadratic form but in using a quadratic form at all.

## Scope: what is and is not shown

**Shown.** Across two public pretrained models, no *fixed linear projection* of the
per-head key and value -- coordinate axes, principal directions, or the score-optimal
oblique frame -- compresses the cache much below about 50 of 64 dimensions at a
0.1-nat budget, despite second moments that look strongly low-rank.

**Not shown.** Nothing about non-linear compression, per-token adaptive rank,
quantisation, or token eviction. Those are most of the methods that work in practice,
and this result is consistent with *why* they work: the compressible structure is not a
fixed subspace. It is, incidentally, an argument for the per-token adaptive rank that
this project's router implements, rather than against it.

**Also not shown.** Only GPT-2 family, only WikiText-2, only MHA without RoPE. A RoPE
model caches keys *after* a position-dependent rotation, which does not commute with a
fixed projection, so the weight-folding method does not transfer unchanged and that
extension is left undone rather than approximated.

## What this does to the earlier claim

The "coordinate-basis measurement overstates compressibility by ~3x" finding stands
for this project's three models and is now bounded: it does **not** generalise to
public pretrained checkpoints, where the same measurement gives 1.1x. The most
economical reading is that the three models -- 3,000 steps on TinyStories -- are
unrepresentatively compressible, which is a caution about the whole project's
empirical base, not only about this measurement. Reporting the 3x without this
control would not survive a reviewer running the same check.

## Artifacts

- `experiments/analyze_public_kv_basis.py`, `tests/test_public_kv_basis.py`
- `experiments/public_kv_basis_gpt2.json`, `experiments/public_kv_basis_gpt2_medium.json`
- Run locally on Apple MPS; `transformers` added as a dependency for this experiment.
