# The coordinate-basis measurement bias is not specific to MLA (2026-09-17)

> **Correction (2026-10-20, `notes/code_review_2026-10_corrections.md`).** Two qualifications: only the content key (W_UK c) and value are truncated, never the RoPE key, so this is not quite an 'ordinary' per-head KV cache; and the coord/pca ratio peaks at rank 48 at 30M/122M (20.0x, 10.8x) but at rank 32 at 250M (8.8x).


## Why this experiment exists

`notes/basis_dependence_prediction7.md` shows that measuring how far an MLA latent can be
truncated **in the model's coordinate axes** overstates the requirement by roughly 3x. On its own
that is a statement about this architecture, and the obvious objection is that it is a quirk of
MLA's latent. Most of the KV-compression literature does not compress an MLA latent; it compresses
ordinary per-head key and value vectors. So the question is whether the same bias is there.

## Method

Truncating a cached key to the top `k` directions of a per-head orthonormal basis and
reconstructing is exactly `k_hat = P k` for the rank-`k` projection `P`. Attention scores become
`q . P k`, which needs no change to the query path because `P` is symmetric. Folding `P` into the
rows of `W_UK` that belong to that head therefore *is* the truncated-cache model; likewise `W_UV`
for values. Rotations are block-diagonal over heads, because a KV cache stores each head's vector
separately -- a global rotation across the concatenated heads would be an easier object than a
per-head compressor can actually use, and would overstate the effect.

Two differences from the risk-capacity probe, both deliberate:

- **Every position is truncated, not one.** Cache compression compresses the whole cache, so a
  uniform truncation is the comparable measurement and is what the literature's numbers describe.
- **The endpoint is validation loss**, not a per-token future-loss criterion, for the same reason.

Bases, per head: `coord` = the head's own axes ordered by energy `E[x_i^2]` (the axis-aligned
choice, and the analogue of what is usually reported), `pca` = principal directions of that head's
second moment. `experiments/analyze_kv_basis.py`, `tests/test_kv_basis.py`.

## Result: the bias is there at all three scales

`d_head = 64` at every scale. Rank needed to stay within a loss budget, interpolated on the loss
curve so the answer is not quantised to the rank grid:

| loss budget | 30M | 122M | 250M |
|---|---|---|---|
| 0.05 nats | 53.6 / 27.8 = **1.92x** | 51.5 / 31.2 = **1.65x** | 50.7 / 30.7 = **1.65x** |
| 0.10 nats | 47.7 / 21.9 = **2.17x** | 46.0 / 24.8 = **1.85x** | 45.4 / 23.5 = **1.93x** |
| 0.25 nats | 39.0 / 14.3 = **2.72x** | 37.3 / 18.4 = **2.03x** | 36.7 / 15.2 = **2.42x** |
| 0.50 nats | 30.7 / 9.5 = **3.22x** | 29.7 / 12.8 = **2.31x** | 29.0 / 10.0 = **2.91x** |

(coordinate axes / principal directions.) The factor is 1.65-2.2x at tight budgets and 2.3-3.2x
when compressing aggressively: the harder the compression, the more the basis matters.

Read the other way, at a fixed retained rank the coordinate basis costs far more loss. At rank 24
of 64 it costs +0.79/+0.78/+0.73 nats at 30M/122M/250M against +0.068/+0.107/+0.092 for the
principal basis -- gaps of 11.5x, 7.3x and 7.9x -- and the gap peaks near rank 40-48 at 20x, 11x
and 9x.

## The mechanism is the same one, and it survives inside a 64-dimensional head

Effective rank `tr(Sigma)/lambda_1` of the cached vector, layer 0 head 0, and the share of energy
in the top 8 directions:

| scale | key eff. rank | top-8 energy (eigen / axes) | value eff. rank | top-8 energy (eigen / axes) |
|---|---:|---|---:|---|
| 30M | 5.15 / 64 | 69.3% / 19.9% | 11.58 / 64 | 41.0% / 16.9% |
| 122M | 6.17 / 64 | 62.4% / 16.8% | 13.40 / 64 | 35.9% / 18.0% |
| 250M | 8.94 / 64 | 53.3% / 17.0% | 17.04 / 64 | 29.8% / 15.3% |

The cached vectors are strongly anisotropic even within a single head, and the coordinate axes see
almost none of it: the top 8 axes hold 15-20% of the energy at every scale, close to the 12.5% a
perfectly flat spread would give, while the top 8 principal directions hold 30-69%.

Keys are consistently more anisotropic than values, and both become *less* anisotropic with scale
(key effective rank 5.15 -> 8.94), which is the direction that makes the bias slightly smaller at
250M than at 30M.

## What this licenses, and what it does not

**Licensed.** The bias is not an MLA artifact. It appears, in the same direction and at a
comparable magnitude, for the ordinary per-head key/value cache that the compression literature
actually targets, under a whole-cache truncation and a plain validation-loss endpoint.

**Not licensed.** These are three models trained here on TinyStories, not public pretrained
checkpoints. The claim that published compression ratios are systematically understated would need
the same measurement on models the field actually uses. The machinery is architecture-agnostic --
per-head second moments, a rotation, folded into the projection weights -- so this is a resourcing
question, not a methodological one.

**Also not licensed.** "The measurement overstates the requirement" is not the same as "you can
therefore compress further in practice". Realising it needs the rotation to be part of the deployed
model, which is free here (it folds into `W_UK`/`W_UV` at identical cache bytes) but has not been
demonstrated end to end with a trained router.

## Artifacts

- `experiments/analyze_kv_basis.py`, `tests/test_kv_basis.py` (7 tests pinning the premise:
  folding the projection equals projecting the cached vectors head by head, full rank is an
  identity, rank-1 leaves each head rank 1 with no cross-head mixing, and a below-full rank does
  change the output so the identity checks cannot pass vacuously)
- `experiments/contextual_router_{30m,122m,250m}/kv_basis.json`
- `experiments/kv_basis_kernel.log`; kernel `softkleenex/elasticmla-kv-basis` v2 (Kaggle T4)
