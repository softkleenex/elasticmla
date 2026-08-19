# ElasticMLA: Context-Aware Token-Wise Latent Capacity Allocation for Multi-Head Latent Attention

> **Working draft.** Numerical claims in this document use the corrected v4 analysis and the
> pre-specified fresh-window confirmation only. Earlier v3 and four-sequence exploratory results
> are excluded from confirmatory claims.

## Abstract

Multi-head Latent Attention (MLA) reduces autoregressive key-value (KV) cache storage by caching a
compressed latent vector instead of per-head keys and values, but it assigns the same latent width
to every token. We investigate whether the required latent capacity varies by token and whether a
contextual router can exploit this variation without merely shifting memory among positions. We
first implement a variable-width packed MLA cache that stores token-specific latent prefixes,
offsets, ranks, channel-order metadata, and decoupled rotary keys. On 30.6M- and 122.1M-parameter
MLA language models, a corrected isolated-position intervention shows the same descriptive separation
between average and tail capacity: the mean required rank is 9.00% and 8.22% of the full latent
width, whereas the maximum future-loss criterion requires 74.07% and 72.92%, respectively. We
then introduce a layer-0-full contextual router and train it with straight-through hard tier
selection under a joint-rollout language-model loss and expected-rank penalty. Policies were
frozen using 16 training and four validation sequences and evaluated once on 24 new,
nonoverlapping sequences per scale. Relative to a random position-independent static interpolation, conditional on the router's realized per-sequence budget and with
exactly the same cache tensor-payload bytes, the router reduces cross-entropy by 0.0196 nat at 30M
(95% paired bootstrap CI: [0.0094, 0.0291] improvement) and 0.0325 nat at 122M (CI: [0.0196,
0.0458]), while using 68.80% and 61.46% of fixed-width MLA cache tensor-payload bytes. The 30M policy
incurs +0.1001 nat versus full MLA; the 122M policy incurs +0.1823 nat and therefore misses its
+0.15-nat validation-time quality budget. These results provide evidence for contextual capacity allocation as
a storage-quality effect at two small scales, but do not establish latency or peak-memory
improvements because the correctness-first implementation reconstructs dense temporary latents.

**Keywords:** multi-head latent attention, KV cache compression, adaptive computation, dynamic
rank, contextual routing, language-model inference

## 1. Introduction

Autoregressive transformer inference stores a key and value representation for each past token in
each attention layer. This KV cache grows linearly with context length, batch size, layer count,
and attention width, and increasingly constrains serving capacity. Multi-head Latent Attention
(MLA), introduced in DeepSeek-V2 [2] and retained in DeepSeek-V3 [3], addresses this cost by
caching a shared low-dimensional latent representation and reconstructing per-head content keys
and values when needed. Related work converts pretrained multi-head or grouped-query attention
models to MLA [4] and combines low-rank attention with layer sharing or quantization [5].

Existing MLA designs nevertheless use one fixed latent width for all tokens. This is conservative
if difficult tokens are rare: the width must accommodate tail cases even when most cached states
can tolerate stronger compression. Conversely, naïve average-capacity routing can fail because a
small number of under-provisioned tokens may affect many future predictions. This tension raises
three questions:

1. Does token-level latent capacity vary under a future-loss criterion, and does the pattern
   replicate across model scales?
2. Can a router use contextual hidden states to place capacity better than a noncontextual policy
   at exactly the same persistent-cache tensor-payload byte budget?
3. Can token-wise ranks be represented as actual packed storage rather than only simulated by a
   dense mask?

We answer these questions with **ElasticMLA**, a correctness-first prototype for token-specific MLA
latent widths. The core design preserves a full-width first layer, predicts one discrete tier from
the resulting contextual state, and shares the selected tier across downstream layers. This makes
the routing decision contextual while avoiding the circular requirement of compressing the state
needed to decide its own capacity.

Our contributions are:

- a genuine variable-width packed latent cache with token-level ranks and verified dense/full-rank
  equivalence;
- a corrected, provenance-tracked future-loss intervention that separates average from tail
  capacity at 30M and 122M scales;
- a joint-rollout straight-through router objective that optimizes hard deployment tiers while
  regularizing expected rank; and
- a pre-specified, untouched 24-sequence-per-scale confirmation against exact-byte static and
  matched-histogram shuffle controls.

The strongest supported conclusion is narrow but useful: at both evaluated scales, contextual
allocation produces lower next-token loss than random position-independent allocation conditional on the router's realized per-sequence budget at the same
persistent cache tensor-payload bytes. We do not claim optimized serving speed, lower peak memory,
scale invariance, or reliable satisfaction of a fixed quality constraint at 122M.

## 2. Background and Related Work

### 2.1 Multi-head Latent Attention

For hidden state $h_t$, MLA computes a compressed KV latent

$$c_t = W_{DKV}h_t \in \mathbb{R}^{d_c},$$

then reconstructs content keys and values using $W_{UK}c_t$ and $W_{UV}c_t$. A separate rotary
key $k_t^R \in \mathbb{R}^{d_R}$ carries positional information. A fixed-width MLA cache stores
$(c_t,k_t^R)$ instead of per-head K/V tensors. For sequence length $T$, $L$ layers, and element
size $b$, its leading storage term is

$$M_{\text{MLA}} = bTL(d_c+d_R).$$

DeepSeek-V2 reports substantial KV-cache reduction and throughput gains from this representation
[2]. MHA2MLA [4] and TransMLA [6] study converting existing architectures to MLA. These works
motivate latent compression but do not directly establish token-wise variable-width storage.

### 2.2 KV-cache compression and adaptive allocation

KV-cache compression includes eviction, quantization, low-rank approximation, cross-layer
sharing, and sparse attention. Cross-Layer Latent Attention combines several axes and reports
aggressive compression [5]. Such methods show that cache states contain redundancy, but a global
compression rate can conceal variation across positions. ElasticMLA differs by allocating a
nested latent prefix to every token while retaining all token positions.

Dynamic allocation is challenging because quality depends jointly on many cached states. An
isolated token intervention gives useful sensitivity labels but is not a joint-rollout oracle.
Our early supervised routers exposed this mismatch: high accuracy on isolated mean-risk labels
collapsed toward the smallest tier during simultaneous compression. This negative result motivates
direct optimization under the deployed hard-routing rollout.

## 3. Method

### 3.1 Nested latent prefixes and packed storage

For each layer $\ell$, calibration produces a permutation $\pi_\ell$ of the $d_c$ latent channels
using gradient-times-activation importance. A token assigned rank $r_t$ stores the nested prefix

$$\tilde c_{\ell,t}=c_{\ell,t}[\pi_\ell(1:r_t)].$$

The persistent representation contains a values buffer, int32 offsets, int16 ranks, an int16
channel permutation, and the decoupled RoPE key. For the evaluated batch-one configurations, its
exact cache tensor-payload byte count is

$$M_{\text{packed}}=
b\left[Td_c+(L-1)T\bar r+TLd_R\right]
+4L(T+1)+2LT+2Ld_c,$$

where $\bar r=T^{-1}\sum_t r_t$. The final three terms count int32 offsets, int16 ranks, and
int16 channel permutations. The implementation uses $b=4$. This excludes allocator rounding,
framework-object overhead, router weights, and workspace memory. During attention, the prototype
reconstructs a dense temporary latent tensor; the metric is therefore neither process/device
memory nor a peak-memory or kernel-latency measurement.

### 3.2 Corrected future-loss capacity analysis

For each source position $p$ and candidate rank $r$, we simultaneously zero channels outside the
layer-specific nested prefix at position $p$ in every layer; all other positions remain full rank.
Let $\delta_{p,j}(r)$ denote the resulting change in next-token cross-entropy at future-loss
offset $j\in\{0,\ldots,H-1\}$. We define

$$A_{p,\mathrm{mean}}(r)=\frac{1}{H}\sum_{j=0}^{H-1}\delta_{p,j}(r),\qquad
A_{p,\mathrm{max}}(r)=\max_{0\le j<H}\delta_{p,j}(r),$$

and, for $A\in\{\mathrm{mean},\mathrm{max}\}$,

$$r^*_{p,A}=\min\{r\in\mathcal R:
A_p(r')\le\epsilon\;\text{for every}\;r'\in\mathcal R,\;r'\ge r\}.$$

The suffix condition guards against nonmonotone rank-loss curves. We report the mean of
$r^*_{p,A}$ over 768 probed positions, separately for the mean- and maximum-over-horizon criteria,
with $H=32$ and $\epsilon=0.10$ nat. Version 4 fixes an earlier off-by-one error by including
`logits[p]`, the first prediction affected by masking source position $p$. Calibration repeats,
calibration windows, and evaluation windows are nonoverlapping. Each scale uses 24 sequences and
32 probed positions per sequence, with checkpoint, data, source, and record hashes stored for audit.

### 3.3 Contextual router

Layer 0 runs at full rank. Let $z_t$ be the normalized pre-attention state entering layer 1 after
the layer-0 contextual update. A two-layer MLP router produces tier logits

$$a_t=g_\theta(z_t), \qquad r_t\in\mathcal T.$$

The selected tier is shared by all downstream layers. The 30M policy uses
$\mathcal T=\{16,64,160,256\}$. The 122M rollout policy uses a finer 14-tier grid from 16 to 384,
with 16-step spacing up to 64 and 32-step spacing thereafter.

### 3.4 Joint-rollout training

Deployment uses the hard one-hot choice $y_t=\text{onehot}(\arg\max a_t)$. During training we use
the straight-through estimator

$$\hat y_t = y_t + p_t-\operatorname{stopgrad}(p_t),\quad
p_t=\operatorname{softmax}(a_t/\tau).$$

The forward pass therefore matches hard-tier inference, while gradients follow the soft
probabilities. The frozen base language model is optimized only through the router objective

$$\mathcal J(\theta)=\mathcal L_{LM}(\hat y)+\lambda\frac{1}{Td_c}
\sum_t\sum_{k}p_{t,k}\mathcal T_k.$$

The base model receives no gradients. Candidate $\lambda$ values are trained on 16 sequences and
selected on four validation sequences. The selection rule chooses the minimum-byte candidate
within a +0.15-nat validation loss budget that also beats an exact-byte static control. The chosen
policies are $\lambda=0.4$ at 30M and $\lambda=0.8$ at 122M. Four older development sequences
were repeatedly observed and are treated only as exploratory; they are not used for the final
claim.

### 3.5 Exact-byte controls

For the primary static control, let the router's downstream rank sum on one sequence be $R$.
Writing $R=qT+m$, the control assigns rank $q+1$ to $m$ random positions independently of token
content and rank $q$ to all others. Twenty allocations are averaged. It has exactly the same
downstream rank sum and cache tensor-payload bytes as the router, while approximating a continuous
uniform rank. This placement is position-independent only after borrowing the router's
content-dependent realized budget for that sequence; it is not a globally fixed-budget policy.

The secondary control permutes the router's exact tier histogram across positions 20 times. It
preserves both the cache tensor-payload byte budget and the marginal tier distribution, isolating whether contextual
rank placement matters.

## 4. Experimental Setup

### 4.1 Models and data

We train two decoder-only MLA language models on TinyStories-tokenized data.

| Model | Parameters | Layers | $d_{model}$ | Heads | $d_c$ | $d_R$ | Context | Training tokens seen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLA-30M | 30.6M | 6 | 384 | 6 | 256 | 32 | 256 | 24.576M sampled tokens |
| MLA-122M | 122.14M | 12 | 768 | 12 | 384 | 32 | 384 | 147.46M tokens |

The 30M model was trained for 3,000 steps from an approximately 8.8M-token training corpus and its checkpoint has validation loss 1.9618. The 122M final step-8,000 checkpoint has validation
loss 1.5662; the best logged value, 1.5179 at step 7,250, was not checkpointed. We therefore make
no best-checkpoint claim.

### 4.2 Confirmation protocol

Commit `a4bcc7f` froze both policy files, their SHA-256 hashes, seed 91,827, 24 fresh sequences per
scale, 20 control permutations, endpoints, and the success rule before examining new results.
Fresh windows are sampled from the held-out evaluation region and separated from all 24 prior
oracle/development windows and from one another by at least `block_size + 1` tokens. The primary
endpoint is paired per-sequence router loss minus mean exact-byte static loss. A scale passes when
the upper bound of its 95% sequence-cluster bootstrap interval is below zero. We also report exact
one-sided paired sign-flip tests and sequence win counts.

## 5. Results

### 5.1 Mean and tail capacity separate at both scales

| Scale | Mean-over-horizon: mean $r^*$ | Mean / $d_c$ | Max-over-horizon: mean $r^*$ | Mean / $d_c$ |
|---|---:|---:|---:|---:|
| 30M | 23.04 | 9.00% | 189.63 | 74.07% |
| 122M | 31.56 | 8.22% | 280.00 | 72.92% |

The large typical-versus-tail separation replicates descriptively. The normalized 122M-minus-30M
difference is -0.781 percentage points for the mean (95% bootstrap CI: [-1.617, +0.076]) and
-1.156 points for the maximum (CI: [-3.849, +1.611]). Both intervals include zero. This is not an
equivalence test and does not establish scale invariance.

### 5.2 Fresh contextual routing beats exact-byte static allocation

| Scale | Mean rank | Packed / fixed MLA | Δloss vs full | Router − static | 95% CI | Wins | Exact $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30M | 145.78 / 256 | 68.80% | +0.1001 | **-0.01959** | [-0.02911, -0.00938] | 20/24 | 0.000655 |
| 122M | 206.93 / 384 | 61.46% | +0.1823 | **-0.03251** | [-0.04575, -0.01958] | 21/24 | 0.0000493 |

Both scales satisfy their separately pre-specified primary criteria. The protocol did not
pre-specify a family-wise two-scale test, so we do not elevate the post-hoc relationship between
the two-sided intervals and a Bonferroni bound to a confirmatory claim. The conclusion is
allocation efficiency at the realized budget, not lossless compression.

The exact-histogram shuffle comparison is stronger descriptively: router-minus-shuffle is -0.07989
nat at 30M (CI: [-0.08999, -0.06956]) and -0.13907 nat at 122M (CI: [-0.15439, -0.12408]), with
24/24 sequence wins at both scales. This supports the interpretation that contextual placement,
not merely the tier histogram, contributes to quality.

**Figure 1.** (a) Mean required rank under the mean- and maximum-over-32-token-horizon criteria,
averaged over 768 probed positions per scale. (b) Per-sequence router-minus-control differences;
dots are sequences, diamonds are means, and error bars are paired sequence-bootstrap 95%
intervals. (c) Mean fresh-sequence loss increases versus full MLA; both controls have exactly the
router's per-sequence cache tensor-payload byte count. Points are discrete evaluated
configurations, not an interpolated operating curve. Source: `figures/elasticmla_main_results.pdf`.

### 5.3 Quality-constraint generalization is imperfect

The 30M policy remains within the +0.15-nat selection budget on fresh sequences. The 122M policy
does not: validation Δloss was +0.1408, whereas fresh Δloss is +0.1823. It nevertheless beats its
same-byte static control. This distinction matters operationally: contextual allocation improves
the Pareto point relative to noncontextual allocation, but a small validation set did not reliably
calibrate an absolute quality constraint. A deployable system should use a larger calibration set,
a conservative risk margin, or online fallback to a higher tier.

## 6. Validity, Reproducibility, and Limitations

**Statistical scope.** The unit of inference is a full sequence, not an individual token. Twenty-
four clusters per scale support paired uncertainty estimates but remain modest. The exact
sign-flip p-values enumerate all sign assignments conditional on the observed magnitudes, but
their inferential interpretation still requires exchangeability of paired-difference signs under
the null. The intervals and tests do not incorporate checkpoint, policy-seed, or training-run
variability. We evaluate one checkpoint per scale, one data distribution, and one router seed.

**Threshold scope.** Raw rank-loss curves are frequently nonmonotone: 83.6%/89.3% of
mean-over-horizon curves and 59.4%/69.7% of maximum-over-horizon curves at 30M/122M,
respectively. Accordingly, $r^*$ uses the conservative suffix rule and is defined only on the
tested discrete rank grid; it should not be interpreted as a smooth or uniquely identified
intrinsic rank.

**Control scope.** The primary control randomizes rank placement conditional on each router
sequence's realized total rank; it is therefore position-independent, but not a globally fixed-
budget policy. Twenty random allocations estimate its per-sequence loss. The experiments do not
compare against stronger heuristic or separately optimized byte-matched allocation policies.

**Oracle scope.** Capacity labels are isolated-position interventions, not joint-rollout-optimal
labels. Joint training alleviates but does not solve this limitation.

**System scope.** Packed storage is real and cache tensor-payload byte counts include values and metadata. However, the
prototype reconstructs dense temporaries before attention. We therefore do not claim improved
peak memory, latency, throughput, energy, or superiority to optimized MHA/GQA/FlashMLA kernels.

**Quality scope.** Both compressed policies increase loss relative to full MLA, and the 122M policy
misses the held-out +0.15-nat budget. Task-level generation quality is not evaluated.

**Architecture scope.** Layer 0 is always full, and one tier is shared by every downstream layer.
Per-layer routing could improve efficiency but increases metadata, training complexity, and
kernel divergence.

**Reproducibility.** Result artifacts contain checkpoint/data/policy/source hashes and exact starts.
A post-hoc audit reconstructs the pre-specified sampler, verifies nonoverlap, derives cache bytes
from authenticated checkpoint dimensions, and recomputes all summary statistics. Deterministic
training replays reproduce the frozen router tensors, histories, splits, and hyperparameters
bit-for-bit. The original cloud commands preceded the later manifest-enforcement arguments; this
is disclosed, and the frozen protocol commit plus complete logs and audits bind the original
results. Future runs fail unless their manifest inputs match.

## 7. Conclusion

ElasticMLA replaces a single fixed MLA latent width with contextual token-level tiers stored in a
genuine packed cache. Corrected interventions show that typical and tail latent requirements are
widely separated at both 30M and 122M scales. More importantly, frozen rollout-trained routers
beat random position-independent exact-byte static allocations conditional on each router budget on untouched sequences at both scales.
This provides small-scale evidence for contextual allocation in persistent cache storage. The next
systems step is to remove dense reconstruction through grouped-tier or fused packed attention and
to benchmark peak memory and latency against optimized MHA, GQA, MLA, and FlashMLA baselines. The
next modeling step is conservative quality calibration across more seeds, domains, and scales.

## References (draft)

[1] A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.

[2] DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
Model,” arXiv:2405.04434, 2024.

[3] DeepSeek-AI, “DeepSeek-V3 Technical Report,” arXiv:2412.19437, 2024.

[4] T. Ji et al., “Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in
Any Transformer-based LLMs,” arXiv:2502.14837, 2025.

[5] “Lossless KV Cache Compression to 2%,” arXiv:2410.15252, 2024.

[6] “TransMLA: Multi-Head Latent Attention Is All You Need,” arXiv:2502.07864, 2025.
