# ElasticMLA: Context-Aware Token-Wise Latent Capacity Allocation for Multi-Head Latent Attention

> **Working draft.** Numerical claims in this document use the corrected v4 analysis and the
> pre-specified fresh-window confirmation only. Earlier v3 and four-sequence exploratory results
> are excluded from confirmatory claims.

## Abstract

Multi-head Latent Attention (MLA) reduces autoregressive key-value (KV) cache storage by caching a
compressed latent vector instead of per-head keys and values, but it assigns the same latent width
to every token. We recast token-wise latent width as an **operational, basis-dependent rate
allocation problem**: persistent cache bytes are provably affine in the sum of token rates, and
upper-tail future-loss risk induces a formally ordered spectrum of safe rates between a token's
average and worst reuse offset (Risk-Capacity Ordering, proved here). We implement a variable-width
packed MLA cache with token-specific nested latent prefixes, offsets, ranks, and channel-order
metadata, and a layer-0-full contextual router trained with a straight-through joint-rollout
surrogate for the corresponding rate-penalized Lagrangian. On 30.6M- and 122.1M-parameter MLA
language models, we measure the full risk-capacity spectrum (not only mean and max): required rate
rises monotonically from 9.00%/8.22% of the full latent width at the mean criterion to
74.07%/72.92% at the worst-offset criterion, with an almost scale-invariant normalized
tail-capacity premium (0.651 at 30M, 0.647 at 122M). Diagnostic decomposition shows this separation
reflects **pervasive cancellation across the reuse horizon, not rare catastrophic tokens**: over 93%
of mean-safe positions still have at least one future offset exceeding the loss budget, and the
positive-part mean loss is roughly double the signed mean. The frozen, pre-registered router beats
random and matched-histogram-shuffle placement on 24 untouched sequences per scale (30M: -0.0196
nat, 95% CI [-0.0291,-0.0094]; 122M: -0.0325 nat, CI [-0.0458,-0.0196]) at 68.80%/61.46% of
fixed-width MLA cache bytes. However, against simple **causal heuristic** baselines (absolute
position, token identity, token rarity) selected only on the original training/validation splits
and evaluated at the identical byte budget, the router does not consistently win: it loses to a
trivial position rule at 30M and to a trivial rarity rule at 122M. We report this negative result
transparently. The 122M policy also misses its own +0.15-nat validation-time quality budget on
fresh data (+0.1823 nat realized). Overall, this work provides a formal rate-allocation framework
and a rigorously audited, mostly negative empirical picture: contextual placement beats naive
random allocation but has not yet been shown to beat simple hand-designed heuristics, and no
peak-memory or latency benefit is established because the correctness-first implementation
reconstructs dense temporaries before attention.

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


### 2.3 Relation to token-conditioned and layer-wise MLA variants

Closest in spirit is EG-MLA [7], which gates latent content per token via an embedding-conditioned
mechanism, and CARE [8], which allocates rank across layers using covariance-aware decomposition.
ElasticMLA differs along three axes simultaneously: (i) it changes the *persistent cache width*
itself through a genuinely packed, ragged representation rather than a fixed-width gate; (ii) the
routing decision is trained end-to-end with a straight-through hard-tier joint-rollout surrogate
under an explicit rate penalty, rather than a fixed heuristic or layer-only allocation; and (iii)
we evaluate it against random, shuffled, *and* causal-heuristic exact-byte controls, exposing a
result those comparisons alone would miss: the router does not consistently beat simple hand-
designed rules (Section 5.4). We see this combination -- runtime per-token packed rate allocation,
joint hard-routing optimization, and honest heuristic comparison -- as the paper's main empirical
novelty, while acknowledging that "learn an importance signal, then spend a discrete resource
budget nonuniformly" is a general pattern shared with adaptive-computation and mixture-of-experts
routing more broadly.

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

### 3.1a Terminology and two propositions

We deliberately avoid calling $r_t$ an intrinsic matrix rank. Because MLA's latent coordinates can
be rotated while compensating the up-projections ($c\mapsto Qc$, $W_U\mapsto W_UQ^{-1}$ for
invertible $Q$), a coordinate-wise nested prefix is **basis dependent**: full-model equivalence
under $Q$ does not imply prefix-code equivalence. We therefore call $r_t$ an operational retained-
coordinate rate in the calibrated nested codebook $P_{\ell,r_1}\preceq\cdots\preceq P_{\ell,d_c}=I$.

**Proposition 1 (exact budget affinity).** From the byte formula above, for any two per-sequence
rate allocations $\mathbf r,\mathbf s$ with the same downstream rank sum,
$M_{\text{packed}}(\mathbf r)=M_{\text{packed}}(\mathbf s)$; more generally
$M_{\text{packed}}(\mathbf r)-M_{\text{packed}}(\mathbf s)=b(L-1)\sum_t(r_t-s_t)$. This licenses
exact-byte controls: any two allocations with equal rank sums have identical cache tensor-payload
bytes.

Let $\Delta_{t,j}(r)=\ell_{t+j}(P_rc_t)-\ell_{t+j}(c_t)$ be the signed loss change at future offset
$j\in\{0,\ldots,H-1\}$ from truncating only source token $t$ to rate $r$, and for tail count $k$
let $A_{t,k}(r)$ be the mean of the $k$ largest $\Delta_{t,j}(r)$ (so $k=H$ is the signed mean and
$k=1$ is the maximum). Define the conservative suffix-safe rate
$r^*_{t,k}(\epsilon)=\min\{r\in\mathcal R: A_{t,k}(r')\le\epsilon\ \forall r'\ge r\}$.

**Proposition 2 (risk-capacity ordering).** If $k_1\ge k_2$ then $A_{t,k_1}(r)\le A_{t,k_2}(r)$ for
every $r$, hence $r^*_{t,k_1}(\epsilon)\le r^*_{t,k_2}(\epsilon)$. *Proof.* The mean of the top-$k$
elements of a fixed vector cannot decrease as $k$ shrinks, so the suffix-feasible rank set for
$k_2$ is a subset of that for $k_1$; their minima are ordered. This holds for any realized loss
vector, including the empirically nonmonotone rank-loss curves we observe, and requires no
assumption about the sign or shape of $\Delta_{t,j}$. It is what licenses reporting an ordered
risk-capacity spectrum (Section 5.1) rather than only two endpoints.

Under a bounded-Lipschitz assumption on the future logit map, nested-prefix truncation additionally
admits the sensitivity envelope $|\Delta_{t,j}(r)|\le\sqrt2K_{t,j}\|e_t(r)\|_2$, where $e_t(r)$ is
the discarded latent tail and $K_{t,j}$ a local Lipschitz constant; this gives a monotone safety
bound consistent with, but not proving, the realized risk-capacity ordering. Full derivations,
the constrained joint rate-distortion formulation that motivates the router objective, a formal
statement of the straight-through surrogate's scope, and further falsifiable predictions are given
in `notes/theory_contextual_tail_rate.md`.

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

### 5.1 The full risk-capacity spectrum is monotone and nearly scale-invariant

We extend the two-point mean/max analysis to a full upper-tail spectrum computed from the same
corrected, provenance-tracked windows: for each source position we retain the largest $k$ of the
$H=32$ future loss deltas, for $k\in\{32,16,8,4,2,1\}$, and take the smallest suffix-safe rank.

| $k$ (of 32) | $\alpha=1-k/H$ | 30M mean $r^*$ | 30M $/d_c$ | 122M mean $r^*$ | 122M $/d_c$ |
|---:|---:|---:|---:|---:|---:|
| 32 (mean) | 0.000 | 23.04 | 9.00% | 31.56 | 8.22% |
| 16 | 0.500 | 54.96 | 21.47% | 88.33 | 23.00% |
| 8 | 0.750 | 101.50 | 39.65% | 151.75 | 39.52% |
| 4 | 0.875 | 142.71 | 55.75% | 208.19 | 54.22% |
| 2 | 0.938 | 171.35 | 66.94% | 251.75 | 65.56% |
| 1 (max) | 0.969 | 189.62 | 74.07% | 280.00 | 72.92% |

Every step is monotone nondecreasing at both scales, exactly as Proposition 2 (risk-capacity
ordering) guarantees. The normalized **tail-capacity premium** $\mathrm{TCP}=\mathbb E[r^*_{\max}
-r^*_{\mathrm{mean}}]/d_c$ is 0.6507 (95% CI [0.6335, 0.6673]) at 30M and 0.6470 (CI [0.6283,
0.6641]) at 122M -- nearly identical intervals, making this the most scale-consistent quantitative
result in the paper.

**The separation is driven by pervasive cancellation, not rare catastrophic tokens.** At the rank
that is safe under the signed mean criterion, the positive-part mean loss (mean of $\max(\Delta,0)$
over the horizon) is 2.05x (30M) and 2.31x (122M) larger than the signed mean, and 93.1%/94.1% of
such "mean-safe" positions still have at least one future offset whose loss increase exceeds
$\epsilon$. If harm were concentrated in a few rare spikes, the vast majority of mean-safe positions
would have zero offsets above $\epsilon$; instead nearly all of them do, and the signed mean is kept
small by cancellation against negative excursions elsewhere in the horizon. **Figure 2**
(`figures/elasticmla_risk_spectrum.pdf`) shows the spectrum and this diagnostic.

The normalized 122M-minus-30M difference in the original mean/max endpoints is -0.781 percentage
points for the mean (95% bootstrap CI: [-1.617, +0.076]) and -1.156 points for the maximum (CI:
[-3.849, +1.611]); both include zero. This is not an equivalence test and does not establish scale
invariance of the endpoints, but the tail-capacity premium result above is a stronger and separate
scale-consistency finding computed across the whole spectrum rather than two endpoints.

The per-record raw deltas underlying this spectrum were computed on ephemeral cloud job storage and
are not independently re-auditable; only the aggregate summary (with verified checkpoint/data
provenance) was recovered. See `notes/risk_capacity_spectrum_results.md`.

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
configurations, not an interpolated operating curve. Panel (a) shows only the mean/max endpoints;
see Figure 2 for the full six-point risk-capacity spectrum. Source: `figures/elasticmla_main_results.pdf`.

**Figure 2.** (a) Normalized suffix-safe rate against upper-tail level $\alpha=1-k/H$ at both
scales, with paired sequence-bootstrap 95% bands; monotonicity is guaranteed by Proposition 2. (b)
Signed mean, positive-part mean, and maximum loss delta at the rank that is safe under the signed
mean criterion, annotated with the fraction of records having at least one future offset exceeding
$\epsilon$; the gap between signed and positive-part means shows cancellation rather than rare
spikes. Source: `figures/elasticmla_risk_spectrum.pdf`.

### 5.3 Quality-constraint generalization is imperfect

The 30M policy remains within the +0.15-nat selection budget on fresh sequences. The 122M policy
does not: validation Δloss was +0.1408, whereas fresh Δloss is +0.1823. It nevertheless beats its
same-byte static control. This distinction matters operationally: contextual allocation improves
the Pareto point relative to noncontextual allocation, but a small validation set did not reliably
calibrate an absolute quality constraint. A deployable system should use a larger calibration set,
a conservative risk margin, or online fallback to a higher tier.

### 5.4 The router does not consistently beat simple causal heuristics

The random and matched-histogram-shuffle controls above are content-independent given the router's
realized budget, but they are weak baselines: nothing prevents a simple, cheap, strictly causal
rule (depending only on the current token and/or absolute position, never on future tokens or a
sequence-global budget) from doing better. We fit four such rules -- absolute position, token
identity (frequency-smoothed), inverse token rarity, and coarse token type -- using only the
original 16 training sequences, select an additive rate bias on the original 4 validation
sequences under the same +0.15-nat budget rule as the router, and evaluate once on the same 24
frozen fresh sequences at the router's own per-sequence byte budget.

| Scale | Control | Router − control (nat) | 95% CI | Result |
|---|---|---:|---|---|
| 30M | position | +0.0138 | [0.0025, 0.0254] | **router loses** |
| 30M | lexical identity | -0.0253 | [-0.0399, -0.0114] | router wins |
| 30M | token rarity | -0.0268 | [-0.0412, -0.0124] | router wins |
| 30M | token type | +0.0065 | [-0.0042, 0.0174] | tie |
| 122M | position | +0.0040 | [-0.0098, 0.0171] | tie |
| 122M | lexical identity | +0.0067 | [-0.0044, 0.0177] | tie |
| 122M | token rarity | +0.0468 | [0.0340, 0.0593] | **router clearly loses** |
| 122M | token type | +0.0117 | [-0.0028, 0.0258] | tie |

At 30M a trivial position-dependent rate schedule beats the trained router with a confidence
interval that excludes zero. At 122M a trivial inverse-token-frequency rule beats the router
decisively, and two more controls are statistically tied. Only two of eight scale/control pairs
show a router win with a CI excluding zero. **This narrows the paper's central claim**: contextual
placement is better than *random or shuffled* placement at equal bytes, but current evidence does
not show it is better than the strongest simple hand-designed causal heuristic at that budget. We
report this as an honest negative finding (`notes/causal_heuristic_baseline_results.md`,
`experiments/evaluate_causal_heuristic_routers.py`) rather than omit it: it indicates that the
isolated-position future-loss signal used to construct oracle labels, and the joint-rollout
straight-through training that refines it, are not yet strong enough to reliably out-perform
inexpensive non-learned rules, motivating stronger joint objectives, larger calibration sets, or
hybrid heuristic-plus-learned designs as future work.

## 6. Validity, Reproducibility, and Limitations

**Statistical scope.** The unit of inference is a full sequence, not an individual token. Twenty-
four clusters per scale support paired uncertainty estimates but remain modest. The exact
sign-flip p-values enumerate all sign assignments conditional on the observed magnitudes, but
their inferential interpretation still requires exchangeability of paired-difference signs under
the null. The intervals and tests do not incorporate checkpoint, policy-seed, or training-run
variability. We evaluate one checkpoint per scale, one data distribution, and one router seed. The
per-record raw deltas underlying the risk-capacity spectrum (Section 5.1) were computed on
ephemeral cloud job storage and could not be retrieved after job completion; only the aggregate
summary statistics (independently checkpoint/data-hash verified) were recovered, so that spectrum
is not re-auditable at the per-position level the way the confirmation results are.

**Threshold scope.** Raw rank-loss curves are frequently nonmonotone: 83.6%/89.3% of
mean-over-horizon curves and 59.4%/69.7% of maximum-over-horizon curves at 30M/122M,
respectively. Accordingly, $r^*$ uses the conservative suffix rule and is defined only on the
tested discrete rank grid; it should not be interpreted as a smooth or uniquely identified
intrinsic rank.

**Control scope.** The primary control randomizes rank placement conditional on each router
sequence's realized total rank; it is therefore position-independent, but not a globally fixed-
budget policy. Twenty random allocations estimate its per-sequence loss. Section 5.4 adds four
causal heuristic baselines (position, lexical identity, rarity, type) at the same realized budget;
the router does not consistently beat these, and loses to two of them with high confidence. We
still lack a comparison against separately optimized global-budget static policies, a learned
orthogonal (PCA/SVD) nested basis, and eviction/quantization baselines from the wider KV-cache
compression literature.

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

We formalize token-wise MLA latent width as a basis-dependent operational rate allocation problem:
persistent cache bytes are exactly affine in summed token rates (Proposition 1), and upper-tail
future-loss risk induces a provably monotone spectrum of safe rates between average and worst-case
reuse (Proposition 2). Measured at 30M and 122M, this spectrum is nearly scale-invariant in its
normalized tail-capacity premium (~0.65 at both scales) and shows that the mean/tail separation
arises from pervasive cancellation across the reuse horizon rather than rare catastrophic tokens --
a mechanistic finding that revises the intuitive "rare spike" story. Frozen, pre-registered
contextual routers beat random and shuffled placement at equal cache bytes at both scales, but a
rigorous comparison against simple causal heuristics (position, token identity, rarity, type) shows
the router does not consistently win, and in two cases loses with high confidence. We report this
transparently rather than overclaim: the present evidence establishes a formal allocation framework
and a real but limited placement effect, not superiority over hand-designed heuristics, not
peak-memory or latency gains (the correctness-first implementation reconstructs dense temporaries),
and not a reliably calibrated quality constraint at 122M. Priority future work is (1) stronger joint
-rollout objectives or hybrid heuristic-plus-learned routers that can beat causal baselines with
statistical confidence, (2) a grouped-tier or fused packed attention kernel to convert the
established persistent-byte savings into measured peak-memory and latency gains against optimized
MHA/GQA/MLA/FlashMLA baselines, and (3) replication across more seeds, domains, and a
larger-parameter, more realistically trained checkpoint.

## References (draft)

[1] A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.

[2] DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
Model,” arXiv:2405.04434, 2024.

[3] DeepSeek-AI, “DeepSeek-V3 Technical Report,” arXiv:2412.19437, 2024.

[4] T. Ji et al., “Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in
Any Transformer-based LLMs,” arXiv:2502.14837, 2025.

[5] “Lossless KV Cache Compression to 2%,” arXiv:2410.15252, 2024.

[6] “TransMLA: Multi-Head Latent Attention Is All You Need,” arXiv:2502.07864, 2025.

[7] "EG-MLA: Embedding-Gated Multi-head Latent Attention for Scalable and Efficient LLMs,”
arXiv:2509.16686, 2025.

[8] “CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent
Attention,” arXiv:2603.17946.

[9] Through the Bottleneck: How Multi-head Latent Attention Separates Content from Position in
Language Models,” arXiv:2607.23054.
