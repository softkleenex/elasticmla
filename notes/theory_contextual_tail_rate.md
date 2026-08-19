# Contextual Tail-Rate Allocation Theory for ElasticMLA

## Scope and terminology

ElasticMLA should not describe a token's retained coordinate count as an intrinsic matrix rank.
For layer `l`, a calibrated permutation `pi_l` defines a nested codebook

\[
P_{l,r_1} \preceq P_{l,r_2} \preceq \cdots \preceq P_{l,d_c}=I,
\]

where `P_{l,r}` retains the first `r` coordinates in that basis. The selected `r` is therefore an
**operational retained-coordinate rate**. It is basis dependent. Indeed, for any invertible `Q`,
replacing `c` by `Qc` and each up-projection `W_U` by `W_U Q^{-1}` preserves the full-width model
but generally changes every coordinate prefix. Full-model equivalence does not imply prefix-code
equivalence.

This limitation is also an opportunity: channel ordering or an orthogonal rotation is part of the
code design, not merely an analysis convenience.

## 1. Exact affine cache-rate identity

For batch one, `T` cached tokens, `L` layers, layer 0 full, and a shared downstream rate `r_t`, the
implemented packed cache tensor-payload bytes are

\[
C(\mathbf r)=b\left[Td_c+(L-1)\sum_{t=1}^T r_t+TLd_R\right]
+4L(T+1)+2LT+2Ld_c.
\]

The final terms are int32 offsets, int16 ranks, and int16 channel orders. Hence

\[
C(\mathbf r)-C(\mathbf s)=b(L-1)\sum_t(r_t-s_t).
\]

**Proposition 1 (exact budget equivalence).** Two allocations with the same downstream rank sum
have exactly the same cache tensor-payload bytes in the evaluated implementation.

This identity licenses a rate-allocation analysis and exact-byte controls. It does not cover
allocator rounding, object overhead, temporary tensors, router weights, latency, or peak device
memory.

## 2. Future-loss risk and safe operational rate

When only source token `t` is truncated and all other positions remain full, define its signed
future loss delta at offset `j` as

\[
\Delta_{t,j}(r)=\ell_{t+j}(P_r c_t)-\ell_{t+j}(c_t),\qquad j=0,\ldots,H-1.
\]

For a tail count `k`, let

\[
A_{t,k}(r)=\frac1k\sum_{j\in \operatorname{TopK}(\Delta_{t,0:H-1}(r),k)}
\Delta_{t,j}(r).
\]

`k=H` is the signed mean and `k=1` is the maximum. Intermediate values are empirical upper-tail
means of truncation-induced loss deltas. They are not population CVaR and not the difference of
CVaRs of two loss distributions.

Because rank-loss curves are often nonmonotone, define the suffix-safe rate on grid `R`:

\[
r^*_{t,k}(\epsilon)=\min\{r\in\mathcal R:
A_{t,k}(r')\le\epsilon\ \text{for every}\ r'\in\mathcal R,r'\ge r\}.
\]

**Proposition 2 (risk-capacity ordering).** If `k_1 >= k_2`, then

\[
A_{t,k_1}(r)\le A_{t,k_2}(r)\quad\forall r,
\qquad r^*_{t,k_1}(\epsilon)\le r^*_{t,k_2}(\epsilon).
\]

*Proof.* The mean of the largest `k` elements cannot decrease when `k` is reduced. Therefore the
suffix-feasible set under `k_2` is a subset of that under `k_1`; their minimum grid elements are
ordered. No monotonic rank-loss curve or nonnegative delta assumption is needed.

Define the normalized **tail-capacity premium**

\[
\operatorname{TCP}_H(\epsilon)=\frac{1}{d_c}
\mathbb E_t[r^*_{t,1}(\epsilon)-r^*_{t,H}(\epsilon)]\ge0.
\]

This is a grid- and basis-dependent robust-rate premium. It measures the extra retained-coordinate
rate required when protecting the worst future reuse offset rather than the average offset.

A signed mean can be small because positive and negative deltas cancel. With
`Delta^+=max(Delta,0)`, however,

\[
\bar\Delta^+\le\max_j\Delta_j^+\le H\bar\Delta^+,
\qquad
\frac1H|\{j:\Delta_j^+>a\}|\le\bar\Delta^+/a.
\]

Accordingly, a “rare harmful spike” interpretation must be supported by positive-part and
exceedance diagnostics, not by signed mean/max alone.

## 3. A conservative sensitivity envelope

Let `e_t(r)` concatenate discarded latent-prefix tails over the intervened layers. Nested prefixes
imply `||e_t(r')||_2 <= ||e_t(r)||_2` for `r' >= r`. Assume the future logit map at offset `j` is
`K_{t,j}`-Lipschitz along the bounded intervention path. Multiclass cross-entropy has logit-gradient
norm at most `sqrt(2)`, so

\[
|\Delta_{t,j}(r)|\le\sqrt2 K_{t,j}\|e_t(r)\|_2.
\]

Thus

\[
|A_{t,H}(r)|\le \frac{\sqrt2}{H}\sum_jK_{t,j}\|e_t(r)\|_2,
\quad
|A_{t,1}(r)|\le\sqrt2\max_jK_{t,j}\|e_t(r)\|_2.
\]

This is a monotone safety envelope even when realized signed curves are nonmonotone. It is a
formal sensitivity bound, not a claim that residual energy tightly predicts language-model loss.

## 4. Contextual joint rate-distortion problem

Let `Z_t` be the causal layer-0 feature, `pi_theta(Z_t)` a tier, and `L_LM(r;X)` the simultaneous
hard-routing loss. The operational primal is

\[
\min_\theta D(\theta)=\mathbb E[L_{LM}(\mathbf r_\theta;X)-L_{LM}(\mathbf d_c;X)]
\quad\text{s.t.}\quad
R(\theta)=\mathbb E\left[\frac1{Td_c}\sum_t r_{\theta,t}\right]\le\rho.
\]

Its Lagrangian is `D(theta)+lambda(R(theta)-rho)`. The constant `-lambda rho` can be dropped during
training, matching the implemented loss-plus-expected-rate objective.

For finite tiers and randomized mixtures of policies, the achievable expected `(R,D)` region is
convex and supported Pareto points admit a Lagrange multiplier. A deterministic neural router with
nonconvex finite-sample training has no strong-duality or monotone-in-lambda guarantee; the lambda
sweep searches operating points rather than exactly solving the primal.

The joint loss is nonseparable because cached tokens are reused and truncation effects interact.
Under the explanatory special case

\[
D(\mathbf r,Z)=\sum_t d_t(r_t,Z),
\]

the Lagrangian decomposes into a contextual marginal-value rule:

\[
r_t^\lambda(Z)\in\arg\min_{r\in\mathcal T}
\{d_t(r,Z)+\lambda r/d_c\}.
\]

This approximation is falsifiable through marginal upgrade/downgrade and pair-interaction tests.
It is not assumed by the present method.

## 5. Value of contextual information at a fixed budget

For a realized per-sequence budget `B`, let `A_B` be all allocations with the same rank sum. Let
`Z` denote causal contextual features and `X` the sequence. The Bayes contextual allocator has risk

\[
V_{ctx}(B)=\mathbb E[\min_{a\in A_B}\mathbb E[L(a,X)\mid Z]],
\]

whereas the best allocator that ignores `Z` has

\[
V_{static}(B)=\min_{a\in A_B}\mathbb E[L(a,X)].
\]

**Proposition 3 (value of contextual information).** `V_ctx(B) <= V_static(B)`.

*Proof.* Policies that ignore `Z` are a subset of policies allowed to depend on `Z`.

Strict inequality requires conditional marginal rate value to vary predictably with context. A
learned router is not guaranteed to attain `V_ctx`, and the current experiment compares it with
specific random/static allocations rather than the optimal context-free policy. Stronger lexical,
position, and heuristic controls are therefore necessary.

## 6. Straight-through optimization statement

The implementation uses

\[
g=y_{hard}+p-\operatorname{stopgrad}(p).
\]

The forward value is exactly the deployed hard tier, while derivatives flow through `p`. The rate
term uses the soft expected tier. Therefore training supplies a **surrogate pseudogradient aligned
with the rate-penalized primal**; it is not the exact gradient of the discrete policy objective and
carries no global optimization guarantee. Selection must use realized hard loss and bytes.

## 7. Falsifiable predictions

1. As upper-tail count decreases, safe operational rate must rise or remain equal (theorem check),
   and the empirical curve quantifies the tail-capacity premium.
2. If the mean/max gap reflects sparse harm rather than cancellation, mean-safe rates should have
   few offsets above epsilon and a modest positive-part mean.
3. Worst-offset safe rate should rise or saturate with a longer matched-prefix horizon; signed mean
   need not be monotone.
4. Assigned tiers should correlate with realized marginal loss saved per byte under joint rollout.
5. Pairwise intervention residuals should be nonzero if isolated-label supervision is structurally
   misaligned with joint routing.
6. A contextual router should beat position-only and lexical-only exact-histogram placements if
   hidden-state context contributes information beyond token identity and location.
7. A learned orthogonal nested basis should reduce safe rates or nonmonotonicity relative to a raw
   coordinate permutation if basis design is a material bottleneck.

These predictions separate theorem-implied orderings from empirical mechanisms and define clear
failure modes.
