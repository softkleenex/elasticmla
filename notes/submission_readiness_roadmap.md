# Submission-readiness roadmap

## Honest current tier

The frozen two-scale result is suitable for a rigorous efficient-ML workshop or short empirical
paper, but not yet a convincing main-track MLSys/NeurIPS/ICLR submission. The main blockers are
realistic scale/domain coverage, stronger baselines, multiple training seeds, and measured
end-to-end efficiency. Provenance, audits, and matched-byte confirmation are current strengths.

## Claim architecture

1. **Theory:** nested latent prefixes form a basis-dependent operational rate code; cache tensor
   bytes are affine in token rates; upper-tail future-loss risks induce ordered suffix-safe rates;
   contextual routing is a constrained joint rate-distortion problem.
2. **Mechanism:** quantify the tail-capacity premium, positive-part concentration, horizon law,
   marginal tier value, and interaction residuals.
3. **Empirics:** contextual allocation must beat random, position, lexical, frequency, and tuned
   static controls at equal realized bytes across seeds and domains.
4. **Systems:** only claim serving benefit after a grouped/fused packed kernel shows peak-memory and
   throughput improvements against optimized MLA/GQA baselines.

## Required work packages

### A. Theory/mechanism (current priority)
- Two-scale upper-tail risk-capacity and horizon spectra from raw per-offset deltas.
- Positive-part/cancellation diagnostics.
- Contextual marginal-value and pair-interaction audits.
- Basis-dependence ablation: raw coordinate order vs PCA/SVD or learned orthogonal nested basis.

### B. Stronger controls
- Exact-histogram position-only, lexical-identity, token-frequency, and token-type placements.
- Global-budget fixed-width and learned-static Pareto curves, not only router-conditioned budgets.
- Full lambda frontier on development data; untouched confirmation only for the final frozen choice.

### C. Replication
- At least three router seeds per scale and a substantially larger calibration/validation split.
- A second text domain and longer contexts.
- Prefer a credible native-MLA or converted >=1B checkpoint; otherwise target a workshop/TMLR-style
  mechanistic paper and state the scale boundary prominently.

### D. Systems
- Direct projection from packed prefixes or grouped-tier decode without dense latent reconstruction.
- CUDA peak allocated/reserved memory, decode latency, tokens/s, batch throughput, and break-even
  context length.
- Compare fixed MLA, grouped-tier ElasticMLA, MHA/GQA where shape-compatible, and report router
  overhead.

### E. Submission package
- Expand related work and explicit novelty table.
- Add theorem proofs, assumption/claim table, ablations, failure cases, and reproducibility appendix.
- Produce LaTeX, bibliography, artifact README, exact commands, environment lock, and anonymous
  archive.
