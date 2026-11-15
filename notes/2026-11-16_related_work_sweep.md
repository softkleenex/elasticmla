# 2026-11-16 - Related-work sweep (two subagents, every entry checked against its arXiv record)

## Positioning that follows

- "Mean hides the tail" is not new. Prior work makes the point at several levels:
  - Dutta 2024 (2407.09141): flips cancel in aggregate accuracy.
  - DefensiveKV (2510.13334): max instead of mean over future queries for eviction importance.
  - Fang 2024 (2410.23771) and Murshed 2026 (2610.14475): position averaging hides damage.
  - Pitfalls (2510.00231): some instructions degrade far faster.
  - Haverbeck 2026 (2607.01520): minimax risk theory.
- Per-token nested-prefix truncation is not new either. DynaKV (2603.04411) learns per-token truncation of
  a PCA-ordered prefix, outside MLA. It is fine-tuned on 128M tokens and reports only mean and benchmark
  scores, with no tail measure.
- MLA latent quantization at 2-8 bits is an active area: SnapMLA 2602.10718, QuantMLA 2609.36760,
  RoLaQ 2610.20686 and RAQ-MLA 2610.33625. The FP8 latent cache of DeepSeek-V3.2 (2512.02556) has its
  layout documented in the FlashMLA repository. So "int8 dominates the router" is expected.
- No prior work found for:
  - the size of the gap between mean-safe and worst-offset-safe rates per token (~10x);
  - its order-statistic and equal-norm controls;
  - the joint test (a mean-safe allocation passes a mean check while 29% of positions exceed epsilon);
  - CVaR/worst-case criteria for rate allocation itself.

  Frame the novelty as that quantification and mechanism.

## Verified papers to add

| arXiv | Short title | First author, year | Where |
|---|---|---|---|
| 2603.04411 | DynaKV, token-wise adaptive KV compression | Lu 2026 | 2.2: closest prior; reword our contribution |
| 2410.14731 | MatryoshkaKV | Lin 2024 | 2.2 nested projections |
| 2503.11132 | X-EcoMLA | Li 2025 | 2.1 conversion list |
| 2610.17202 | MLAfication | Wang 2026 | 2.1 conversion list |
| 2602.10718 | SnapMLA | Zhang 2026 | 6.3 MLA quantization |
| 2609.36760 | QuantMLA | Su 2026 | 6.3 |
| 2610.20686 | RoLaQ | Ji 2026 | 6.3 (DeepSeek-V2-Lite) |
| 2610.33625 | RAQ-MLA | Chen 2026 | 6.3 |
| 2512.02556 | DeepSeek-V3.2 (FP8 KV; layout in FlashMLA repo) | DeepSeek-AI 2025 | 6.3 |
| 2503.18893 | xKV | Chang 2025 | 2.2 low rank |
| 2407.21018 | ThinK | Xu 2024 | 2.2 low rank |
| 2511.01815 | KVTC | Staniszewski 2025 | 2.2 low rank / quantization |
| 2607.12550 | JoLT, joint rank and bit allocation | Krishnan 2026 | 6.3 |
| 2402.18096 | MiKV, No Token Left Behind | Yang 2024 | 2.2 per-token mixed precision |
| 2605.06675 | RateQuant | Zuo 2026 | 3 rate allocation |
| 2606.26875 | InfoKV, forward influence | Kai 2026 | 2.2 / 5.3 |
| 2407.09141 | Accuracy is Not All You Need (flips, KL) | Dutta 2024 | 1, 5.3, R6 |
| 2310.01382 | Compressing LLMs: The Truth is Rarely Pure | Jaiswal 2023 | 1 |
| 2410.23771 | What is Wrong with Perplexity for Long-context LM | Fang 2024 | 1, 2.2 |
| 2610.14475 | EdgeLLM-Nano: auditing key-cache compression | Murshed 2026 | 1, 2.2 |
| 2606.19558 | Displacement Is Not Direction (KL limits) | Nikolic 2026 | 7 |
| 2609.16617 | Divergence Timing under KV-Cache Eviction | Luo 2026 | 2.2 |
| 2407.01527 | KV Cache Compression, But What Must We Give in Return? | Yuan 2024 | 1 |
| 2510.00231 | The Pitfalls of KV Cache Compression | Chen 2025 | 1, 2.2 |
| 2503.24000 | Rethinking KV Cache Compression for LLM Serving | Gao 2025 | 2.2 |
| 2510.13334 | DefensiveKV | Feng 2025 | 2.2: closest on mean vs max over future queries |
| 2502.03805 | CriticalKV | Feng 2025 | 2.2 |
| 2607.01520 | The risk of KV cache compression | Haverbeck 2026 | 2.2 |
| 2410.19258 | HeadKV, Not All Heads Matter | Fu 2024 | 2.2 next to Ada-KV |
| 2404.06654 | RULER | Hsieh 2024 | 2.2 evaluation |
| 2602.03203 | ForesightKV | Dong 2026 | 2.2 |
| 2007.01162 | Tilted ERM | Li 2020 | 3 next to CVaR |

## Reviewer expectations noted

- At least one downstream or long-context task on DeepSeek-V2-Lite.
- A fixed-rank baseline in the style of DynaKV or MatryoshkaKV.
- Per-token KL and flips (R6), and tail columns for the joint test (CVaR 0.9/0.99, flips, KL).
