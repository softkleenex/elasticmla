# MLA Literature Landscape (수집: arXiv API, 기준일 2026-08-15)

## 1. 핵심/원조
- **DeepSeek-V2** (2405.04434) — MLA 최초 제안. low-rank KV latent + decoupled RoPE.
- **DeepSeek-V3** (2412.19437) — 671B 규모에서 MLA 재확인, MoE와 결합.
- **Insights into DeepSeek-V3** (2505.09343) — scaling/hardware 회고.

## 2. Conversion (MHA/GQA → MLA)
- **MHA2MLA** (2502.14837, 2025-02) — 최초 data-efficient MHA→MLA fine-tuning. Llama2-7B KV cache -92.19%, LongBench -0.5%.
- **CARE** (2603.17946, 2026-03) — activation covariance 기반 decomposition + **layer별 non-uniform rank allocation**. (내 layer-adaptive 아이디어 이미 선점)
- **MHA2MLA-VLM** (2601.11464, 2026-01) — VLM으로 확장.
- **Whisper-MLA** (2603.00563, 2026-02) — ASR로 확장.
- **Beyond KV Reconstruction** (2607.27269, 2026-07) — MLA draft model의 speculative decoding, functional reconstruction.

## 3. 시스템/하드웨어
- **Hardware-Centric Analysis of MLA** (2506.02523, 2025-06) — 최초 하드웨어 관점 분석 (reuse vs recompute latent).
- **EG-MLA** (2509.16686, 2025-09) — token-specific embedding-gating로 추가 압축.
- (SnapMLA 관련은 arXiv API에서 직접 미확인, 재검증 필요)

## 4. 이론/분석 (내 아이디어와 가장 가까움)
- **Through the Bottleneck** (2607.23054, 2026-07-25) — **최초 MLA mechanistic interpretability**. 114M 모델, single seed/scale.
  latent bottleneck이 content는 보존하고 position은 버림, effective rank 평균 46%만 사용 (128 중 58.8).
  ⚠️ 저자 스스로 "scale/domain/seed 다양화한 replication 필요"라고 명시.
- **Random Matrix Theory Perspective on MLA** (2507.09394, 2025-07) — MP-law 기반 학습 동역학 분석.

## 5. Adaptive-rank KV compression (직접 경쟁자 — 반드시 차별화 필요)
- **STAR-KV** (2606.08382, 2026-06) — differentiable thresholding, **head-level + block-level** adaptive rank.
  MLA 전용 아님(범용 low-rank KV), block=고정 크기 청크 단위. 진짜 per-token continuous adaptivity는 아님.
- **DynaCalKV** (2607.24331, 2026-07-27, 가장 최근!) — CKA 기반 head grouping + rank budget 할당.
  **오프라인/정적** 할당, K/V 다르게 취급. MHA/GQA에 특화, GQA에는 보수적으로 적용 권고.
  MLA 아키텍처(decoupled RoPE) 자체를 다루지 않음.

## 6. 결론: 남은 gap
아직 아무도 명시적으로 하지 않은 것:
  (a) **DeepSeek 스타일 MLA (decoupled RoPE 포함) 아키텍처**에서
  (b) **런타임에 토큰 단위(over block/head 아님) latent rank를 content-adaptive하게 결정**하고
  (c) nested/elastic 표현으로 inference 시 truncation만으로 rank를 조절하는 것.
STAR-KV/DynaCalKV는 (a)를 안 하고 block/head 레벨에 머무름. CARE는 (a)를 하지만 layer-level 정적 할당.
Through-the-Bottleneck은 분석만 하고 method를 제안하지 않음 (그리고 replication이 필요하다고 스스로 인정).

=> ElasticMLA 포지셔닝: "토큰별 latent capacity가 실제로 크게 다르다"는 것을
   Through-the-Bottleneck의 분석을 확장해 직접 검증하고,
   STAR-KV/DynaCalKV보다 더 미세한(token-level, MLA-native) adaptive mechanism을 제안.

⚠️ 이 리스트는 arXiv API 검색 기반이며 완전한 systematic review는 아님.
   투고 전 Semantic Scholar/Google Scholar cross-check와 최신 재검색 필수 (특히 2026-07~08 신규 논문).
