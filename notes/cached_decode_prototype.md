# Compressed MLA cached decoding prototype

## 구현 범위
- `MultiHeadLatentAttention.forward_cached`: 과거 토큰마다 복원된 head-wise K/V 대신
  shared `c_kv`와 decoupled post-RoPE key만 영구 저장.
- token-by-token decode 및 chunked prefill 지원.
- 레이어별 compressed cache, cache byte 측정, 간단한 cached generation 지원.

## 정확성 검증
- 10개 unit test 통과: full/token/chunk 동치, cache shape/bytes, layer-specific rank mask,
  기존 cache 불변성, dtype/autocast, mixed layer length 거부, generation off-by-one.
- 실제 Exp0 30.6M checkpoint (MPS), 길이 16:
  - max absolute logits difference: `1.1444e-05`
  - MLA persistent cache: `110,592` bytes
  - 보수적인 standard-MHA theoretical cache: `294,912` bytes
  - ratio: `0.375` (62.5% persistent cache reduction for this config)

## 속도 수치의 제한
`benchmark_cached_decode.py`의 timing baseline은 **naive full-prefix recomputation**이다.
최적화된 MHA KV-cache baseline이 아니므로 MLA-vs-MHA speedup으로 인용하면 안 된다.
현재 correctness-first 구현은 매 decode step마다 전체 latent history로부터 content K/V를
재구성하므로 persistent cache는 작지만 임시 K/V 메모리와 decode compute는 context length에
따라 증가한다. absorbed projection/custom kernel 또는 packed elastic cache는 아직 미구현이다.

## 다음 구현 과제
1. optimized MHA/GQA cached baseline과 공정 비교.
2. absorbed MLA decode 또는 FlashMLA 계열 kernel 연결.
3. ~~token별 variable-width packed storage~~ — `elastic_cache.py`로 correctness prototype 구현 완료.
4. 학습된 tier router의 quality–memory Pareto 평가.
