# Packed Tiered ElasticMLA prototype

## 구현
- 토큰별 rank에 맞춰 saliency-ordered latent 값만 1D buffer에 실제 저장.
- int32 offsets, int16 ranks, int16 channel-order metadata, shared post-RoPE key 포함.
- cache 생성 때 사용한 channel order를 저장하고 이후 호출에서 불일치 거부.
- batch>1 append/unpack, mixed tier, autocast를 지원.
- `TieredRankRouter`와 `ElasticMLAGPT` wrapper 제공. 기본 tier 후보는
  `{16, 64, 160, 256}`이며 base LM 동결 후 supervised oracle imitation이 가능함.

## 실제 30.6M 체크포인트 저장량 (예시 tier mix)
예시 정책은 70% rank16, 각 10% rank64/rank160/rank256이며 **학습된 router 결과가 아님**.

- 평균 rank: 56.5
- packed vs 동일 dense-mask logits max diff: 0
- packed cache: 141,336 bytes
- fixed-width MLA cache: 442,368 bytes
- conservative standard-MHA theoretical cache: 1,179,648 bytes
- packed / fixed MLA: 0.3195
- packed / standard MHA: 0.1198

즉 이 예시 mix에서는 persistent payload가 fixed MLA 대비 약 68.1%, standard MHA 대비
약 88.0% 감소한다. 단, Python unpack은 전체 latent/K/V를 임시 복원하므로 peak memory와
latency 절감 주장이 아니라 persistent storage 절감만 검증한 수치다.

## 검증
전체 test suite 23개 통과. gpt-5.6-sol Codex 재리뷰 결과 남은 P0/P1 없음,
commit-safe 판정.

## 다음 단계
v3 oracle label을 tier `{16,64,160,256}`로 양자화해 router를 학습하고, 독립 validation에서
(1) router 정확도, (2) 전체 시퀀스 동시 압축 PPL, (3) 실제 packed bytes의 Pareto를 평가한다.
