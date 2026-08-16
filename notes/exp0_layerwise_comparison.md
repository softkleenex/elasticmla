# Exp0: 균일(uniform) truncation vs 레이어별(layer-wise) truncation 비교

## 배경
Exp0 최초 분석(`analyze_rank_variance.py`)은 마지막 레이어에서만 채널 중요도를 계산해서
전 레이어에 동일하게 적용했음 (uniform). CARE 논문이 레이어마다 채널 중요도가 다르다고
보였기 때문에, 각 레이어에서 독립적으로 채널 중요도를 계산해 레이어별로 다르게 truncate하는
`analyze_rank_variance_layerwise.py`를 추가로 작성/검증함.

(참고: 초기에 GPT-OSS-120B 서브에이전트에게 이 작업을 위임했는데, 스크립트만 작성하고
실제 실행 검증 없이 완료 보고만 해서 버그(`MLAGPT.forward()`에 없는 `return_latent` 인자를
전달)가 있었음. 직접 패치해서 실행 검증 완료.)

## 결과 비교

| 지표 | 균일(last-layer) | 레이어별(layer-wise) |
|---|---|---|
| r*_t 평균 | 162.6 | **149.6** |
| r*_t 표준편차 | 88.8 | 88.1 |
| rank=16으로 충분한 토큰 비율 | 18.2% | **19.0%** |
| rank=256 필요한 토큰 비율 | 26.4% | **20.8%** |
| space 평균 r* | 107.3 | 100.3 |
| capitalized 평균 r* | 154.9 | 144.9 |
| other 평균 r* | 164.0 | 151.4 |
| punct 평균 r* | 178.4 | 160.0 |

## 해석
1. **레이어별 채널 중요도를 쓰면 평균적으로 더 적은 rank로도 같은 loss degradation 기준(ε=0.10)을
   만족한다** (평균 162.6 → 149.6, 약 8% 절감). 이는 "레이어마다 중요한 latent 채널이 다르다"는
   가설을 지지하며, CARE의 layer-heterogeneity 주장과 일관됨.
2. **Bimodality(양극단 쏠림)는 유지되지만 다소 완화됨** — rank=256 필요 토큰 비율이 26.4%→20.8%로
   줄어듦. 즉 레이어별로 정확히 중요한 채널을 골라내면, "무조건 풀랭크가 필요해 보였던" 토큰 중
   일부는 사실 레이어마다 다른 채널 조합만 맞으면 더 적은 rank로도 충분했다는 뜻.
3. **토큰 타입별 순위(space < capitalized < other < punct)는 두 방식 모두 동일하게 유지됨** —
   이 정성적 패턴은 truncation 방법론에 견고(robust)한 것으로 보이며, 논문에서 주장할 수 있는
   더 신뢰도 높은 발견.
4. 표준편차는 거의 그대로(88.8→88.1) — 토큰 간 variance 자체는 truncation 방법과 무관하게
   존재하는 진짜 신호로 보임 (falsify 대상이었던 "variance가 클까?"에 대한 답은 방법론 안 바뀜).

## 결론
- **핵심 가설("토큰마다 필요한 latent capacity가 다르다")은 레이어별 정밀 분석에서도 재확인됨.**
- 레이어별 truncation이 더 효율적이라는 사실 자체도 흥미로운 부차 발견 — Elastic MLA 설계 시
  "토큰 단위"뿐 아니라 "레이어 단위"까지 함께 adaptive하게 가는 (2축) 설계가 정당화됨.
- 다음 단계(Exp1, Kaggle 114M)에서는 레이어별 truncation을 기본 방법론으로 채택해야 함.

## 한계 (유지)
- 여전히 단일 시드/스케일(30M)/도메인(TinyStories).
- 채널 중요도 = 단순 분산 기준 (SVD/covariance 기반 아님).
- epsilon=0.10 임의적 — sensitivity analysis 필요.
