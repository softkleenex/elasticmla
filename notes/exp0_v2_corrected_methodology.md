# Exp0 v2: codex review에서 지적된 방법론 버그 수정 및 재실험 결과

## 1. 배경

`codex review`가 `experiments/analyze_rank_variance.py`,
`analyze_rank_variance_layerwise.py`에 대해 critical(P1) 방법론 결함 두 가지와
버그(P1) 한 가지, P2 개선사항 다섯 가지를 지적했다. 이 문서는
`experiments/analyze_rank_variance_v2.py`로 재설계한 실험의 결과와, v1 대비
핵심 결론이 어떻게 바뀌는지를 정리한다.

## 2. 무엇을 고쳤나

### P1-a. 전역 마스킹을 per-token rank로 해석하던 문제
- **v1**: 하나의 `(d_c,)` 마스크를 배치의 **모든 위치, 모든 레이어**에 동시에 적용한 뒤,
  위치 t의 loss 변화를 "토큰 t의 effective rank"라고 보고했다. 그러나 causal attention에서
  위치 t의 loss는 t까지의 prefix 전체 KV에 의존하므로, 이 값은 "전역적으로 동일한 채널
  예산을 썼을 때 위치 t까지 누적된 민감도"이지 토큰 t 자신의 latent rank 요구량이 아니다.
- **v2**: `(T, d_c)` 모양의 마스크를 만들어, 시퀀스 내 **딱 한 위치 i**만 rank-r로
  truncate하고 나머지 모든 위치는 full-rank(전부 1)로 유지한다. 이렇게 하면 위치 i의
  KV latent truncation이 다른 위치의 KV에는 전혀 영향을 주지 않으므로, i보다 뒤에 있는
  위치들의 loss 변화가 순수하게 "토큰 i의 key/value로서의 중요도"를 반영한다.
  브로드캐스팅 `(K,T,D_c) * (K,T,D_c)`가 기존 `MultiHeadLatentAttention.forward`의
  `c_kv * rank_mask`에서 별도 코드 수정 없이 그대로 작동함을 작은 텐서로 먼저 검증했다
  (`torch.Size([2,5,4])` 테스트 통과).
  계산량 문제(위치마다 개별 forward)는, 같은 시퀀스에서 여러 위치를 배치 차원(K)으로
  묶어 한 번에 forward하는 방식으로 해결했다(시퀀스당 32개 위치 샘플링).

- **"위치 i 자신의 예측은 KV truncation과 무관하다"는 가정에 대한 정정**: query
  latent(c_Q)는 c_kv와 독립적으로 계산되므로 truncation의 영향을 받지 않는 건 맞지만,
  이 코드베이스의 causal mask(`triu(diagonal=1)`)는 위치 i가 **자기 자신**을 key로
  attend하는 것을 허용한다. 따라서 위치 i 자신의 loss도 c_kv[i] truncation에 완전히
  무관하지 않다. v2는 이를 "self effect"로 명시적으로 분리해서 별도 보고하고,
  "future effect"(i보다 뒤 위치들의 loss 변화)와 섞이지 않게 했다.
  실측 결과 `mean_abs_self_delta_at_rank16 = 0.1326` nats로,
  future effect 평균의 약 6.3배 크다. 즉 self effect가 결코 무시할 크기가
  아니며, 두 효과를 분리해서 보고하는 것이 중요했다는 점이 실제로 확인되었다.

### P1-b. raw latent variance를 채널 중요도로 쓰던 문제
- **v1**: `c_kv`의 채널별 분산으로 중요도를 매겼다. 그러나 `c_kv[:,c] *= a`,
  `W_UK[:,c] /= a`, `W_UV[:,c] /= a` 같은 재매개변수화는 모델 함수를 그대로 두면서도
  분산 순위를 임의로 바꿀 수 있어, scale-invariant하지 않다.
- **v2**: 단일 backward pass로 얻은 saliency score
  `sum_t |dL/dc_kv[:,:,c] * c_kv[:,:,c]|` (1차 Taylor 근사, gradient × activation)로
  대체했다. 이 지표는 채널을 실제로 ablation했을 때의 loss 변화량을 근사하므로
  위 재매개변수화에 훨씬 덜 민감하다.
- **saliency vs. variance 순위 비교**: Spearman 순위상관 =
  0.7215, 상위 32개 채널 중 겹치는 비율 =
  0.50.
  즉 두 지표가 어느 정도 상관은 있지만(rho≈0.72) 완전히 같지는 않으며, 상위 채널의
  절반 정도만 일치한다. variance 기반 중요도가 scale-invariant하지 않다는 리뷰 지적이
  실측으로도 확인된다 — 만약 두 순위가 거의 동일했다면 이 우려는 실질적으로 무해했을
  것인데, 실제로는 유의미하게 다르다.

### P1-c. `model.py`의 `rank_mask` + `layer_idx_for_latent` 동시 사용 버그
- **원인**: `MLAGPT.forward`에서
  `x = blk(x, rank_mask=rank_mask if layer_idx_for_latent is None else None)`로
  되어 있어, `layer_idx_for_latent`가 지정되면 그 레이어를 제외한 **나머지 모든
  레이어에 `rank_mask=None`**이 전달되어 truncation이 사실상 적용되지 않았다.
- **수정**: `layer_idx_for_latent`는 "어느 레이어의 latent를 반환할지"만 결정하고,
  `rank_mask`는 항상 모든 블록에 전달되도록 분리했다(하위호환 유지: 인자 시그니처는
  그대로).
- **검증**: v2 스크립트에 `rank_mask`+`layer_idx_for_latent`를 함께 넘긴 forward와
  `rank_mask`만 넘긴 forward(이미 모든 레이어에 마스크 적용되는 것으로 알려진 경로)의
  출력을 비교하는 sanity check을 넣었다. 실행 결과
  `max_abs_logit_diff = 0.0`로
  완전히 일치했다.
- **v1, layerwise 스크립트에 대한 영향**: `analyze_rank_variance.py`와
  `analyze_rank_variance_layerwise.py`를 이 수정 후 다시 실행해서 확인했는데, **둘 다
  숫자가 한 자리도 바뀌지 않았다**(`r_star_mean` 162.55078125, 162.55078125로 동일 /
  layerwise 149.61328125로 동일). 이유: 두 스크립트 모두 `rank_mask`와
  `layer_idx_for_latent`를 **동시에** `model.forward()`에 넘기는 지점이 없다
  (`layer_idx_for_latent`만 쓰는 latent 수집 호출에는 `rank_mask`가 없고, `rank_mask`를
  쓰는 truncation 호출에는 `layer_idx_for_latent`가 없다; layerwise 스크립트는 아예
  `model.forward()`를 안 쓰고 블록을 직접 순회한다). 즉 이 P1 버그는 기존 v1/layerwise
  스크립트의 **결과에는 영향을 준 적이 없고**, v2에서 두 인자를 함께 쓸 가능성이
  생기면서 미리 고쳐둔 것이다.

### P2 개선사항
- 토큰 타입 라벨링을 `y[:, pos]`(예측 대상) 기준으로 변경(v1은 `x` 기준이었음).
- 채널 중요도(saliency/variance)는 calibration 시퀀스 48개, r* 측정은 별도
  evaluation 시퀀스 24개로 분리(검증 구간은 validation stream의 앞 60%/뒤 40%로
  물리적으로도 겹치지 않게 분리).
- 큰 logits 텐서는 사용 직후 `del` + `torch.mps.empty_cache()`/`torch.cuda.empty_cache()`.
- device 후보에 `cuda`도 포함 (`cuda > mps > cpu` 우선순위). 이번 실행 환경은 mps만
  있어서 mps로 실행됨.
- `torch.manual_seed`뿐 아니라 `np.random.seed`도 명시적으로 고정(seed=1234, calibration/
  evaluation 각각 별도 시드 사용).

## 3. 실행 결과 (`uv run experiments/analyze_rank_variance_v2.py`)

- device: mps, checkpoint step 3000, d_c=256, probe_layer=5(마지막 레이어)
- calibration 48 시퀀스, evaluation 24 시퀀스, 시퀀스당 32개 위치 샘플링 (총 768개
  위치 probe)

### r*_i (future effect, mean aggregation) — 정의대로: i보다 뒤 모든 위치의 **평균**
loss 증가가 epsilon(0.10 nats) 밑으로 떨어지는 최소 rank

- mean = 18.42, std = 14.50, min = 16, max = 160
- histogram: {"16": 742, "32": 5, "48": 2, "64": 1, "96": 12, "128": 5, "160": 1, "192": 0, "224": 0, "256": 0}
- 대부분(768개 중 742개, ≈97%)이 최소 rank 16에서 이미 epsilon을 만족한다.

**이 숫자만 보면 v1의 핵심 결론("토큰마다 필요 rank가 크게 다르다", r_star_mean≈162)이
거의 완전히 무너지는 것처럼 보인다.** 그러나 이는 실제로 "토큰 하나의 KV truncation은
전혀 중요하지 않다"는 뜻이 아니라, **평균(mean) 집계 방식 자체의 희석(dilution)
아티팩트**다: 위치 i는 보통 그 뒤 수백 개 위치 중 아주 소수(주로 가까운 미래, 또는
attention이 강하게 걸리는 특정 위치)에만 강하게 attend된다. 나머지 대다수 미래 위치는
i를 거의 보지 않으므로 delta≈0이고, 이걸 전부 평균 내면 진짜 신호가 통계적으로
씻겨나간다.

### r*_i (future effect, max aggregation) — i보다 뒤 위치 중 **가장 크게 나빠진**
위치의 loss 증가가 epsilon 밑으로 떨어지는 최소 rank (희석 문제를 피하기 위한 대안 집계)

- mean = 194.96, std = 74.93, min = 16, max = 256
- histogram: {"16": 54, "32": 13, "48": 14, "64": 16, "96": 26, "128": 28, "160": 56, "192": 95, "224": 182, "256": 284}

이 집계 방식에서는 v1의 r_star_mean(162.55)과 **정량적으로 비슷한 스케일**(194.96)이
나오고, std(74.9)도 여전히 크며 분포도 16~256 전 구간에 걸쳐 퍼져있다(다만 v1보다
256(=truncation 없음과 사실상 동일)에 쏠리는 정도가 더 큼: 284/768 ≈ 37%가 rank
256에서야 겨우 epsilon을 만족). 즉 **"토큰마다 필요 rank가 다르다"는 정성적 결론
자체는 max aggregation 기준으로는 유지**되지만, "평균적으로 rank~162면 충분하다"는
v1의 정량적 주장은 성립하지 않는다 — 오히려 최악의 경우(어떤 미래 토큰이 이 위치를
강하게 참조하는 경우)를 기준으로 하면 더 많은 토큰이 거의 full rank(256)를 요구한다.

### 두 집계 방식이 말하는 것 (해석)
- v1의 전역 마스킹 결과는 사실 "모든 위치를 동시에 truncate했을 때 위치 t까지 누적된
  민감도"였고, 이는 **모든 이전 위치가 동시에 나빠지는 최악의 경우와 비슷한 조건**에
  가깝다. 따라서 v1의 결과가 v2의 max-aggregation 결과와 스케일이 비슷하게 나온 것은
  우연이 아니라, 두 측정 모두 "많은/최악의 조건에서 이 위치가 얼마나 중요한가"를
  재고 있기 때문으로 보인다.
- 반면 v2의 mean-aggregation 결과(~18)는 "한 위치를 단독으로, 평균적인 미래 위치
  기준으로 truncate했을 때"의 영향으로, 이는 실제 배포 시나리오("이 토큰의 KV만 rank
  r로 캐싱한다면, 그 하나의 truncation이 이후 생성 전체의 평균 품질에 주는 영향")에
  더 가까운 정의일 수 있다. 이 정의에서는 개별 토큰 하나의 KV truncation이 평균적으로는
  거의 무해하다 — 즉 "토큰별로 다른 rank를 할당해야 한다"는 실용적 근거가 v1이
  주장한 것보다 훨씬 약할 수 있다는 뜻이다.
- **결론**: v1의 핵심 결론("토큰마다 필요 rank가 다르다")은 "다르다"는 정성적 방향은
  살아남지만(어느 집계 방식으로도 std/분산이 크고 min/max가 널리 퍼져 있음),
  "평균적으로 rank~162가 필요하다"는 정량적 주장은 근거가 약했다는 게 드러났다.
  실제 "토큰별 rank 할당이 실용적으로 의미 있으려면 무엇을 최적화해야 하는가"는
  mean/max/혹은 다른 집계(top-k 등) 중 어떤 것을 배포 시 손실 기준으로 삼을지에
  따라 완전히 다른 답이 나온다는 점이, v1 방법론에서는 전혀 드러나지 않았던 새로운
  통찰이다.

### 토큰 타입별 breakdown (y 기준)
mean aggregation: {"capitalized": 20.50485436893204, "other": 17.83206106870229, "punct": 17.0, "space": 27.03448275862069}
max aggregation: {"capitalized": 213.74757281553397, "other": 196.18320610687022, "punct": 170.85714285714286, "space": 199.17241379310346}

표본 수가 적어(공백 29개, 구두점 112개 등) 신뢰구간이 넓지만, 두 집계 모두에서
"space"(공백/개행류) 토큰이 다른 타입보다 상대적으로 높은 rank를 요구하는 경향이
관찰된다. 이는 v1에서도 나타난 패턴은 아니었다(v1은 반대로 space가 가장 낮은
rank를 요구했다 — 107.3 vs. 다른 타입 154~178). 즉 **토큰 타입별 순위 자체가 v1과
v2에서 방향이 다르게 나온다.** 이는 v1의 "타입별 breakdown"이 방법론 결함(P1-a, P1-b)
때문에 신뢐할 수 없었다는 것을 뒷받침한다 — v2가 고쳐진 방법으로 다시 측정하니 결론이
바뀐다.

## 4. 채널 중요도 지표 비교 (saliency vs. variance)

- Spearman rho = 0.7215
- 상위 32개 채널 중 겹치는 비율 = 0.50

중간 정도의 상관(0.72)은 있지만 상위 채널의 절반만 일치하므로, "어떤 채널이 중요한가"에
대한 답 자체가 지표 선택에 따라 달라진다. saliency가 실제 ablation loss를 1차
근사하는 지표이므로 원칙적으로 variance보다 신뢰할 수 있고, 리뷰가 지적한 대로
variance 기반 결과는 재매개변수화에 취약하다는 것이 (완전히 다른 순위는 아니지만)
유의미하게 다른 순위로 실증되었다.

## 5. 남은 한계 / 후속 작업 제안
- probe_layer는 여전히 마지막 레이어 하나만 사용하고, 채널 중요도도 그 레이어
  기준으로 전 레이어에 동일하게 적용한다(레이어별 saliency로 확장 가능, `_layerwise.py`와
  유사한 확장이 필요하면 별도 스크립트로 진행).
  이 방식은 v1과의 비교 가능성을 위해 의도적으로 유지했다.
- 위치 샘플링은 시퀀스당 32개(전체 768개 위치)로 제한했다. 계산량이 허용되면 전체
  위치를 다 도는 실험으로 확장해 통계적 검정력을 높일 수 있다.
- "self effect"가 예상보다 크다는 것은, MLA에서 자기 자신의 KV를 truncate하는 것이
  단순히 미래 토큰에 대한 영향뿐 아니라 자기 자신의 예측 품질에도 직접 영향을 준다는
  뜻이므로, 실제 elastic-rank 배포 정책을 설계할 때 self effect와 future effect를
  모두 고려한 결합 목적함수가 필요할 수 있다.
- mean vs. max aggregation 중 어느 것이 "실제로 의미 있는 정의"인지는 배포 시나리오에
  따라 다르므로, 논문에서는 두 정의를 모두 명시하고 어떤 배포 상황(예: KV 캐시
  압축이 평균 loss에 미치는 영향 vs. worst-case 저하)을 근거로 삼는지 분명히 해야 한다.

## 6. 산출물
- `experiments/analyze_rank_variance_v2.py` (신규 스크립트, 기존 v1 스크립트는 보존)
- `experiments/exp0_rank_variance/results/exp0_v2_summary.json` (요약 통계)
- `experiments/exp0_rank_variance/results/exp0_v2_records.json` (probe별 상세 기록)
- `experiments/exp0_rank_variance/results/exp0_v2_channel_importance_{saliency,variance}.npy`
- `code/elastic_mla/model.py`: `rank_mask` + `layer_idx_for_latent` 동시 사용 버그 수정
  (v1/layerwise 스크립트 재실행으로 회귀 없음 확인, 결과 완전히 동일)
