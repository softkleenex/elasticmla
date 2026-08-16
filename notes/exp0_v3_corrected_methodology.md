# Exp0 v3 교정 방법론 및 결과

## 1. 목적과 해석 범위

Exp0 v3는 각 입력 위치의 MLA KV latent를 단독으로 truncation했을 때 이후 예측 loss가
얼마나 변하는지를 측정한다. v2 재검토에서 확인된 두 P1 문제, 즉 서로 정렬되지 않은
레이어에 마지막 레이어의 채널 순서를 재사용한 문제와 probe 위치마다 미래 집계 길이가
달랐던 문제를 직접 교정했다.

이 실험은 전체 시퀀스를 매번 다시 계산하는 **full-attention truncation simulation**이다.
실제 압축 KV cache를 저장하고 한 토큰씩 autoregressive decode하는 구현이 아니므로,
cache 메모리 절감률이나 decode latency 개선을 입증하는 결과로 해석하면 안 된다.

## 2. 데이터 분할과 재현성

- 체크포인트: step 3000, 6 layers, `d_c=256`, context length 256
- validation token stream의 앞쪽 calibration 영역과 뒤쪽 evaluation 영역 사이에 context
  length만큼 간격을 두어, 어떤 calibration 시퀀스의 token span도 evaluation span과
  겹치지 않게 했다.
- calibration은 seed 12341, 12342, 12343마다 16개 시퀀스를 사용했다. 세 repeat의
  saliency를 평균해 최종 레이어별 순서를 만들었다(총 48개 calibration 시퀀스).
- evaluation은 seed 23452로 24개 시퀀스를 고정하고, 각 시퀀스에서 32개 위치를
  비복원 추출했다(총 768개 probe).
- Torch와 NumPy seed를 명시적으로 고정했다. 장치는 CUDA, MPS, CPU 순서로 자동 선택하며,
  이번 실제 실행은 격리 환경의 CPU에서 수행됐다.
- calibration batch는 2, probe batch는 8로 제한했다. calibration에서는 한 번의
  forward/backward로 모든 레이어의 latent gradient를 모으고, 큰 logits는 batch마다
  해제해 peak memory를 제한했다.

## 3. 레이어별 gradient × activation 순서

각 MLA 레이어 `l`과 latent channel `c`에 대해 calibration loss의 1차 saliency를

`S[l,c] = sum |(dL/dc_kv[l,c]) * c_kv[l,c]|`

로 계산했다. 레이어마다 `S[l,:]`를 별도로 내림차순 정렬했으며, rank `r` intervention에서
레이어 `l`에는 반드시 그 레이어 자신의 상위 `r`개 채널만 남겼다. 따라서 독립적으로
학습된 레이어 사이에서 같은 channel index가 같은 의미를 가진다고 가정하지 않는다.

세 calibration repeat 사이 top-32 평균 pairwise overlap은 레이어 0부터 차례로
0.8646, 0.7812, 0.8021, 0.7917, 0.8646, 0.8125였다. 완전히 동일하지는 않으므로
calibration 표본에 따른 순위 불확실성이 남지만, v2처럼 한 레이어의 좌표계를 다른
레이어에 강제로 복사하지는 않는다.

## 4. 위치 단독 intervention과 고정 미래 horizon

평가 시퀀스 `b`의 입력 위치 `pos` 하나만 truncation하고, 같은 시퀀스의 다른 모든
위치는 full rank로 유지했다. 조작되는 KV latent는 `x_eval[b,pos]`에서 생성되므로 token
ID와 token type도 `x_eval[b,pos]`에 귀속했다. 다음-token target인 `y_eval[b,pos]`로
라벨링하지 않았다.

모든 probe는 정확히 다음 32개 loss만 사용한다.

- fixed-horizon mean: `pos+1`부터 `pos+32`까지 loss 증가의 평균
- fixed-horizon max: 같은 32개 값 중 최대 loss 증가

따라서 앞 위치의 평균이 더 긴 tail에 희석되거나, 앞 위치의 max가 더 많은 극값 기회를
얻는 v2 confound가 없다. rank grid는
`[16, 32, 48, 64, 96, 128, 160, 192, 224, 256]`, 허용 오차는 0.10 nats다.

## 5. 비단조 곡선에서의 r*와 bootstrap

loss 변화는 retained rank가 커질 때 실제로 단조 감소하지 않았다. 따라서 처음으로
epsilon을 통과한 고립된 rank를 선택하지 않고, **그 rank와 그보다 큰 모든 tested rank가
동시에 0.10 nats 이하인 최소 rank**를 r*로 정의했다(suffix-all-satisfy).

비단조 여부는 raw delta curve에서 rank를 한 단계 높였을 때 delta가 한 번이라도 증가하는지
검사했다. 별도의 양의 tolerance를 두지 않았다.

headline mean r*의 95% 신뢰구간은 개별 위치가 아니라 evaluation 시퀀스 24개를 cluster로
보고 복원 추출하는 percentile bootstrap으로 계산했다. 한 cluster가 선택되면 그
시퀀스의 32개 위치가 함께 들어간다. mean과 max에 독립적인 고정 seed를 사용해 각각
2,000회 반복했다.

## 6. 실제 실행 결과

실행 명령은 `uv run experiments/analyze_rank_variance_v3.py`였고 exit code 0으로
완료됐다.

### Fixed-horizon mean aggregation

- mean r*: **22.4375**
- sequence-cluster bootstrap 95% CI: **[21.5833, 23.3339]**
- histogram: `{16: 678, 32: 27, 48: 13, 64: 16, 96: 20, 128: 10, 160: 3, 192: 1, 224: 0, 256: 0}`
- non-monotonic: **658/768 = 85.6771%**

### Fixed-horizon max aggregation

- mean r*: **180.4167**
- sequence-cluster bootstrap 95% CI: **[175.0620, 186.0000]**
- histogram: `{16: 61, 32: 11, 48: 9, 64: 20, 96: 45, 128: 61, 160: 71, 192: 117, 224: 185, 256: 188}`
- non-monotonic: **451/768 = 58.7240%**

입력 token type별 표본 수와 평균 r*는 다음과 같다. 이는 보조적인 slice 결과이며 작은
그룹에 대한 별도 cluster CI는 계산하지 않았다.

| 입력 token type | n | mean 집계 r* | max 집계 r* |
|---|---:|---:|---:|
| capitalized | 92 | 25.9130 | 210.7826 |
| other | 552 | 21.4203 | 178.7246 |
| punct | 98 | 22.2041 | 168.0000 |
| space | 26 | 32.6154 | 155.6923 |

고정 horizon에서도 mean과 max의 결론은 크게 다르다. 평균적인 32-token 영향은 대부분
rank 16에서 epsilon을 만족하지만, 같은 구간의 최악 loss 하나까지 보호하려면 훨씬 높은
rank가 필요하다. 따라서 하나의 r* 수치만으로 배포 결론을 내리기보다 어떤 위험 함수
(평균 품질 또는 worst affected future token)를 최적화하는지 함께 명시해야 한다.

## 7. 남은 한계

1. 실제 cache-aware incremental decoding이 아니므로 cache 용량·latency 주장은 아직
   검증되지 않았다.
2. 평가 시퀀스 시작점은 서로 다르지만 token span이 일부 겹칠 수 있다. bootstrap은
   샘플된 시퀀스를 cluster로 유지했으나, 겹친 window 사이의 추가 상관까지 별도 상위
   cluster로 묶지는 않았다.
3. calibration repeat를 3회 포함했지만 headline CI는 evaluation sequence sampling
   불확실성만 나타낸다. calibration-order 불확실성을 포함한 two-way bootstrap은 아니다.
4. rank는 연속값이 아니라 10개 grid에서만 평가됐으므로 r*는 grid 해상도에 의존한다.
5. TinyStories validation과 단일 step-3000 소형 체크포인트 결과이므로 다른 데이터,
   모델 크기, 학습 단계로 일반화하려면 별도 반복 실험이 필요하다.

## 8. 산출물

- `experiments/analyze_rank_variance_v3.py`
- `experiments/exp0_rank_variance/results/exp0_v3_summary.json`
- `experiments/exp0_rank_variance/results/exp0_v3_records.json`
- `notes/exp0_v3_corrected_methodology.md`

v1/v2 스크립트와 기존 결과 파일은 provenance를 위해 수정하거나 덮어쓰지 않았다.
