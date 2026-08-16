# Experiment 0 결과 및 해석 (2026-08-16)

## 세팅
- 모델: MLAGPT 30.6M params, d_model=384, n_layers=6, n_heads=6, d_c=256 (KV latent dim)
- 데이터: TinyStories 4만 문서, 878만 train 토큰, 3000 step 학습 (M4 Pro, MPS)
- 최종 val_loss: 1.96
- 분석: 64개 검증 시퀀스 x 256 토큰 = 16,384 토큰에 대해, latent channel을
  분산 기준으로 정렬 후 rank grid {16..256}로 전 레이어 균일 truncate,
  토큰별 next-token loss degradation < 0.10 nats를 만족하는 최소 rank r*_t 측정.

## 핵심 수치
- r*_t 평균 162.6 / 표준편차 88.8 (d_c=256 대비 변동계수 ~0.55 — 상당히 큼)
- 분포가 **양극단에 몰리는 bimodal 형태**: rank=16으로 충분한 토큰이 18.2%,
  반대로 rank=256(=full) 필요한 토큰이 26.4%. 중간 구간은 상대적으로 얇음.
- 토큰 타입별 평균 r*_t:
  - space/개행류: 107.3 (가장 낮음, 예측대로)
  - capitalized: 154.9
  - 일반 단어(other): 164.0
  - **구두점(punct): 178.4 (직관과 반대로 가장 높음)**

## 해석
1. **1차 가설("토큰마다 필요한 latent capacity가 다르다")은 이 파일럿에서 지지된다.**
   분산이 크고, 토큰 타입에 따라 유의미하게 갈린다.
2. **분포가 연속적(continuum)이라기보다 bimodal에 가깝다** — 이건 예상 밖의 발견이고
   오히려 논문 스토리에 유리할 수 있다. "모든 토큰에 각기 다른 rank를 정교하게
   배정"하기보다 "적음/많음 2~3단계(tiered) elastic 구조"만으로도 충분할 가능성을 시사.
   → Elastic MLA를 nested rank의 continuous predictor보다 **discrete tier gating**
     (예: {64, 256} 2단계)로 단순화하는 설계가 오히려 더 맞을 수 있음.
3. **구두점이 제일 높은 rank를 요구하는 건 흥미로운 반직관적 결과.** 가설: 마침표(.) vs
   느낌표(!) vs 물음표(?) 중 어떤 걸 선택할지는 전체 문맥(서술/감탄/의문)에 의존하므로
   punctuation token 자체의 예측이 narrative-level 문맥 정보를 많이 요구하기 때문일 수
   있음. Through-the-Bottleneck 논문의 "content is preserved, position is discarded"
   결과와 연결지어 스토리를 만들 수 있음.

## 이 파일럿의 한계 (반드시 명시)
- 채널 중요도를 **마지막 레이어에서만** 계산해서 전 레이어에 동일하게 적용 —
  레이어마다 "중요한 채널"이 다를 수 있다는 CARE의 layer-heterogeneity 결과와
  상충 가능. Exp1에서는 레이어별 별도 중요도 계산 필요.
- 단일 시드, 단일 스케일(30M), 단일 도메인(TinyStories) — Through-the-Bottleneck
  논문도 스스로 지적한 replication 문제를 그대로 안고 있음.
- 채널 중요도 = 단순 분산(variance) 기준. SVD/covariance 기반이 더 엄밀함 (CARE 방식).
- 토큰 타입 분류가 매우 조야함(공백/구두점/대문자/기타 4종) — 진짜 논문에서는
  POS tagging이나 entity recognition으로 정교화 필요.
- epsilon=0.10 nats 임계값이 임의적 — sensitivity analysis 필요.

## 다음 단계 (Exp1, Kaggle CLI로 진행)
1. 114M 규모로 스케일업 (`code/kaggle_notebook/train_kaggle.py`, 이미 준비됨)
2. 레이어별 채널 중요도 별도 계산 (uniform 대신 layer-wise)
3. 여러 시드로 반복해서 bimodality가 재현되는지 확인
4. POS 기반 정밀 token-type 분석
5. 결과가 유지되면 → Elastic/tiered MLA 아키텍처 설계 및 학습으로 진행
