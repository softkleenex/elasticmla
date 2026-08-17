> **철회 상태 (2026-08-17):** 이 router는 철회된 v3 horizon label로 학습되었으므로 정량 수치를 논문 근거로 사용하지 않는다. packed-cache 구현 correctness와 별개이며, v4 label로 재학습해야 한다.

# Global lexical tier-router PoC

## 설계 정정
Exp0-v3 oracle은 모든 레이어를 같은 rank로 동시에 개입한 **global token-difficulty label**이다.
따라서 초기의 레이어별 router/레이어별 정확도 해석을 폐기하고, 하나의 global router가
선택한 tier를 모든 MLA 레이어에 공유하도록 재설계했다.

Router feature는 첫 attention 이전의 `LayerNorm(token embedding)`이다. 따라서 현재 PoC는
**문맥적 routing이 아니라 lexical token-identity routing**이다. 동일 token ID는 문맥과 위치에
관계없이 동일 결정을 받는다.

## 누수 방지
256-token evaluation window가 겹치는 seq들은 connected component로 묶은 뒤 component
단위로 train/val/test를 분할했다. mean/max 정책은 동일 split임을 evaluation에서 assert한다.

## Router 분류 결과 (4개 비중첩 test windows)
- mean policy: accuracy 0.8438, macro-F1 0.2893, 평균 예측 rank 23.5
- max policy: accuracy 0.4531, macro-F1 0.2615, 평균 예측 rank 200.1

## 전체 동시 압축 Pareto
Full MLA test loss 2.106.

- mean router: loss 3.986 (Δ+1.880), fixed-MLA persistent bytes의 20.2%, 평균 rank 24.16
- max router: loss 2.260 (Δ+0.154), fixed-MLA persistent bytes의 77.5%, 평균 rank 189.16
- fixed rank160: loss 2.318 (Δ+0.212), fixed-MLA bytes의 67.4%

동일 tier histogram을 token 위치 사이에서 20회 셔플한 control 대비:
- mean router: -0.111 nat
- max router: -0.112 nat

4개 sequence 모두를 cluster로 한 exploratory bootstrap interval은 0 아래였지만, 독립 cluster가
4개뿐이므로 population-level significance로 해석하지 않는다.

## 방어 가능한 결론
이 네 window에서 lexical token-tier assignment가 같은 memory budget의 무작위 위치 배정보다
낮은 loss를 보였다는 PoC 신호는 있다.

## 아직 주장할 수 없는 것
- contextual token-adaptive routing
- 전체 동시 압축에 대한 oracle-optimal policy
- 일반화된 Pareto 우위나 통계적 유의성
- peak-memory/latency 개선

## 다음 연구 단계
layer 0은 고정 full-rank로 처리해 contextual representation을 만든 뒤, 그 hidden으로 layers 1+
공유 tier를 결정하는 contextual router와 그 정의에 맞는 새 oracle label을 생성해야 한다.
