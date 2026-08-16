# 리소스/자격증명 & 컴퓨트 환경 체크 (2026-08-15)

## 1. API / 계정 (모두 사용자 ~/.zshrc 에 설정되어 있음 — 값은 이 문서에 기록하지 않음)

| 서비스 | 상태 | 확인 방법 | 비고 |
|---|---|---|---|
| Kaggle (`kaggle.json` + KAGGLE CLI) | ✅ 정상 작동 확인 | `uv run kaggle competitions list` 성공 | 데이터셋/노트북 GPU(P100/T4x2) 활용 가능 |
| Hugging Face (`HF_TOKEN`) | ✅ 정상 작동 확인 | `HfApi().whoami()` → user "softkleenex", fineGrained 토큰 | 모델/데이터셋 업로드, HF Hub 체크포인트 배포 가능 |
| Weights & Biases (`WANDB_API_KEY`) | ❌ 401 인증 실패 | `wandb.login(verify=True)` | **키가 만료/폐기된 것으로 보임 → wandb.ai에서 새 키 재발급 필요.** 재발급 전까지는 실험 로깅은 CSV/로컬 TensorBoard로 대체 |
| DACON (`DACON_TOKEN`, team: softkleenex) | 미검증 | - | MLA 논문과 직접 관련은 낮음, 필요시 활용 가능 |
| Serper (`SERPER_API_KEY`) | 존재함, Prime Agent MCP 연결은 별도 필요 | `/login` → MCP Connections → Serper에 같은 키 재등록하면 `websearch.run()` 정상화됨 | arXiv API로 대체 가능해서 필수는 아님 |
| Codex LB API Key | 존재함 (용도: 별도 codex 관련 워크플로우로 추정) | - | 이번 프로젝트와 직접 연관 없음 |

## 2. 컴퓨트 자원 현황

| 자원 | 사양 | 접근성 | 용도 |
|---|---|---|---|
| 로컬 Mac (M4 Pro) | GPU 16코어, 통합메모리 25.8GB, MPS 지원 확인됨 | 이 세션에서 직접 사용 중 | 코드 개발, 소규모(114M~ 수백M) 프로토타입, 디버깅 |
| **4090 PC** | RTX 4090 (24GB VRAM) | **이 세션에서 직접 접근 불가** (별도 물리 머신) | 본 실험(Experiment 0~1)의 메인 학습 머신으로 적합. SSH 붙여주면 원격 실행 가능, 아니면 코드/스크립트만 준비해서 사용자가 직접 4090에서 실행 |
| Kaggle Notebook GPU | P100 또는 T4×2, 세션당 최대 30h/week (무료 티어) | Kaggle CLI로 노트북 push/실행 가능 | 대규모 ablation, 여러 config 병렬 실험에 활용 |
| Colab | T4 무료 / A100 (Pro) | 브라우저 기반, CLI 직접 제어는 제한적 | 빠른 prototyping, 시각화 공유용 |

**결론: 4090 PC가 메인 학습 리소스, Kaggle/Colab은 보조(ablation, 백업, 공유용).**
4090에 SSH 접속 정보를 주면 이 세션에서 직접 스크립트를 실행/모니터링 할 수 있어. 아니면 내가 학습 스크립트+실행 커맨드를 만들어주고 네가 4090에서 직접 돌리는 방식으로 가면 돼.

## 3. 다음에 필요한 조치
1. **WANDB API 키 재발급** (wandb.ai → Settings → API keys) 후 `.zshrc`에 갱신
2. 4090 PC 접근 방식 결정: (a) SSH 터널 열어서 나에게 직접 접근 허용 / (b) 스크립트만 받아서 직접 실행
3. Serper 키를 Prime Agent MCP Connections에 재등록하면 arXiv 외 일반 웹서치도 가능해짐 (급하지 않음)
