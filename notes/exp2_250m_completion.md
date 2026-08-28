# 250M Kaggle training completion (Exp2)

- Kernel: `softkleenex/elasticmla-exp2-scale-up-training-250m` (v1)
- Status: COMPLETE
- Hardware/runtime: Tesla P100 16GB, torch 2.2.0+cu118, fp16
- Model: 249,269,248 unique parameters (tied token embedding/output head)
- Config: d_model 1024, 16 layers, 16 heads, d_head 64, d_rope 32, d_c 512, max_len 384
- Data: 88,081,007 train tokens; 1,797,572 validation tokens (same TinyStories corpus
  construction as the 122M run: identical N_DOCS=400000, seed 1337, split fraction. val.bin is
  byte-identical to the 122M val.bin, SHA-256 `03246b576bb789ac6bf15996fd642167008620f2ee96d8a45496d69c13f073ce`.)
- Training: 3,000 optimizer steps, micro-batch 4, gradient accumulation 12 (effective batch 48)
- Runtime: 15,604 seconds (4.34 hours)
- Final train loss: 1.6174
- Final validation loss: 1.6588 at step 3000 (single logged checkpoint; no best-checkpoint claim)
- Checkpoint SHA-256: `8a8774502016bb98a97eeb6f90de7af23f70f82283f933c6bd70fd2a315d275b`
- Strict local state-dict load: passed
- Non-finite tensors: none
- CPU 384-token forward smoke test: loss 1.9242, finite logits

Downloaded artifacts live under `kaggle_output/exp2_250m/` and are intentionally ignored by git
(same policy as `kaggle_output/exp1_v6/`).

Used only for the risk-capacity spectrum third scale point (`notes/risk_capacity_spectrum_results.md`);
not yet used for contextual router training, oracle generation, or fresh-window confirmation.
