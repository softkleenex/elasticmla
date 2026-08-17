# Exp1 v6 Kaggle completion status

- Kernel: `softkleenex/elasticmla-exp1-scale-up-training-114m`
- Status: COMPLETE
- Hardware/runtime: Tesla P100 16GB, torch 2.2.0+cu118, fp16
- Model: 122,141,952 unique parameters (tied token embedding/output head)
- Config: d_model 768, 12 layers, 12 heads, d_head 64, d_rope 32, d_c 384, max_len 384
- Data: 88,081,007 train tokens; 1,797,572 validation tokens
- Training: 8,000 optimizer steps, micro-batch 8, gradient accumulation 6
- Approximate processed tokens: 147,456,000 (1.67 train-data epochs)
- Runtime: 20,592 seconds (5.72 hours)
- Final train loss: 1.3965
- Final validation loss: 1.5662 at step 8000
- Best logged validation loss: 1.5179 at step 7250
- Output checkpoint: final step 8000 only; v6 did not preserve the best checkpoint
- Strict local state-dict load: passed
- Non-finite tensors: none
- CPU 384-token forward smoke test: loss 1.7398, finite logits
- Checkpoint SHA-256: `eeb8abf6366399298350eb183d2ff740b0966b9e1bf4426adaa6dde60dcf8b93`

The downloaded artifacts live under `kaggle_output/exp1_v6/` and are intentionally ignored by git.
The kernel title is stale: the trained model is 122.14M unique parameters, not 114M.
