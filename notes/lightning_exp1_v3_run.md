# Lightning AI Exp1-v3 122M run — invalidated and stopped

- Trigger: Kaggle weekly GPU quota exhausted.
- Policy: `notes/compute_fallback_policy.md`
- Teamspace: `softkleenex/training-optimization-project`
- Source Studio: `optimize-training-devbox`
- Job: `elasticmla-exp1-v3-122m-0817`
- Machine: T4
- Scheduler launch: successful.
- Exact command:
  ```bash
  cd /teamspace/studios/this_studio/elasticmla_exp1_v3 && mkdir -p results && .venv/bin/python experiments/analyze_rank_variance_v3.py --device cuda --checkpoint ckpt/latest.pt --data data/val.bin --output-dir results --rank-grid 16 32 48 64 96 128 160 192 224 256 288 320 352 384
  ```
- Checkpoint SHA-256: `eeb8abf6366399298350eb183d2ff740b0966b9e1bf4426adaa6dde60dcf8b93`
- Validation-data SHA-256: `03246b576bb789ac6bf15996fd642167008620f2ee96d8a45496d69c13f073ce`
- Intended output: `/teamspace/studios/this_studio/elasticmla_exp1_v3/results/`
- Five-minute observation: scheduler status was `Running` at approximately 327 seconds. Batch-job stdout, job GPU utilization, and intermediate files were not exposed by the CLI, so this **did not satisfy** the full health-check policy and does not prove GPU activity or log progress.
- Review outcome: stopped after first code review found a P1 off-by-one error in the intended future-loss window. Reported final scheduler status after stop was `Completed`, total reported cost `0.19872223`; any output from this job is invalid for paper use.
- Replacement: corrected analysis is versioned as Exp0-v4 and must pass the second review before relaunch.
