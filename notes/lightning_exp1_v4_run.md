# Lightning AI Exp1-v4 122M run

- Precondition: two code reviews completed. Review 1 found and triggered correction of the immediate-next-token horizon; review 2 found only a default-grid P2, corrected before launch. No P0/P1 remained.
- Code commit: `0704c74`
- Job: `elasticmla-exp1-v4-122m-0817`
- Teamspace/Studio: `softkleenex/training-optimization-project` / `optimize-training-devbox`
- Machine: T4 batch job
- Script SHA-256: `03f7164d6cd0fc936b4a785a750381a378f5aeaae83e6efc4463483ac0cf5f69`
- Checkpoint SHA-256: `eeb8abf6366399298350eb183d2ff740b0966b9e1bf4426adaa6dde60dcf8b93`
- Validation-data SHA-256: `03246b576bb789ac6bf15996fd642167008620f2ee96d8a45496d69c13f073ce`
- Exact command:
  ```bash
  bash -lc 'set -o pipefail; cd /teamspace/studios/this_studio/elasticmla_exp1_v3 && mkdir -p results_v4 && .venv/bin/python experiments/analyze_rank_variance_v4.py --device cuda --checkpoint ckpt/latest.pt --data data/val.bin --output-dir results_v4 --rank-grid 16 32 48 64 96 128 160 192 224 256 288 320 352 384 2>&1 | tee results_v4/analysis.log'
  ```
- Five-minute observation: scheduler status `Running` at approximately 345 seconds; reported total cost `0.049816668`.
- The batch filesystem/log had not synced to the Studio file API at the check, so GPU utilization and advancing stdout were not directly observed. This is a partial scheduler health check, not a full policy health-pass or result validation.
- Expected outputs after completion: `results_v4/analysis.log`, `results_v4/exp0_v4_summary.json`, `results_v4/exp0_v4_records.json`.

## Completion

Job completed successfully. SDK logs show CUDA execution, all 14 ranks completed, and both v4 JSON files written. Artifacts were downloaded from `/teamspace/jobs/elasticmla-exp1-v4-122m-0817/artifacts/`, validated locally, and the temporary CPU Studio used for retrieval was stopped.
