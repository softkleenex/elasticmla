# Lightning AI Exp1-v3 122M run

- Trigger: Kaggle weekly GPU quota exhausted.
- Policy: `notes/compute_fallback_policy.md`
- Teamspace: `softkleenex/training-optimization-project`
- Source Studio: `optimize-training-devbox`
- Workload isolation: separate Lightning batch job; the already-running Studio T4 process was not interrupted.
- Job: `elasticmla-exp1-v3-122m-0817`
- Machine: T4
- Command: Exp0-v3 analysis on the final 122.14M checkpoint, 24 evaluation sequences, 32 probes/sequence, ranks `{16,32,48,64,96,128,160,192,224,256,288,320,352,384}`.
- Launch: successful after correcting the CLI teamspace argument format.
- Five-minute check: `Running` at approximately 327 seconds after successful launch; no immediate dependency, input-path, CUDA, or OOM failure. Reported cost at check: 0.030761112 credits/currency units.
- Intermediate result directory is job-isolated and is expected to sync back to the Studio after completion; the CLI did not expose stdout progress during this check.
- Interpretation: the five-minute health check confirms continued execution, not experimental completion or result validity.
