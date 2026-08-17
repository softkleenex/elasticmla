# Compute fallback policy

## Priority order

1. Use Kaggle GPU/CLI for training and large analyses when quota and a compatible GPU are available.
2. If Kaggle cannot start or continue a workload because of quota, session limits, or platform failure, continue on **Lightning AI using the account's free credits**.
3. Use local MPS only for smoke tests, debugging, and reduced pilots unless both cloud routes are unavailable or an explicit local run is requested.

## Lightning AI operating rules

- Prefer an already-running free-credit Studio before creating another resource.
- Current fallback target: `softkleenex/training-optimization-project`, Studio `optimize-training-devbox`, T4.
- Do not intentionally exceed available free credits or switch to a paid machine/account without explicit approval.
- Upload only required source, checkpoint, and validation data; keep large artifacts out of git.
- Launch the workload asynchronously and record its command, PID/job identifier, log path, machine, and output location.
- **Five minutes after launch**, verify that the process/job is still running, GPU execution is active, logs are advancing, and no OOM/dependency/input-path failure occurred.
- If the five-minute check fails, stop the resource where appropriate, diagnose, and do not report the experiment as running.
- Download result JSON/logs after completion and independently validate them locally before use in the paper.

## Current trigger

On 2026-08-17, Kaggle rejected the 122M Exp0-v3 analysis push with `Maximum weekly GPU quota of 30.00 hours reached`. This activates the Lightning AI fallback policy.
