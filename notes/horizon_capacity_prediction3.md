# Matched-prefix horizon sweep: theory prediction 3 (2026-09-12)

## What prediction 3 says

From `notes/theory_contextual_tail_rate.md`, Section 7:

> 3. Worst-offset safe rate should rise or saturate with a longer matched-prefix horizon; signed
>    mean need not be monotone.

## It was already computed, just never analyzed

`notes/submission_readiness_roadmap.md` listed this sweep under Package A as "designed but not
run". That was wrong. `experiments/analyze_risk_capacity_spectrum.py` has emitted a
`horizon_capacity_spectrum` block since commit `ae187c2`, and all three published, provenance-
checked summaries carry it:
`experiments/contextual_router_{30m,122m,250m}/risk_capacity_v5_summary.json`.

The block recomputes `r*` for each criterion at horizons `h in {1,2,4,8,16,32}` from
`record["future_delta_by_rank"][rank][:h]` -- the **first `h` offsets of the same probes'
future-delta vectors**. The horizons are nested prefixes of one probe population (768 positions at
each scale), so this sweep has no position confound. That distinction matters: sweeping
`--future-horizon` across separate runs does *not* give this, because `valid_probe_positions`
returns `block_size - h + 1` positions, so a longer horizon silently restricts probing to earlier
positions and confounds horizon length with prefix length. (`--position-horizon` was added to the
analysis script so that cross-run sweeps can pin the position set; the within-run spectrum used
here never needed it.)

`experiments/analyze_horizon_capacity.py` reads the summaries and tests the prediction. It runs no
model and needs no GPU. Report: `experiments/horizon_capacity_published.json`.

## Result: confirmed at all three scales

Normalized mean `r*` (percent of `d_c`), worst-offset (`maximum`) criterion:

| h | 30M | 122M | 250M |
|---:|---:|---:|---:|
| 1 | 21.00% | 14.54% | 18.75% |
| 2 | 45.40% | 39.13% | 42.13% |
| 4 | 60.56% | 55.24% | 57.12% |
| 8 | 67.20% | 65.06% | 67.30% |
| 16 | 72.01% | 70.14% | 73.08% |
| 32 | 74.07% | 72.92% | 76.03% |

- **Rises: yes.** Strictly nondecreasing at every step at all three scales.
- **Saturates: yes.** Step sizes shrink monotonically. At 30M: +24.40, +15.16, +6.64, +4.81,
  +2.07 pp. The 122M and 250M curves behave the same way, ending at +2.78 and +2.95 pp.
- **Signed mean is non-monotone: yes.** It peaks at `h=2` (31.62% / 26.18% / 27.34%) and then falls
  steadily to 9.00% / 8.22% / 7.15% at `h=32`. `positive_mean` peaks at `h=4` and also falls.

All three sub-claims hold at all three scales.

## Why the two criteria move in opposite directions

This is the cancellation mechanism of Section 5.1, seen along a different axis. Lengthening the
horizon adds offsets. For the worst-offset criterion each added offset is another chance to exceed
epsilon, so the required rate can only rise -- and it saturates once the horizon is long enough to
almost surely contain the worst offset. For a mean criterion the added offsets mostly cancel, so
the same lengthening *lowers* the required rate. The h=1 column is the one place the three criteria
must agree (a single offset has mean = positive part = maximum), and they do, to the digit
(21.00% / 14.54% / 18.75%), which is a useful internal consistency check on the three independently
bootstrapped curves.

## Extension in progress

The worst-offset curve is still rising by ~2-3 pp at the last available step, so h=32 shows
saturation setting in rather than reached. A T4 sweep to h=128 is running
(`elasticmla-horizon-{30m,122m,250m}-0912`), together with a position-matched H=32 control
(`--position-horizon 128`) that isolates how much of any level shift is due to that run's
restricted position set rather than to the longer horizon. Results will be appended here.
