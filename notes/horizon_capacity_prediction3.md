# Matched-prefix horizon sweep: theory prediction 3 (2026-09-12)

> **Correction (2026-10-20, `notes/code_review_2026-10_corrections.md`).** All 250M values below came from a rank grid that jumped 384 -> 512. Corrected 250M worst-offset curve: published h=1..32 = 18.61/40.80/54.74/63.92/69.00/71.78%; extended h=1..128 = 17.70/40.38/53.60/63.30/68.69/71.61/73.12/74.28% (final doubling +1.16 pp, h=16->32 +2.78 pp). Prediction 3 and the position-control conclusion (max shift 1.73 pp, premium <= 0.009) are unchanged.


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

## Extension to h=128: saturation reached

The published sweep stops at h=32, where the worst-offset curve is still rising by 2-3 pp per
doubling -- saturation setting in, not reached. A T4 sweep to h=128 was run at all three scales
(`elasticmla-horizon-{30m,122m,250m}-0912b`), with the same defaults otherwise. Artifacts:
`experiments/contextual_router_{30m,122m,250m}/horizon_capacity_A_extended.json`, report
`experiments/horizon_capacity_extended.json`.

Worst-offset (`maximum`) criterion, normalized mean `r*`:

| h | 30M | 122M | 250M |
|---:|---:|---:|---:|
| 1 | 23.27% | 15.84% | 18.40% |
| 2 | 46.39% | 39.15% | 41.50% |
| 4 | 61.21% | 54.28% | 55.84% |
| 8 | 68.25% | 64.77% | 66.76% |
| 16 | 72.50% | 69.53% | 72.82% |
| 32 | 74.97% | 72.25% | 75.89% |
| 64 | 76.18% | 74.02% | 77.65% |
| 128 | **76.61%** | **74.95%** | **78.95%** |

Still strictly nondecreasing with shrinking steps at every scale, and the final doubling
(h=64 -> 128) now moves only **+0.43 pp (30M), +0.93 pp (122M), +1.30 pp (250M)** -- against
+2.07 / +2.78 / +2.95 pp for h=16 -> 32. Prediction 3's "rise or saturate" is satisfied in the
strong sense: it rises, and the rise flattens.

This curve is a real measurement, not a boundary artifact: at h=128 the mean `r*` is 196.1 of
`d_c`=256 (30M), 287.8 of 384 (122M) and 404.2 of 512 (250M), far from both the rank grid's
minimum (16) and its maximum.

## Caveat: the signed-mean curve is floor-censored at long horizons

The signed mean keeps falling, as predicted, but past h=32 it runs into the rank grid's smallest
entry (16) and stops being measurable:

| scale | h=32 | h=64 | h=128 |
|---|---:|---:|---:|
| 30M (floor 16 = 6.25%) | 23.04 | 17.52 | **16.00, CI [16.00, 16.00]** |
| 122M (floor 16 = 4.17%) | 30.77 | 18.98 | 16.29, CI [16.00, 16.69] |
| 250M (floor 16 = 3.12%) | 33.23 | 20.54 | 16.38, CI [16.02, 16.92] |

At 30M, h=128 every single probe is at the floor -- the bootstrap interval has zero width. So the
long-horizon signed-mean values are **upper bounds on a quantity the grid cannot resolve**, not
estimates. This does not affect prediction 3, whose non-monotonicity claim is established by the
h=1 -> 2 rise and the h=2 -> 32 fall, all well above the floor. Resolving the tail of this curve
would need a rank grid extended below 16.

## Position control: the published table is robust to probe-position restriction

A longer horizon can only be probed at earlier positions (`valid_probe_positions` returns
`block_size - h + 1`), so the extended run's probes sit in positions `0..block-128` rather than the
published run's `0..block-32`. To separate that from the horizon effect, a second run per scale
recomputed the *published* criterion (H=32) on the extended run's position set
(`--future-horizon 32 --position-horizon 128`), saved as `horizon_capacity_B_position_matched.json`.

Headline tail-count table, published positions -> matched positions (normalized mean `r*`, pp):

| k | 30M | 122M | 250M |
|---:|---|---|---|
| 32 (mean) | 9.00 -> 9.00 (+0.00) | 8.22 -> 8.01 (-0.21) | 7.15 -> 6.49 (-0.66) |
| 16 | 21.47 -> 22.88 (+1.42) | 23.00 -> 23.38 (+0.38) | 22.53 -> 22.21 (-0.32) |
| 8 | 39.65 -> 40.69 (+1.04) | 39.52 -> 40.40 (+0.88) | 40.14 -> 40.40 (+0.26) |
| 4 | 55.75 -> 57.14 (+1.39) | 54.22 -> 55.94 (+1.73) | 57.23 -> 57.64 (+0.41) |
| 2 | 66.94 -> 68.06 (+1.12) | 65.56 -> 66.08 (+0.52) | 69.18 -> 68.73 (-0.44) |
| 1 (max) | 74.07 -> 74.97 (+0.90) | 72.92 -> 72.25 (-0.66) | 76.03 -> 75.89 (-0.14) |

Every shift is at most 1.73 pp, most are under 1 pp, and the sign is inconsistent across scales and
tail counts -- i.e. noise, not a position trend. The tail-capacity premium, the paper's strongest
scale-consistency claim, moves by at most 0.009 (30M 0.6507 -> 0.6597, 122M 0.6470 -> 0.6424,
250M 0.6888 -> 0.6940), comfortably inside the published bootstrap intervals. The published spectrum
is therefore not an artifact of which positions were eligible to be probed.

As an internal check, run A's `h=32` entry and run B's `h=32` entry agree to machine precision at
all three scales and on every criterion, as they must: separate jobs, same evaluation seed, same
eligible position set, same probes.
