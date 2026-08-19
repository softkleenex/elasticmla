# Fresh-window contextual routing confirmation protocol

This protocol was fixed **before** examining the new confirmation windows.
The older four-window `test` results are exploratory because they were repeatedly observed during
method development. They are not used as confirmatory evidence.

## Frozen policies

| Scale | Policy | SHA-256 | Tiers | Training |
|---|---|---|---|---|
| 30M | `joint_lambda_0p4.pt` | `8ca2b654bbbe6e4fc050aaed0625d84d1c4586ac6bf880db01cd797056e9d71e` | `{16,64,160,256}` | ST joint rollout, λ=0.4, seed 3031, 15 epochs, oracle-router initialization |
| 122M | `fine_lambda_0p8.pt` | `200508e6b44ccff7bd15c55e4fcf1ba9b091a1f3868ec64bacf9a4626f85c8ec` | v4 14-rank grid `16..384` | ST joint rollout, λ=0.8, seed 3031, 15 epochs, random initialization |

Both policies were selected using only their original 16 training and four validation windows.
No parameter, tier, λ, epoch, or threshold will be changed after confirmation results are viewed.

## New data

- Deterministic seed: 91,827.
- 24 new windows per scale.
- Each window has the model's full context length plus one target token.
- New windows are sampled from the same held-out evaluation region used by v4.
- Every new window must be separated from every prior oracle/train/validation/exploratory window
  and every other new window by at least `block_size + 1` tokens.

## Controls and endpoints

Primary control: **exact-byte static interpolation**. For each sequence, if the router's total
rank is not divisible by sequence length, a content-independent mixture of ranks `q` and `q+1`
is randomized 20 times. Every repeat has exactly the router's downstream rank sum and packed
bytes.

Secondary control: the router's exact tier histogram shuffled across token positions 20 times.

Primary endpoint: paired per-sequence `router_loss - exact_static_loss`.

Pre-specified success criterion at each scale: the upper bound of the 95% sequence-cluster
bootstrap interval is below zero. One-sided paired sign-flip randomization p-values are reported
as supporting statistics. The two scales are reported separately; success at one scale is not
substituted for failure at the other.

Other reported quantities: loss increase versus full MLA, persistent packed bytes versus dense
fixed-width MLA, average downstream rank, sequence win counts, and shuffled-control difference.
No peak-memory or latency conclusion is permitted by this experiment.
