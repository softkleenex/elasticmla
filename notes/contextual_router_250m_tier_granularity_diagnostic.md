# Diagnostic: tier granularity vs scale as the cause of the 250M confirmation result

## Motivation

The initial 250M contextual-router replication used the same coarse 4-tier grid {32,128,320,512}
as 30M, while 122M used a 14-tier fine grid {16,...,384}. Since 30M (coarse) and 122M (fine) both
passed pre-registered confirmation while 250M (coarse) failed, tier granularity is a genuine
confound for any "the router's advantage vanishes with scale" claim. We ran a second, fully
independent, pre-registered confirmation at 250M using an 18-tier fine grid {16,32,...,512} with
random router initialization (mirroring exactly how the 122M fine-grid policy was built), reusing
the same 250M base checkpoint, oracle, and channel orders, to isolate tier granularity from scale.

## Result: tier granularity, not scale alone, explains the coarse-grid failure

| 250M variant | Tiers | vs exact-byte static | 95% CI | Result |
|---|---|---:|---|---|
| Coarse (original) | {32,128,320,512} | +0.00190 nat | [-0.00222, +0.00610] | **failure** |
| Fine (diagnostic) | {16,32,...,512} (18 tiers) | **-0.01165 nat** | **[-0.01911, -0.00394]** | **success** |

With the fine grid, the pre-registered criterion (upper 95% CI < 0) is met: 17/24 sequence wins,
exact sign-flip p=0.0044. Audited independently (`fresh_confirmation_fine_audit.json`, status
passed) and reproduced bit-for-bit by a from-scratch retrain
(`joint_training_replay_audit_fine.json`, status passed).

**Interpretation:** the coarse 4-tier grid becomes too low-resolution at d_c=512 for the router to
usefully beat pure random rate allocation; a finer grid restores that advantage. This means the
earlier headline claim "the router's benefit vanishes with scale" was **partly a tier-resolution
artifact of our own experimental design**, not purely an intrinsic scale limitation. We correct
this here rather than let the overstated version stand.

## But causal heuristics beat the router regardless of, and more decisively with, tier granularity

Re-fitting the same four causal heuristics (position, lexical identity, token rarity, token type)
against the **fine** tier grid gives:

| Control | Router - control (nat), fine grid | 95% CI | Result |
|---|---:|---|---|
| position | +0.0131 | [0.0020, 0.0254] | **router loses** |
| lexical identity | +0.0557 | [0.0448, 0.0682] | **router loses** |
| token rarity | +0.0371 | [0.0252, 0.0501] | **router loses** |
| token type | +0.0612 | [0.0492, 0.0744] | **router loses** |

This is a reversal from the coarse-grid causal-heuristic table, where lexical identity beat the
router by a similar margin (i.e. the router won). With finer tiers, **all four heuristics beat the
router, more decisively than at the coarse grid.** The mechanism is intuitive: finer tiers let a
cheap heuristic allocate its budget more precisely too, and the heuristics benefit from that extra
resolution more than the learned router does. So while tier granularity determines whether the
router beats *pure random* allocation, it does not rescue the router against these stronger,
still very cheap, non-learned baselines -- if anything the gap versus heuristics widens.

## Combined, corrected conclusion

1. The tail-capacity premium and cancellation-mechanism results (Section 5.1) are unaffected; they
   do not involve the router.
2. The "router beats random static allocation" result depends on tier granularity at 250M: fails
   coarse, succeeds fine. We no longer claim this specific comparison degrades monotonically with
   parameter count; it is at least partly a design choice (tier grid) that we did not hold fixed
   across scales in the original protocol.
3. The "router beats causal heuristics" result is more robust and more concerning: at every scale
   and every tier grid we have tested (30M coarse, 122M fine, 250M coarse, 250M fine), at least one
   and typically several simple, non-learned, causally-valid heuristics match or beat the router at
   equal bytes, and this does not improve with finer tiers. This is the paper's most durable
   negative finding about the current router design, independent of the tier-granularity confound.

## Provenance

- `experiments/contextual_router_250m/fresh_confirmation_fine.json`
- `experiments/contextual_router_250m/fresh_confirmation_fine_audit.json`
- `experiments/contextual_router_250m/joint_training_replay_audit_fine.json`
- `experiments/contextual_router_250m/causal_heuristic_controls_fine.json`
- `experiments/contextual_router_250m/fine_lambda_0p8.pt` (frozen policy), `fine_joint_rollout_selection.json`
- Manifest scale key `250m_fine` in `experiments/fresh_confirmation_manifest.json`
