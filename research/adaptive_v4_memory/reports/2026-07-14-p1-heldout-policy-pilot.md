# P1 paired held-out policy pilot

Date: 2026-07-14  
Decision: **directional signal found; protected-pin ablation and multi-seed expansion required**

## Design and audit

One S55 checkpoint (training seed 6071401) was evaluated with its disjoint
7071401 calibration artifact on held-out seed 8071401. The pilot covered all
nine preregistered workload families, all five context lengths, 20 conversations
per family, and 13 policies: native; fixed 1x/2x/4x; calibrated local and
hierarchical 1x/2x/4x; and hierarchical+dense-fallback 1x/2x/4x.

This yielded 180 conversations, 560 scored queries, and 2,340 paired
policy-conversation records. The audit recomputed every prediction's correctness,
verified exact example pairing, family/context balance, clean source provenance,
checkpoint and calibration SHA-256 digests, calibration/evaluation seed isolation,
and zero budget violations. Statistics use 10,000 conversation-cluster bootstrap
resamples and 10,000 paired sign-flip permutations with analysis seed 9071401.

The raw result SHA-256 is
`7dcdadfca99ee3fe0ecc18d0445ed60703b283ec2823fd5aa4853e1426372265`.

## Core result

| Budget | Fixed accuracy | Hierarchical accuracy | Difference | 95% paired cluster CI | Sign-flip p | H2D ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1x | 46.43% | 48.75% | +2.32 pp | [+0.51, +4.70] pp | 0.064 | 0.994x |
| 2x | 54.11% | 53.93% | -0.18 pp | [-1.53, +1.20] pp | 1.000 | 0.761x |
| 4x | 55.00% | 55.18% | +0.18 pp | [-1.35, +2.10] pp | 1.000 | 0.603x |

The 1x arm recovered 13 queries and regressed none relative to fixed 1x, but
the conservative conversation-level permutation test did not cross 0.05. The
2x and 4x quality differences were effectively unresolved at this pilot size.
The 4x hierarchical policy transferred about 40% fewer H2D bytes than fixed 4x
while matching its accuracy directionally; this is promising but not a
multi-seed Pareto claim.

## Causal interpretation limits

All 13 gains at 1x came from `instruction-persistence`: 57/80 correct for the
hierarchical controller versus 44/80 for fixed. The other eight families had
identical 1x correctness. Since the calibrated 1x layer quota is deliberately
identical to the uniform fixed floor, this is not evidence for non-uniform quota
allocation. It points instead to the controller path, most plausibly the
protected instruction block. A no-protected-pin arm is therefore mandatory
before attributing the gain.

A subsequent exact-example causal diagnostic completed that ablation. Disabling
pins reproduced fixed-1x predictions bit-for-bit (44/80 on instruction
persistence), while enabling pins scored 57/80. All 13 recoveries occurred at
context 256 or longer. The 1x gain is therefore attributed to protected pin,
not quota calibration; see the separate checked diagnostic.

Cross-layer hierarchy added only one net correct query at 2x and at 4x relative
to the local controller. Those differences are too small for a mechanism claim.

Dense fallback changed no correctness outcome at any budget. It increased mean
H2D traffic by 34.8% at 1x, 8.1% at 2x, and 2.5% at 4x. It is retained as a
reported negative ablation but should not consume the core full-matrix budget.

## Worst slice and execution lesson

Every policy, including native, scored 0/20 on
`irrelevant-context-local-only`. Native full-sequence forward reproduced the
same 0/20 predictions, ruling out cache decode and tier/controller paths. The
generator audit also verifies that the local evidence token equals the target,
the evidence key equals the query key, and the distance is inside the sliding
window. This is a checkpoint generalization failure, not an adaptive-memory
regression or malformed target.

The reference run took roughly 25 minutes because long-generation evaluation
performs real sequential cache decode and the controller uses a Python/CPU
selection path. Full evaluation must therefore use restartable
family/context/checkpoint shards. These timings are engineering observations,
not fused-runtime performance claims.

## Next decision

The protected-pin and irrelevant-local diagnostics are complete. Retain
protected pin as an independent mechanism and do not credit quota calibration
for its gain. Expand the minimal core set—native,
fixed 1x/2x/4x, and calibrated hierarchical without fallback 1x/2x/4x—to all
five seeds and both scales at 1,000 conversations per family. Local,
cross-layer, fallback, refresh, temporal, and score ablations run as separate
shards rather than multiplying the core matrix.

The checked artifact is `results/p1-heldout-policy-pilot.summary.json`, generated
by `scripts/summarize_p1_heldout_policy_pilot.py`.
