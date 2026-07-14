# P1 paired held-out policy pilot — S151

Date: 2026-07-14  
Decision: **protected retention replicates; calibrated quota advantage remains unproven**

The S151 checkpoint for training seed 6071401 was evaluated on the same frozen
design as the S55 pilot: disjoint calibration seed 7071401, held-out seed
8071401, nine families, five contexts, 20 conversations per family, and 13
policies. The audit verified 2,340 exact paired policy-conversation records and
560 scored queries per policy with 10,000 cluster-bootstrap and sign-flip
resamples.

| Budget | Fixed | Hierarchical | Difference | 95% paired cluster CI | Sign-flip p | H2D ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1x | 35.36% | 41.25% | +5.89 pp | [+2.65, +9.74] pp | 0.0006 | 0.973x |
| 2x | 39.82% | 42.50% | +2.68 pp | [-0.18, +5.82] pp | 0.112 | 0.998x |
| 4x | 43.75% | 43.75% | 0.00 pp | [-1.88, +1.98] pp | 1.000 | 0.867x |

The apparent 1x gain is not a quota-calibration result. The 1x quotas are
uniform and identical to fixed; the exact no-pin diagnostic reproduced fixed
predictions. With pins, instruction persistence rose from 30/80 to 63/80.
Across all queries the controller recovered 34 and regressed one. The mechanism
is protected instruction retention.

At 2x the pilot is unresolved, and at 4x calibrated hierarchy exactly matched
fixed accuracy while transferring 13.3% fewer H2D bytes. Dense fallback had
zero net accuracy change at every budget and did not establish value. These
remain single-checkpoint directional findings.

The raw pilot SHA-256 is
`d7580aef918fd7f7b7ee16b6797fa19c61842b2656bf2114988c609c7b9ff939`.
The checked result is `results/p1-heldout-policy-pilot-s151.summary.json`.
