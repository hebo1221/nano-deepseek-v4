# P1 causal layer-quota calibration pilot

Date: 2026-07-14  
Decision: **calibration mechanism valid; held-out quality evaluation not yet run**

## Question

Can same-token control turn the global 1x/2x/4x capacity grid into deterministic
per-layer quotas using only disjoint calibration score statistics, while
preserving a fixed-policy floor in every CSA layer?

This pilot does not compare model accuracy. Targets and native correctness were
not inputs to the quota fit.

## Frozen pilot

- model checkpoint seed: 6071401 at S55 and S151;
- calibration seed: 7071401;
- 32 conversations per each of the nine P2 workload families;
- context grid: 80, 128, 256, 512, and 1024 tokens;
- nearest-rank quantile: 95%;
- score-derived requested blocks plus candidate count only;
- per-layer floor: memory-matched fixed top-k (S55=2, S151=1);
- uncertainty extra allowance: floor × (budget multiplier − 1).

Each S55 layer received 896 scored calibration queries (2,688 total); each S151
layer received 896 (4,480 total). Every checkpoint hash, source-clean flag,
budget bound, and calibration digest was independently audited.

## Calibrated normal quotas

| Scale | 1x | 2x | 4x |
| --- | --- | --- | --- |
| S55 | L2=2, L4=2, L6=2 | L2=4, L4=4, L6=3 | L2=5, L4=6, L6=5 |
| S151 | L2/L4/L6/L8/L10=1 | all five layers=2 | all five layers=4 |

The allocator did not force heterogeneity: S55 showed asymmetric calibrated
demand, while S151 remained uniform. S55's 2x and 4x ceilings have 12 and 24
slots but calibrated quotas sum to 11 and 16; unused capacity is not silently
filled. Later memory matching therefore uses measured hot bytes rather than the
nominal multiplier.

Dense fallback quotas are 8 blocks per S55 layer and 4 per S151 layer. They are
separate from the normal quota and remain subject to the preregistered fallback
ablation.

## Artifact binding

- S55 raw SHA-256:
  `7442dccc4255400da68d28cbd317098cc3631017f72ea73b132861d71d0372f9`
- S151 raw SHA-256:
  `a624761653180fdbb0fb3fe5311193375b4efa05efbbbb3e4a1373a5d42d676d`
- checked summary SHA-256:
  `db89aee9247654cf7c48e925a2c0364dc54c2c3c7a2bbfe7e503fb0bd714cdce`
- source commit for both raw runs:
  `a8dd5d7163f8f2e30a6785c342824509472bd0d3`

The raw files remain ignored under
`artifacts/adaptive_v4_memory/paper_grade/calibration/`. The checked summary is
`results/p1-layer-quota-calibration-pilot.summary.json`.

## Boundary and next gate

This establishes deterministic, leakage-guarded quota fitting on one checkpoint
per scale. It does not show that calibrated quotas improve quality, memory, or
latency. Next, the same calibration must run for all five checkpoint/seed pairs,
then the quotas enter paired held-out evaluation against uniform and strongest
fixed policies. No held-out 807-series result has been inspected.
