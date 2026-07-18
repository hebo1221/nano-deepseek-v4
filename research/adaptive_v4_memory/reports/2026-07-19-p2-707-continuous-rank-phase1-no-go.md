# P2 707 continuous-rank Phase-1 decision

Date: 2026-07-19
Decision: **terminal Phase-1 NO-GO; exact-path Phase 2 remains unexecuted**

## What ran

The frozen ten-cell 707 full-forward matrix completed on source commit
`a8ebaf6ebbb6ec78181a916b6f78fd6c45273840`. It processed 23,040
conversations in 5,760 batches and captured 7,168 registered replay queries per
CSA layer in every checkpoint cell. The ten immutable raw artifacts occupy
171,148,205 bytes.

All complete 1x, 2x, and 4x quota objects, signal configurations, calibration
digests, workload counts, checkpoint bindings, and calibration bindings
reproduced exactly in 10/10 cells. The final audit was then recomputed from the
serialized target-free demand rows rather than trusting recorded eligibility
flags.

The terminal raw audit is
`artifacts/adaptive_v4_memory/paper_grade/p2_707_continuous_layer_rank/audits/full-forward.audit.json`
(file SHA-256
`d813ce37ee50b6ff9b452606598576a90f646abfb5659bc274f75112a02805b6`,
payload SHA-256
`6b20fb42c2cea94f9faead3433a3266b3634dd33014d4d2521e29b095be838d1`).
The compact tracked decision is
[`p2-707-continuous-rank-phase1.audit.json`](../results/p2-707-continuous-rank-phase1.audit.json).

## Result

The continuous relaxation solved the narrow point-tie problem but failed the
pre-registered stability requirement:

- All 20/20 budget cells had a unique point top and unique point bottom.
- All 40/40 adjacent-boundary point gaps exceeded the frozen float32 tolerance.
- Only 8/20 seed-scale-budget cells passed the complete stability gate.
- 16/40 boundary decisions had a non-positive 99% bootstrap lower bound.
- Six boundary decisions failed the deterministic A-half direction check and
  two failed the B-half direction check.

The four passing seed-scale pairs were S55 seeds 6071402 and 6071405, and S151
seeds 6071404 and 6071405, at both 2x and 4x. The other six seed-scale pairs
failed at both budgets. The problem is therefore no longer exact equality of
the macro point statistic; it is insufficiently stable separation across
paired trace batches and workload slices.

## Decision

The frozen continue gate required stable identified top and bottom layers in
all 20 cells. Because 12 cells failed, `exact_path_phase_permitted` is false.
No exact-path cell, exact-path terminal audit, rank probe, quality input,
quality execution, or Stage-B work was generated. The GPU lock was released
and the exact-path output root remains absent.

This narrows the scientific diagnosis: a deterministic point ranking can be
constructed from the richer continuous scores, but it is not sufficiently
reliable to define a five-seed layer-alignment intervention. Running quality
experiments from these ranks would convert calibration noise into the treatment
definition. More examples from the same frozen calibration stream might reduce
uncertainty, but that is a new estimand/design decision and is not authorized by
this result.

## Claim boundary

This is calibration-only structural evidence. It does not estimate adaptive
quota value, protected-pin value, exact-path transfer, natural-language
transfer, larger-model transfer, latency, throughput, HBM use, or quality. It
also does not show that any population quality effect is zero.
