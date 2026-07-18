# P2 707 continuous layer-rank enrichment freeze

Date: 2026-07-19
Status: frozen before any new continuous-rank GPU result

The old calibrated-versus-shuffled Stage A remains structurally blocked. Its
discrete quotas identify only 10 of 20 seed-scale-budget mappings, so increasing
the number of quality conversations cannot repair the estimand. This new study
uses only the already authorized 707 calibration namespace and does not revive
the old arm, quality matrix, or Stage B.

The design is explicitly adaptive to the existing calibration-only tie diagnosis
and the previously observed path-localization result. No quality outcome was used
to choose the metric. There is one primary metric, no rescue metric, and no layer
ID or digest tie-break.

## Frozen primary statistic

For every registered replay query, the controller signal is reconstructed with
the original P1 recipe. The final integer ceiling is relaxed, giving

`min(candidate_blocks, max(min_blocks, top_p_cardinality) + uncertainty * max_extra)`

with zero demand when there are no candidates. Within each family-by-context
slice, the exact empirical upper-five-percent mean is computed with fractional
boundary mass. A layer's score is the equal-weight mean of the 45 slice scores.

## Phase 1: full-forward structural gate

The ten frozen checkpoint/calibration cells are rerun with the original collector:
2,304 conversations and 576 batches per cell, or 23,040 conversations and 5,760
batches total. Before continuous evidence is accepted, the complete prior 1x,
2x, and 4x quota objects and digests must reproduce exactly.

For each 2x/4x cell, uncertainty is a 10,000-replicate paired trace-batch
bootstrap stratified over the 45 slices, with a 99% interval. Canonically
alternating trace batches form independent A/B stability halves. A boundary is
identified only when it is point-unique, exceeds the frozen float32 tolerance,
has a strictly positive lower confidence bound, and is positive in both halves.
All 20 cells must identify both top and bottom. A single failure stops the study
before Phase 2.

## Phase 2: exact-path transfer gate

Only after a complete Phase-1 pass, the first five-context cycle of each family
is run on the literal prefix-then-one-token sequential-tiered path. The reference
is budget-specific uniform-total `fixed+pins`, with fallback and adaptive signals
disabled. A canonical Bresenham low/high schedule preserves the exact aggregate
configured total across 45 slices.

Both budgets are extracted in forward and reverse coordinate order. These orders
are deterministic repeats, not resampling units. Across ten cells this is 1,800
batch runs, 1,800 unique conversations, and 7,200 executed conversations. The
workload, query, exact FP32 score, and demand streams must match across orders.
The 99% paired bootstrap resamples the 45 family-context slices. Every exact-path
top/bottom must be identified and equal its Phase-1 counterpart.

## Decision boundary

Even a complete pass permits only preparation of a separate versioned rank-probe
manifest. It does not permit that probe, quality execution, reuse of the old
quality seed namespace, or Stage B. A failure at either phase is published as a
terminal design result. Supported claims remain limited to target-free
calibration-rank feasibility and path transfer.

The executable contract is
[`p2-707-continuous-layer-rank-enrichment-v1.json`](../manifests/p2-707-continuous-layer-rank-enrichment-v1.json).
