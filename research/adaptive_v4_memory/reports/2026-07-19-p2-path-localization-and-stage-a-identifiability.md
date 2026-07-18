# P2 path localization and Stage-A structural-identifiability decision

Date: 2026-07-19
Decision: **current calibrated-vs-shuffled Stage A is structurally NO-GO before GPU execution**

## Evidence boundary

The path-localization run was completed on commit `e7eea9d0f5aa3328aaa6e1479582be6989497458`.
Its 565,255,204-byte raw artifact has SHA-256
`8e22c83bebb8bf6a24b22070d10080106a03460a47037dcce0f8d2cb7ee56f37`.
The compact audit is
[`p2-path-split-diagnostic.summary.json`](../results/p2-path-split-diagnostic.summary.json)
(SHA-256 `1effb9f4d56f4e4f837407f165c2d94f231062610472af1ea7669aaaf889f546`).

The Stage-A feasibility audit read only the ten frozen 707-series calibration
artifacts, their bound checkpoints, and controller configurations. It did not
generate or open an 807-series input, target, prediction, or quality result.
The audit is
[`p2-targeted-stage-a-structural-identifiability.audit.json`](../results/p2-targeted-stage-a-structural-identifiability.audit.json)
(SHA-256 `0b406fdd7695dccf3ac2cfa143c8c4dca5db5cff01bd6b2a0ecf2dbd1a4a8379`).

Both prospective and targeted quality output roots were absent at the decision
boundary. This is therefore outcome-independent structural stopping, not
early stopping on interim quality.

## Path-localization result

The frozen panel completed 84 arm/path runs: seven known arm-cells, four
execution paths, and three reordered repeats. There were 72 successful runs,
12 repeat-stable expected `tiered-chunk2` budget errors, no successful-run
budget violation, and exact same-path repeats.

- Resident tokenwise versus resident chunk-2 changed the final registered-query
  prediction in all 21 observations, exactly one of sixteen queries each.
- Resident tokenwise versus tiered tokenwise changed no final registered-query
  prediction in 21/21 observations, although logits, intermediate top-1 values,
  and later selected sets did drift at 4x.
- Resident chunk-2 versus tiered tokenwise reproduced all seven historical
  mismatch vectors and directions.

Within these outcome-selected cells, chunking is sufficient and tiering is not
necessary for the categorical registered-query failures. Tiering is not
numerically neutral, and cross-corner interchangeability remains unestablished.
Near-zero BF16 margins support a shape-sensitive numerical explanation, but do
not establish a sole kernel-level cause.

Pin behavior was not exercised: all workloads had empty protected positions
and all 497,664 successful controller actions had empty pin sets. Repeat-stable
pin digests are therefore trivial integrity, not pin-mechanism evidence.

## Why the original Stage A cannot identify its estimand

The clean contrast requires `calibrated+pins - shuffled-quota+pins` with the
same quota multiset and all five checkpoint seeds in each scale-by-budget cell.
A seed cell contributes 900 eligible conversations when its calibrated quota
vector is non-uniform, and zero when it is uniform. Permuting a uniform vector
cannot create an intervention; adding more conversations cannot repair it.

| Cell | Identified seeds | Unidentified seeds | Gate |
|---|---:|---|---|
| S55 / 2x | 3/5 | 6071403, 6071404 | fail |
| S55 / 4x | 4/5 | 6071404 | fail |
| S151 / 2x | 1/5 | 6071401–6071404 | fail |
| S151 / 4x | 2/5 | 6071401, 6071402, 6071404 | fail |

Exactly 10 of 20 seed-scale-budget mappings are identified. Every one of the
four decision cells lacks at least one required seed, so the preregistered
five-seed effect, bootstrap, and continue gate can never be formed.

## Execution decision

- Do not run the current 3,600-run sequential-tiered prospective phase.
- Do not run the current 5,400-run cross-corner phase as a Stage-A prerequisite.
- Do not run the 36,000 arm-conversation Stage-A quality matrix.
- Do not build Stage-B fixed-memory prerequisites or run Stage B.
- Keep the legacy 16-arm factorial paused; this audit does not replace it.

The implementation now fails before GPU allocation on the terminal structural
NO-GO. It also gates any future quality unlock on exact paired measured tier hot
blocks, tier hot bytes, total hot-resident bytes, pin/fallback digests, budget
integrity, and per-protected-coordinate pin exposure.

## Next research move

The safest next step is a new, outcome-independent calibration study before any
807-series execution. The current final quotas are informative in only 10/20
cells, while their target-free score-demand ranks are informative in 16/20.
We will test whether richer continuous 707-only layer summaries resolve the four
fully tied cells. Only then can a new versioned contrast be frozen.

If rank information becomes complete, a separate mechanistic probe may compare
a mean-preserving heterogeneous quota template maximally aligned versus
anti-aligned with the frozen calibration ranks. That would estimate calibrated
rank sensitivity, not performance of the original calibrated quota magnitudes,
and it cannot automatically unlock the old Stage B. If complete ties persist,
they remain structural-zero controls or require new independently trained seeds;
arbitrary layer-ID or digest tie-breaking will not be described as calibration
evidence.

## Claim boundary

These results do not estimate adaptive-quota quality, a protected-pin effect,
population path equivalence, natural-language transfer, larger-model transfer,
latency, throughput, HBM capacity, or production benefit. The positive result
is narrower: the execution-path failure was localized, and a large but
non-identifying experiment was prevented before outcome access.
