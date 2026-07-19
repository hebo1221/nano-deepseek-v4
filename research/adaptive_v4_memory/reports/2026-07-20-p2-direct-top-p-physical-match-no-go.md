# P2 direct top-p physical-match decision

Date: 2026-07-20

Decision: **terminal prerequisite NO-GO; held-out controller quality unrun**

## What ran

The frozen revision-1.2 calibration-only physical-match matrix completed all
40 registered coordinates: two scales, five training seeds, two budgets, and
the protected-pinned comparator arms `fixed-top-p-0.5+pins` and
`fixed-top-p-0.8+pins`. Outcome-dependent stopping was disabled, so every
coordinate ran after the first NO-GO was known. The canonical runner finished
with exit code 2, the registered code for a terminal NO-GO artifact.

The run used clean source commit
`8c88464d3de39dd98a119ec98cef99a5f7a8c0f5`, manifest SHA-256
`eab8e67d9e2d0e162aaf19a637a2978a1a41d71c799570aecfec3c31499754d7`,
and implementation digest
`2978634d0c6da3e45d8a97a494f5a319778653ebd7c9a774f44ff72cee238b96`.
The terminal v1.1 ten-cell training prerequisite reused by v1.2 and the
disjoint revision-1.2 calibration matrix were revalidated before, during, and
after execution. The calibration matrix was terminal 10/10 GO. Every top-p
observation used the calibration-only generation rule and recorded
`evaluation_seed_accessed=false`.

## Authenticated terminal evidence

The authoritative ledger is
`artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/top_p_physical_match/top-p-physical-matrix.summary.json`.

- Status: `terminal`; decision: `NO-GO`
- Completed: 40/40; GO: 0; NO-GO: 40
- File bytes: 158,900
- File SHA-256:
  `ad4b5d2dcdcbc50ccd5008a0927cfb1a3265f904740e22f6ebab23c2b8f40c41`
- Payload SHA-256:
  `e4881c11ff7508d920746059e26be3dfb00e142fb2bc891c79b8e84ff6f8fd09`
- Attestation payload SHA-256:
  `6bb0dfcb7ecb073e5609a5ccf7a8114b728241e9b76efdc357c5e777e835d95b`
- HMAC:
  `8190a5a0bb955b21099c1527ea70a6def79bcda6718ee407f7fd41cbe0b0c750`
- Key ID:
  `67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf`
- Coordinate digest:
  `ecde0f6b82f368dc78f66b0349e8395ca3f9c8345de716985bd66532aa648bb4`

The raw key remained outside the repository and artifact roots, with mode
0600. Its bytes were not placed in a command, log, or artifact.

## Cell results

The two byte columns are sums across the same 450 paired calibration
observations, not instantaneous HBM footprints. Comparing their sums is exactly
equivalent to comparing their means because the observation counts are equal.
`Difference` is the absolute difference between those totals divided by the
exact-fill target total. The frozen GO boundary was at most 1%. Every table arm
includes protected pins; 0.5 and 0.8 abbreviate the two full comparator names
listed above.

| Scale | Training seed | Budget | top-p | Target total (450 obs) | Comparator total (450 obs) | Difference | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| S55 | 6071406 | 2x | 0.5 | 734,400 | 187,816 | 74.43% | NO-GO |
| S55 | 6071406 | 2x | 0.8 | 734,400 | 288,456 | 60.72% | NO-GO |
| S55 | 6071406 | 4x | 0.5 | 1,468,800 | 187,816 | 87.21% | NO-GO |
| S55 | 6071406 | 4x | 0.8 | 1,468,800 | 290,360 | 80.23% | NO-GO |
| S55 | 6071407 | 2x | 0.5 | 734,400 | 233,376 | 68.22% | NO-GO |
| S55 | 6071407 | 2x | 0.8 | 734,400 | 380,936 | 48.13% | NO-GO |
| S55 | 6071407 | 4x | 0.5 | 1,468,800 | 241,536 | 83.56% | NO-GO |
| S55 | 6071407 | 4x | 0.8 | 1,468,800 | 431,120 | 70.65% | NO-GO |
| S55 | 6071408 | 2x | 0.5 | 734,400 | 234,736 | 68.04% | NO-GO |
| S55 | 6071408 | 2x | 0.8 | 734,400 | 408,544 | 44.37% | NO-GO |
| S55 | 6071408 | 4x | 0.5 | 1,468,800 | 236,232 | 83.92% | NO-GO |
| S55 | 6071408 | 4x | 0.8 | 1,468,800 | 447,712 | 69.52% | NO-GO |
| S55 | 6071409 | 2x | 0.5 | 734,400 | 263,024 | 64.19% | NO-GO |
| S55 | 6071409 | 2x | 0.8 | 734,400 | 379,984 | 48.26% | NO-GO |
| S55 | 6071409 | 4x | 0.5 | 1,468,800 | 263,024 | 82.09% | NO-GO |
| S55 | 6071409 | 4x | 0.8 | 1,468,800 | 380,120 | 74.12% | NO-GO |
| S55 | 6071410 | 2x | 0.5 | 734,400 | 335,104 | 54.37% | NO-GO |
| S55 | 6071410 | 2x | 0.8 | 734,400 | 475,048 | 35.31% | NO-GO |
| S55 | 6071410 | 4x | 0.5 | 1,468,800 | 463,760 | 68.43% | NO-GO |
| S55 | 6071410 | 4x | 0.8 | 1,468,800 | 650,896 | 55.69% | NO-GO |
| S151 | 6071406 | 2x | 0.5 | 612,000 | 358,088 | 41.49% | NO-GO |
| S151 | 6071406 | 2x | 0.8 | 612,000 | 543,864 | 11.13% | NO-GO |
| S151 | 6071406 | 4x | 0.5 | 1,224,000 | 405,688 | 66.86% | NO-GO |
| S151 | 6071406 | 4x | 0.8 | 1,224,000 | 625,192 | 48.92% | NO-GO |
| S151 | 6071407 | 2x | 0.5 | 612,000 | 342,856 | 43.98% | NO-GO |
| S151 | 6071407 | 2x | 0.8 | 612,000 | 534,616 | 12.64% | NO-GO |
| S151 | 6071407 | 4x | 0.5 | 1,224,000 | 358,632 | 70.70% | NO-GO |
| S151 | 6071407 | 4x | 0.8 | 1,224,000 | 573,648 | 53.13% | NO-GO |
| S151 | 6071408 | 2x | 0.5 | 612,000 | 376,992 | 38.40% | NO-GO |
| S151 | 6071408 | 2x | 0.8 | 612,000 | 566,032 | 7.51% | NO-GO |
| S151 | 6071408 | 4x | 0.5 | 1,224,000 | 433,160 | 64.61% | NO-GO |
| S151 | 6071408 | 4x | 0.8 | 1,224,000 | 664,632 | 45.70% | NO-GO |
| S151 | 6071409 | 2x | 0.5 | 612,000 | 358,360 | 41.44% | NO-GO |
| S151 | 6071409 | 2x | 0.8 | 612,000 | 559,776 | 8.53% | NO-GO |
| S151 | 6071409 | 4x | 0.5 | 1,224,000 | 359,176 | 70.66% | NO-GO |
| S151 | 6071409 | 4x | 0.8 | 1,224,000 | 607,784 | 50.34% | NO-GO |
| S151 | 6071410 | 2x | 0.5 | 612,000 | 412,080 | 32.67% | NO-GO |
| S151 | 6071410 | 2x | 0.8 | 612,000 | 580,312 | 5.18% | NO-GO |
| S151 | 6071410 | 4x | 0.5 | 1,224,000 | 468,928 | 61.69% | NO-GO |
| S151 | 6071410 | 4x | 0.8 | 1,224,000 | 728,280 | 40.50% | NO-GO |

The closest coordinate was
S151/6071410/2x/`fixed-top-p-0.8+pins` at 5.1778%,
still 5.18 times the allowed difference. The farthest was
S55/6071406/4x/`fixed-top-p-0.5+pins` at 87.2130%.

## Mechanistic diagnosis

This is a failure of the frozen comparator-feasibility assumption, not evidence
of a validator or budget-plumbing defect. Hsoft uses exact fill, so its target
hot bytes scale exactly with the 2x and 4x block budgets. The registered top-p
comparator is deliberately variable-cardinality and never exact-filled or
converted to top-k. Its scheduled cap is only an upper bound: when the fixed
score-mass threshold is reached below that cap, increasing the budget does not
force more blocks into hot memory.

For example, the 450 observations in
S55/6071406/`fixed-top-p-0.5+pins` totaled 187,816 comparator bytes at both
2x and 4x, while the exact-fill target total doubled from 734,400 to 1,468,800
bytes. All 40 artifacts verified the frozen cap search, tensor/device
accounting, raw-observation replay, calibration-only scope, and external
bindings. The observed mismatch is therefore the result the registered
contract asks the matrix to measure.

## Downstream decision

Revision 1.2 required all 40 physical-match cells to be terminal GO before any
held-out controller shard could launch. That prerequisite failed, so the
canonical `controller/` output root and its sibling worker-ledger root remain
absent. No 10071406--10071410 evaluation seed was accessed through this
pipeline. The 9,000-shard, 19-arm controller-quality matrix, integrity audit,
and statistical summary are **unrun**, not failed quality experiments.

The correct bounded conclusion is:

> Under the frozen cap-only schedule, the protected-pinned
> `fixed-top-p-0.5+pins` and `fixed-top-p-0.8+pins` variable-cardinality
> comparators did not match the exact-fill Hsoft target within 1%
> hot-resident bytes in any of the 40 tested calibration cells.

This result does not compare Hsoft quality with fixed, balanced-fixed, or top-p
quality. It does not estimate the value of adaptive quota, protected pins,
local or hierarchical control, any causal ablation, natural-language transfer,
large-model transfer, latency, throughput, transfer traffic, or total HBM.
It does not establish a general impossibility result for top-p matching.

## Integrity audit

An independent read-only audit found no discrepancy:

- exact 2 x 5 x 2 x 2 Cartesian coverage, with no duplicate or missing cell;
- 40/40 canonical public-validator replays passed;
- all raw physical-byte summaries and artifact HMACs recomputed exactly;
- exactly 40 cell artifacts plus one ledger in the closed-world root;
- zero claim, temporary, orphan, extra, or symlink entries;
- the controller and worker-ledger roots absent;
- training, calibration, manifest, source, and quarantine hashes unchanged.

The bound training-ledger SHA-256 remains
`786669b8feb74eef5a4aa1e57dccc3ffada10596a8ae995d78e931daeef06cb5`;
the calibration-ledger SHA-256 remains
`b6da3a7861ac0a7d3d3d94cec9b6031a40270f01ff5eef8a1513761d97cc7b61`.
The superseded calibration matrix, preserved claim, and preserved artifact
remain respectively
`f0dccaa9861e095b297c22a17735b3a379d4f9ed8db628e0bcf222b12da5e426`,
`462793153ad22a19223398ef30c2e4624ae247d179a8c29cb250dca1b3bca2cb`,
and `f805d70cb1379cc71c6d6abbe34d579880bd8cec35b72ea816ce9cbc7775d25b`.

## Follow-up boundary

Revision 1.2 is not repaired or relabeled after observing this result. A future
study may preserve these top-p cells as descriptive feasibility evidence while
placing the unchanged exact-fill arms in a separate, newly versioned quality
cohort. Such a study must use a new manifest, experiment ID, attestation
purposes, and output root; disclose that the v1.2 top-p result was observed;
and describe itself as prospective only with respect to still-unobserved
held-out controller quality.
