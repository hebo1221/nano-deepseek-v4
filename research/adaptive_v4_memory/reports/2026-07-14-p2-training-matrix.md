# P2 five-seed Tier-S training matrix

Date: 2026-07-14  
Decision: **training prerequisite complete; held-out P2 policy evaluation remains active**

## Scope

This result audits the preregistered five-seed training matrix. It is not a
controller comparison and cannot establish a quality-memory Pareto result. All
ten checkpoints will enter the frozen held-out evaluation; no seed was removed
for low accuracy.

The raw matrix is ignored at
`artifacts/adaptive_v4_memory/paper_grade/training-matrix.json` and is bound by
SHA-256 `853fa823ec681eb89710cf286e66a547fad7ef37c879d51a9dc67050fc4515d0`.
The checked audit summary is
`results/p2-training-matrix.summary.json` (SHA-256
`99aa09e8a25be04c6d3d32236d6a8d6728062430880524528151dc232a1307af`).

## Audit result

- 5 frozen initialization seeds × 2 scales = 10 unique runs;
- exactly 1,000 steps for every run;
- separate initialization, data-order, and training-evaluation seeds;
- one clean source commit for every run:
  `ff57ccadb513dba8b687ed113ad0eda338278b54`;
- all ten checkpoint byte counts and SHA-256 digests independently verified;
- zero low-quality exclusions and zero missing runs.

The trainer's `stopped_early` field is true for four runs because the target was
met at the mandatory 1,000th step. No run stopped before the frozen step count.

## Seed-level native validation

| Seed | S55 | S151 |
| ---: | ---: | ---: |
| 6071401 | 0.6615 | 0.6406 |
| 6071402 | 0.7344 | 0.8542 |
| 6071403 | 0.7552 | 0.8125 |
| 6071404 | 0.9167 | 0.8594 |
| 6071405 | 0.7031 | 0.9010 |
| Mean | 0.7542 | 0.8135 |
| Sample SD | 0.0975 | 0.1016 |
| Range | 0.2552 | 0.2604 |

The approximately ten-point standard deviations and 25--26 point ranges are
large. A one-seed study could therefore reverse the apparent scale comparison
or substantially overstate controller quality. The expanded matrix is not just
ceremonial replication; it measures a material source of uncertainty.

## Reproduction

```bash
.venv/bin/python \
  research/adaptive_v4_memory/scripts/run_p2_training_matrix.py

.venv/bin/python \
  research/adaptive_v4_memory/scripts/summarize_p2_training_matrix.py \
  --output research/adaptive_v4_memory/results/p2-training-matrix.summary.json
```

The second command rejects seed/scale drift, incomplete runs, dirty source,
metadata disagreement, missing checkpoints, and byte or digest mismatch.

## Next gate

The training prerequisite is complete. P2 remains incomplete until every
checkpoint is evaluated on the nine frozen workload families with at least
1,000 examples per family, the five evaluation seeds, the context and memory
budget grids, and paired uncertainty analysis from the preregistered protocol.
