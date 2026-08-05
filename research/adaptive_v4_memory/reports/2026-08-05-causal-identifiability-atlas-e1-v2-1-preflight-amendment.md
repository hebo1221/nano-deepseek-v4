# Causal Identifiability Atlas E1 v2.1 preflight amendment

Date: 2026-08-05

Decision: **correct a preflight-only row-count defect and require both the first
and formerly failing coordinates to pass before launch.**

V2 padded the final partial candidate chunk to batch 8, and its first frozen
preflight passed. The targeted preflight for the exact v1 failure coordinate
then failed before publishing any cell with:

`RuntimeError: Repeated full-memory rows diverged inside a batch.`

A quality-blind structural reproduction showed that identical batch-8
full-memory forwards were bitwise equal at all five CSA layers. The defect was
in plan construction: candidate and deletion plans received all eight padded
rows, while core and full plans still multiplied their repeated rows by the
seven real candidates. Their eighth batch row therefore fell back to native
routing.

V2.1 constructs candidate, core, full, and deletion row tuples together in one
pure helper and tests that every tuple has the frozen batch size and declared
cardinality. No outcome, coordinate, seed, route policy, target, metric, or gate
changes. V2 produced zero result cells and its output root remains empty. V2.1
uses a new experiment ID, manifest, and output root; neither v1 nor v2 cells are
reused.
