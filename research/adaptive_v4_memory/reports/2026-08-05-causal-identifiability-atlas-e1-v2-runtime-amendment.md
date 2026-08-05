# Causal Identifiability Atlas E1 v2 runtime amendment

Date: 2026-08-05

Decision: **quarantine the 65-cell v1 prefix without reading quality, fix the
counterfactual batch shape, and restart in a fresh v2 namespace.**

## Observed integrity failure

The frozen v1 service completed 65 atomic cells and then stopped on s151 seed
6071408, context 1024, adversarial replicate 2. The failure occurred before
publishing that cell:

`RuntimeError: Full-memory teacher token changed across candidate chunks.`

No v1 outcome JSON was opened or summarized. Monitoring accessed only cell
counts, service state, restart count, file sizes, and the exception traceback.
The 65 completed cells remain immutable under the v1 output root and are not
admitted into v2.

The failure was not OOM, checkpoint drift, route-cardinality drift, or an
identity-replay failure. A structural diagnostic found 255 causal candidates in
each CSA layer at the failing query. With frozen counterfactual batch size 8,
the first 31 chunks used batch 8 while the final chunk used batch 7. Each
candidate effect still had a same-shape baseline, but the secondary full-memory
teacher argmax was compared across two GEMM batch shapes. A near-tied argmax
changed at that boundary and triggered the deliberately strict check.

## Minimal correction

V2 repeats the final real candidate until every counterfactual forward has
exact batch size 8. Results belonging to repeated padding rows are discarded.
Thus:

- every real candidate is still evaluated exactly once;
- candidate versus empty-core and full versus deletion comparisons use the
  identical batch and kernel shape;
- the full-memory teacher token is now defined on one fixed execution shape;
- route cardinality for every real candidate is unchanged; and
- no target, evidence position, score, arm, metric, seed, coordinate, or gate is
  changed.

The padding rule is deterministic and outcome-independent. Because it can alter
floating-point results for the old final partial chunks, v1 cells are not
reused. V2 receives a new experiment ID, manifest, output root, and manifest
commit. The original preregistration remains authoritative except for this
fixed-batch execution amendment.
