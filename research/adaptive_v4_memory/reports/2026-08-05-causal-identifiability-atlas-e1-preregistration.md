# Causal Identifiability Atlas E1 preregistration

Date: 2026-08-05

Status: **design fixed before implementation and before any E1 outcome is read**

## Why this experiment comes before a learned router

The 180-cell H0 pilot showed that replacing a matched control with the known
task-evidence block improves the model. It established evidence sufficiency and
an oracle headroom, but it did not establish three facts a deployable router
needs:

1. the causally useful block is stable across independently trained models;
2. its utility is observable from inference-time state without the synthetic
   evidence position or answer; and
3. acting on an observable score improves the end task at the same route
   cardinality.

Training a reranker before answering these questions can manufacture a positive
result from labels that are unstable, unobservable, or already encoded by the
short-context evidence-position supervision used to train the base selector.
E1 is therefore a no-training identifiability audit. It adds no model parameter
and makes no natural-language or production claim.

## Frozen scientific question

Can an answer-free, evidence-position-free output-perturbation proxy identify
equal-budget causal block utility consistently enough to improve native CSA
routing under length shift?

E1 keeps four objects separate:

- **task evidence**: the synthetic generator's evidence block, opened only
  after observable score tables and routes have been constructed and hashed;
- **necessity**: loss in full-compressed-memory target log probability when one
  block is deleted;
- **equal-budget utility**: gain when one candidate is added to the same
  `K-1` core; and
- **observable score**: a quantity available from the frozen model and current
  query without the answer or evidence position.

This separation follows the causal-evidence-set distinction between annotated
evidence, necessary blocks, and sufficient sets, and uses same-cardinality route
replay rather than attention-map imitation.

## Fixed 160-cell design

All checkpoints are the five post-rank-direct checkpoints with training seeds
6071406--6071410. Evaluation workload seeds depend only on family, context, and
replicate: they deliberately exclude training seed and scale. Thus every
checkpoint sees byte-identical inputs for a shared workload coordinate, making
checkpoint-seed stability identifiable.

| Cohort | Checkpoints | Contexts | Families | Replicates | Budget | Cells |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| s151 stress/shift | 5 | 640, 1024 | 4 | 3 | native 1x (`K=1`) | 120 |
| s55 saturation control | 5 | 640 | 4 | 2 | fixed 2x (`K=4`) | 40 |
| **Total** |  |  |  |  |  | **160** |

The four families are `single-remote-retrieval`,
`adversarial-lexical-distractors`, `long-generation-changing-evidence`, and
`multiple-independent-needles`.

Every cell evaluates native, bitwise identity replay, evidence oracle, full
read score, output-perturbation proxy, recency, value norm, and a deterministic
random control. All non-native policy routes select exactly `K` blocks per CSA
layer/query. The evidence oracle is evaluation-only and is never an observable
policy.

## Observable scores

The frozen comparators are:

- native CSA index score;
- full compressed-memory read score, computed before the sparse route is
  applied;
- recency (block end position);
- compressed-value L2 norm;
- a deterministic SHA-256 random ordering shared across checkpoint seeds; and
- a CSA-adapted CriticalKV-style output-perturbation proxy.

For candidate block `i`, the primary proxy is

`I_i = || W_O ( p_i / (1 - p_i) * (v_i - o) ) ||_2`.

Here `p_i` is the softmax of the full compressed read scores, `v_i` is the
rotary-positioned compressed value, `o` is their probability-weighted mean, and
`W_O` is the frozen attention output projection. The same per-head delta is
expanded through the model's grouped and final output projections. The proxy
is explicitly approximate: it excludes local/HCA normalization mass and
downstream nonlinear propagation. That limitation is part of what E1 tests.

## Exhaustive causal atlas

The 60 s151 single-query cells (two single-query families x two contexts x
three replicates x five checkpoint seeds) evaluate every causal compressed
block in every CSA layer.

Because s151 uses `K=1`, equal-budget utility is measured by comparing a
one-candidate route with the empty `K-1` core at the target layer/query while
all other routes remain native. Necessity compares full compressed memory at
that layer/query with full memory minus the candidate. Candidate interventions
may be batched, but each effect is compared with its baseline in the identical
batch shape and kernel path.

Gold-target log probability is primary. A secondary annotation-free readout
uses the full-memory model's own predicted token as the teacher token. The
generator evidence position is attached only after the observable candidate
table and route bundle have been hashed.

## Integrity gates

Before quality analysis:

- exactly 160 closed-world cells and exactly 60 exhaustive-atlas cells;
- checkpoint byte count and SHA-256 match the manifest;
- identical input digest across training seeds for each shared workload;
- native and identity-replay full logits are bitwise equal in every cell;
- every observable and oracle policy route has exact cardinality `K`;
- counterfactual candidate/core and full/deletion receipts satisfy their
  declared cardinalities; and
- observable route digests are committed before evidence-oracle construction.

Any failure invalidates the affected run; it is not repaired by dropping a
seed or coordinate.

## Primary gates and estimands

Each query is averaged inside its coordinate, families are equally weighted
inside a checkpoint seed, and checkpoint seed is the independent unit.
Confidence intervals use 200,000 deterministic checkpoint-cluster bootstrap
draws. E1 is a conjunctive diagnostic gate, not a null-hypothesis significance
claim.

1. **Causal topology.** For every exhaustive workload/layer stratum, take the
   single block with maximum gold equal-budget utility in each checkpoint. The
   mean pairwise singleton-set Jaccard across the five seeds must be at least
   0.60.
2. **Observable alignment.** Within each exhaustive workload/layer stratum,
   compute Spearman correlation between the perturbation score and gold
   equal-budget utility. The equally weighted mean correlation must be at least
   0.30 in all five checkpoint seeds.
3. **s151 640 end-task recovery.** Perturbation-proxy minus native target log
   probability must be positive in all five seeds. The ratio of its mean gain
   to evidence-oracle minus native headroom must be at least 25%, and the
   checkpoint-bootstrap 95% lower bound of that recovered fraction must be at
   least 10%.
4. **1024 length shift.** Perturbation-proxy minus native target log probability
   must be positive in at least four of five seeds and in at least three of four
   families.
5. **s55 2x safety.** For perturbation-proxy minus native, the checkpoint-
   bootstrap 95% lower bounds must exceed -0.10 target log probability and
   -2.0 percentage points accuracy.

Full-read, recency, value-norm, random, necessity, task-evidence overlap, and
teacher-token analyses are prespecified diagnostics. They cannot substitute
for a failed primary perturbation-proxy gate.

## Decision tree

- If all gates pass, skip learned synthetic-router training and move to a new,
  small natural-language transfer pilot.
- If causal topology is stable but observable alignment or end-task recovery
  fails, E1 labels may justify a separately approved nested-checkpoint-CV
  reranker pilot; E1 itself does not train it.
- If causal topology is unstable, do not train a shared selector.
- If task evidence is sufficient but weakly necessary, prioritize integration
  or residual restoration over selection.
- If no inference-time observable score tracks utility, pivot to
  restoration/reconstruction (IndexMem/RestoreKV/ResKV route) rather than
  enlarging the synthetic matrix.

No gate may be changed after the first E1 quality value is accessed. The
implementation, analyzer, checkpoint inventory, and this preregistration will
be committed first; the manifest will then be frozen in a separate
manifest-only commit before the one-cell invariant preflight and the 160-cell
run.
