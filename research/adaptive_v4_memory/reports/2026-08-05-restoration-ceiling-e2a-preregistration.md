# Restoration Ceiling E2-A preregistration

Date: 2026-08-05

Status: **design fixed before implementation and before any E2-A outcome is
read**

## Why this experiment comes before a residual learner

The Causal Identifiability Atlas E1 rejected a shared hard selector: the
maximum equal-budget utility block had only 0.3217 mean cross-checkpoint
singleton Jaccard, and the observable perturbation proxy failed its alignment
and s151/640 recovery gates. E1 nevertheless preserved the H0 result that a
known task-evidence route creates large positive headroom.

That combination does not yet establish that restoration is learnable. Both
IndexMem and RestoreKV train a compact mechanism toward information or behavior
available from a fuller cache. If end-to-end full compressed memory is not a
stable positive teacher in this model, training a residual module to imitate it
would spend compute on the wrong target. E2-A is therefore a no-training
restoration-ceiling audit. It changes no model parameter and does not implement
or claim a deployable cache.

## Frozen scientific question

Does end-to-end access to every causal compressed-memory block provide a
checkpoint-stable task advantage over native sparse routing, and does it retain
information that is not recovered merely by expanding the native index route
to eight blocks?

E2-A separates three ideas:

- **evidence sufficiency**: the evaluation-only task-evidence route inherited
  from H0/E1;
- **hard-route capacity**: answer-free native index routes with absolute
  cardinality 1, 2, 4, or 8; and
- **restoration ceiling**: standard attention over every causal compressed
  block at each workload query in every CSA layer.

The full-compressed arm is a scientific teacher ceiling, not a systems method.
It is not matched for HBM, transfer, or query-time bandwidth and cannot support
a production-efficiency claim.

## Frozen 160-cell design

The experiment uses the same ten post-rank-direct checkpoints as E1 but a fresh
evaluation-seed namespace. Workload seeds depend only on family, context, and
replicate and exclude checkpoint seed and model scale. No E1 workload outcome
is reused.

| Cohort | Checkpoints | Contexts | Families | Replicates | Native budget | Cells |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| s151 stress/shift | 5 | 640, 1024 | 4 | 3 | 1 block | 120 |
| s55 saturation control | 5 | 640 | 4 | 2 | 4 blocks | 40 |
| **Total** |  |  |  |  |  | **160** |

The four families remain `single-remote-retrieval`,
`adversarial-lexical-distractors`, `long-generation-changing-evidence`, and
`multiple-independent-needles`.

## Frozen arms

Every cell evaluates:

1. `native`;
2. `identity-replay` at the cohort's native cardinality;
3. `index-k1`;
4. `index-k2`;
5. `index-k4`;
6. `index-k8`;
7. `full-compressed`; and
8. `force-evidence` at the cohort's native cardinality.

The four index routes and full-compressed route are constructed and hashed
without accepting targets or evidence positions. Each index route uses the
frozen model's native CSA index score and selects exactly its declared absolute
cardinality at every CSA layer/workload query. Full-compressed selects every
finite causal candidate. Task-evidence positions are opened only after this
complete observable route bundle is hashed; the evidence block then replaces
the lowest-ranked native member when absent.

Native and identity replay must be bitwise equal. `index-k1` must also be
bitwise equal to native for s151, and `index-k4` must be bitwise equal to native
for s55. These equalities are integrity checks, not quality results.

## Integrity gates

Before quality analysis:

- exactly 160 closed-world cells;
- checkpoint byte counts and SHA-256 values match the manifest;
- identical input digest across checkpoint seeds and scales for every shared
  workload identity;
- native and identity-replay full logits are bitwise equal in every cell;
- the native-equivalent absolute-K replay is bitwise equal in every cell;
- index routes have exact cardinalities 1, 2, 4, and 8 with no duplicate or
  non-causal block;
- full-compressed routes contain every and only finite causal block;
- evidence routes retain the native cardinality; and
- the observable route digest is committed before evidence positions are
  accessed.

Any violation invalidates the affected run. No seed or coordinate may be
dropped after quality access.

## Primary estimands and gates

Queries are averaged inside a coordinate, families are equally weighted inside
a checkpoint seed, and checkpoint seed is the independent unit. Confidence
intervals use 200,000 deterministic checkpoint-cluster bootstrap draws. All
five gates are conjunctive.

1. **s151/640 full-teacher value.** `full-compressed - native` target log
   probability must be positive in all five checkpoint seeds and its 95%
   checkpoint-bootstrap lower bound must exceed 0.
2. **s151/1024 length shift.** `full-compressed - native` target log
   probability must be positive in at least four of five seeds and at least
   three of four families, with a 95% checkpoint-bootstrap lower bound above 0.
3. **Distributed information beyond K=8.** Across the equally weighted s151
   640/1024 panel, `full-compressed - index-k8` target log probability must be
   positive in at least four of five seeds and its 95% checkpoint-bootstrap
   lower bound must exceed 0.
4. **Evidence-headroom recovery.** At s151/640, the ratio of mean
   `full-compressed - native` gain to mean `force-evidence - native` gain must be
   at least 50%, and the checkpoint-bootstrap 95% lower bound of the paired
   ratio must be at least 25%.
5. **s55 2x safety.** For `full-compressed - native`, checkpoint-bootstrap 95%
   lower bounds must exceed -0.10 target log probability and -2.0 accuracy
   percentage points.

The absolute-K quality curve, family details, accuracy, evidence-route
headroom, and full-compressed behavior on long-generation workloads are
prespecified diagnostics. They cannot replace a failed primary gate.

## Frozen decision tree

- If all gates pass, request a separately frozen compact restoration experiment
  using a budget-matched residual state and full-compressed self-distillation.
- If full compressed memory is positive but does not beat K=8, prefer a small
  fixed-K capacity study over a restoration learner.
- If full compressed memory is positive overall but fails a family or safety
  condition, any later restoration mechanism must include a query-conditioned
  fallback gate and must first pass the failed regime.
- If the full-teacher value gates fail, do not train a full-cache restoration
  module on this synthetic model; pivot to model-agnostic external retrieval or
  early natural-language evidence indexing.

E2-A cannot authorize a learned module, a natural-language claim, or a systems
claim. No primary threshold may change after the first E2-A quality value is
accessed.

## Research basis

The ordering is motivated by
[IndexMem (arXiv:2605.25475)](https://arxiv.org/abs/2605.25475), which writes
evicted information into an online latent state and trains an explicit gated
residual readout, and
[RestoreKV (arXiv:2608.01247)](https://arxiv.org/abs/2608.01247), which learns a
budget-matched restore cache by self-distillation from full-cache behavior.
IndexMem also reports that injecting naive latent tokens inside softmax can be
brittle, which is why E2-A does not introduce an untrained mean-memory token.
