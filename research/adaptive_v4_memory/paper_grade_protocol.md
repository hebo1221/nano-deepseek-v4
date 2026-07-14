# Paper-grade expansion protocol

Protocol version: 1.1
Frozen: 2026-07-14  
Amended: 2026-07-14, before the full P2 matrix or causal-ablation analysis
Status: active; P2 synthetic core runs first, followed by causal ablations,
natural-language evaluation, and systems evaluation

This amendment supersedes the experimental scale of the M5 pilot. It does not
erase the pilot result: it narrows that result to the exact one-token global M2
interface and preregisters the evidence required for broader claims.

## 1. Primary questions

1. Can a causal same-token controller beat the strongest memory-matched fixed
   policy without losing long-context quality?
2. Does hierarchical pre-allocation recover useful cross-layer coordination
   without requiring future-layer signals?
3. Does physical non-uniform residency improve end-to-end serving rather than
   merely reducing one cache component?
4. After protected pins are held constant, does adaptive quota allocation add
   quality beyond the benefit of pinning alone at the same measured hot-memory
   footprint?

The primary comparison is same-token hierarchical control versus the strongest
fixed policy selected on calibration data. One-token global M2 is a negative
baseline, not the primary adaptive method.

## 2. Frozen methods

Required arms:

1. native V4 selection with resident cache;
2. native V4 selection with tiered values;
3. dense selection;
4. fixed top-k and fixed top-p budget sweeps;
5. offline non-causal M2 oracle/upper bound;
6. one-token global M2 pilot implementation;
7. same-token layer-local training-free controller; and
8. same-token hierarchical controller with calibration-only layer quotas.

A learned lookahead predictor is exploratory until the training-free arms are
frozen. It must use disjoint train, calibration, and test traces and may not be
substituted for a failed primary arm after test results are observed.

The primary causal factorial contains, at every eligible budget point:

1. fixed allocation without protected pins (`fixed`);
2. fixed allocation with protected pins (`fixed+pins`);
3. calibrated allocation without protected pins (`calibrated-no-pins`);
4. calibrated allocation with protected pins (`calibrated+pins`);
5. calibrated quotas shuffled across eligible layers, evaluated both without
   and with the same pin policy (`shuffled-quota[-pins]`); and
6. layer-local and hierarchical calibrated allocation with identical pin and
   fallback settings.

This factorial is run only after the already-frozen P2 synthetic core finishes;
the core matrix is not modified or restarted in response to ablation results.
Additional one-component ablations remove score concentration, temporal reuse,
cross-layer prior, refresh reuse, protected pins, and dense fallback. Every arm
uses the same model checkpoint, generated examples, dtype, execution order, and
paired conversations.

The enforced causal execution chain is:

1. validate all 4,500 P2 core shards and their dependency/record digests;
2. on 707-series data only, measure 20 conversations per family and context and
   freeze a fixed low/high schedule whose mean physical hot-resident bytes are
   within 1% of `calibrated+pins` for both 2x and 4x at every seed and scale;
3. for every one of the 10 seed-scale checkpoints, require exact prediction and
   budget equivalence between chunked-quality and sequential physical-tier
   paths across all 14 arms, 2 budgets, 9 families, and 5 contexts; and
4. run the 9,000 held-out factorial shards (2,520,000 quality arm-conversations
   plus 360,000 direct physical arm-conversations).

No stage may consume 807-series quality to tune the fixed mixture. A failed
calibration-memory cell or equivalence audit blocks held-out execution for that
cell instead of permitting a post-hoc repair.

## 3. Models and seeds

Tier-S scales: S55 and S151.

Independent training seeds:

```text
6071401, 6071402, 6071403, 6071404, 6071405
```

Data-order and initialization RNGs are derived from the training seed and
recorded separately. A failed training run is rerun only for a documented
infrastructure error; a low-quality completed run remains in the seed-level
analysis.

Evaluation generation seeds:

```text
8071401, 8071402, 8071403, 8071404, 8071405
```

Each model seed is paired with the evaluation seed at the same index. Additional
diagnostic seeds may be run but cannot replace these five.

Controller quota-calibration seeds:

```text
7071401, 7071402, 7071403, 7071404, 7071405
```

Each calibration seed is paired with the model seed at the same index and may
use only score-derived requested-block counts and candidate counts. It cannot
use targets, native correctness, or the 807-series held-out examples. Layer
quotas use the nearest-rank 95th percentile and retain the fixed-policy minimum
in every CSA layer. The 1x point remains uniform; non-uniform allocation is
tested only when 2x or 4x capacity leaves room above that floor.
The score-derived uncertainty allowance is fixed to
`minimum_per_layer * (budget_multiplier - 1)`, so 1x, 2x, and 4x cannot collapse
to the same effective method merely because of a constant request cap.

## 4. Tier-S workloads and sample size

The core test set contains at least 1,000 independent examples or conversations
per model seed, scale, and workload family:

1. single remote retrieval;
2. multiple independent needles;
3. associative recall;
4. multi-turn query shift;
5. dense global aggregation;
6. irrelevant-context/local-only queries;
7. instruction persistence and protected-prefix recall;
8. adversarial lexical distractors; and
9. long generation with changing relevant evidence.

Context-length grid:

```text
80, 128, 256, 512, 1024 tokens
```

Every family must cover at least three grid points; retrieval, irrelevant
context, query shift, and dense aggregation cover all five. Results are reported
per family and length. A macro average is secondary and cannot hide a failed
slice.

Hot-budget sweeps use total selected CSA blocks, normalized by the number of CSA
layers:

```text
1x, 2x, 4x minimum-per-layer budget, native top-k, dense
```

Memory-matched policies are compared using measured hot-resident bytes, not
configuration labels alone. A causal pair is considered memory matched only
when its mean measured hot-resident bytes differ by at most 1%. If the initial
pair misses this tolerance, fixed top-k is retuned on calibration traces and
the held-out pair is rerun; post-hoc accuracy interpolation is forbidden.

## 5. Data isolation

Generator configurations and task templates are fixed before checkpoint
selection. Training, controller training, calibration, and test examples use
disjoint seed namespaces. Test targets, native correctness, and oracle labels
are unavailable to budget calibration.

The paired evaluation unit is an example for single-turn tasks and a complete
conversation for multi-turn tasks. Turns from one conversation cannot cross
bootstrap samples or data splits.

## 6. Statistics

Primary quality metrics are exact match, per-turn exact match, and worst-family
absolute regression against native. Memory and runtime are evaluated only at
quality-qualified operating points.

For every policy difference:

- report seed-level values and the mean, standard deviation, and range;
- compute a paired cluster bootstrap 95% confidence interval with 10,000
  resamples at the example/conversation level;
- report the paired absolute effect and relative effect where defined;
- apply Holm-Bonferroni correction across primary workload families;
- report the worst context-length and worst seed slice; and
- retain failures and timeouts in an explicit accounting table.

A controller passes the primary quality gate only if its mean regression from
native is at most 1 percentage point, no primary family regresses by more than
2 points, and the lower confidence bound of its improvement over the strongest
fixed policy is non-negative on at least two families at both scales.

The central causal claim has a separate, stricter gate. At the same measured
hot-memory footprint, `calibrated+pins` must beat `fixed+pins` on both S55 and
S151: the pooled paired effect must be positive, its 95% cluster-bootstrap lower
bound must be greater than zero after the preregistered family correction, and
all five seed-level effects must be positive at each scale. Any failed clause is
reported as a failed or bounded causal claim rather than averaged away. Every
clause is required separately at both 2x and 4x; one budget cannot rescue the
other. The
contrasts `fixed+pins - fixed` and `calibrated+pins - calibrated-no-pins`
estimate the pin contribution; `calibrated-no-pins - fixed` and
`calibrated+pins - fixed+pins` estimate adaptive-quota contribution; shuffled
quotas test whether layer identity, rather than merely non-uniformity, matters.

## 7. Natural-language evaluation

The minimum publishable natural evaluation is one complete established suite,
not selected tasks. The execution order is:

1. RULER at supported 8K, 16K, 32K, 64K, and 128K lengths;
2. SCBench in both shared-context modes;
3. LongBench v2;
4. LongMemEval; and
5. MRCR or an equivalent dense multi-range retrieval suite.

Dataset commit/version, license, preprocessing, tokenizer, prompt template, and
scorer digest are pinned. Unsupported context lengths are marked unsupported,
not silently truncated.

Official DeepSeek-V4 Flash evaluation is a separate evidence tier. If the
pinned checkpoint and supported runtime cannot be provisioned, the report must
provide the exact revision, launch command, minimum accelerator/storage
requirement, and estimated cost while withholding official-scale claims.

Natural-language comparisons include native/dense, strongest memory-matched
fixed, `fixed+pins`, and any synthetic-qualified calibrated arm. FlashMemory-
and IndexCache-family baselines are included when their pinned implementations
support the selected model/runtime; incompatibility is recorded explicitly and
never replaced by a projected number.

## 8. Systems matrix

At every quality-qualified policy point, measure:

```text
context:     8K, 32K, 128K, and 500K when supported
batch:       1, 4, 8, 16
concurrency: 1, 8, 32
generation:  128, 512, 2048 tokens
```

Each cell has at least five untimed warmups and 30 timed repetitions. Report
TTFT, TPOT, p50/p95/p99, throughput, allocated/reserved HBM, peak HBM, pinned
host bytes, fragmentation, H2D/D2H bytes, useful transfer ratio, misses, late
misses, controller time, indexer time, and total device HBM at process and
system level. OOM, timeout, numerical failure, late transfer, and tail-latency
outliers are retained in an explicit tail-failure table.

Reference PyTorch and fused production runtimes are separate result tables.
Projected kernel speedups are never mixed with measured results.

## 9. Evidence ladder and execution order

The proof is deliberately sequential:

1. finish the frozen synthetic P2 core without changing its arms;
2. separate pin and adaptive-quota causality with the factorial above;
3. transfer qualified arms from synthetic tasks to the full natural suites;
4. transfer conclusions from S55/S151 to the largest available compatible
   model and, separately, official V4 Flash when provisioned; and
5. replace logical block budgets with measured physical HBM, transfer traffic,
   latency, throughput, and tail-failure evidence.

Each rung may falsify transfer from the previous rung. Synthetic success is not
described as natural-language success, small-model success is not described as
large-model success, and logical budget matching is not described as physical
HBM equivalence.

## 10. Early stopping

An individual controller arm may stop after the first scale only when all five
seeds show more than 10 percentage points of quality regression in every
primary family. The failure remains a reported result. This rule saves compute
but does not permit replacing the arm or changing thresholds.

Natural-language and systems evaluation may skip a failed controller, but must
still run native and the strongest fixed/tiered baseline. Thus a controller
failure cannot terminate the broader cache-systems study.

## 11. Claim boundary

The existing M5 evidence supports only this statement:

> The tested one-token, cross-layer global M2 controller is not quality-safe on
> the two Tier-S checkpoints and three synthetic pilot workloads.

It does not establish that adaptive residency, learned lookahead, same-token
control, official V4, or natural-language workloads fail. A paper-level claim
requires the sample sizes, independent seeds, natural suite, and system matrix
defined above.

## 12. Reproducibility gate

Every result table and figure must be generated from a digest-bound artifact.
Manifests record source commit, dirty state, checkpoint and dataset hashes,
complete policy configuration, environment, seeds, warmup/repetition counts,
and exclusions. Raw private prompts and large checkpoints remain external.

The study is complete only when the required Tier-S matrix, at least one full
natural benchmark suite, quality-qualified system measurements, statistical
analysis, tests, and reproduction instructions are committed and pushed.
