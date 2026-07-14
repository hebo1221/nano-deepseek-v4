# Paper-grade expansion protocol

Protocol version: 1.9
Frozen: 2026-07-14  
Amended: 2026-07-14, before causal-factorial held-out execution
Status: active; P2 synthetic core runs first, followed by causal ablations,
natural-language evaluation, and systems evaluation

This amendment supersedes the experimental scale of the M5 pilot. It does not
erase the pilot result: it narrows that result to the exact one-token global M2
interface and preregisters the evidence required for broader claims.

Version 1.2 adds fixed top-p thresholds 0.5 and 0.8 to every held-out causal
factorial cell and freezes a target-aware offline upper bound that selects the
best complete-conversation result among all 16 registered arms. The oracle is
descriptive, non-causal, and prohibited from calibration, memory matching,
method selection, or the primary gate. It also replaces the marginal six-profile
P4 sweeps with the complete 4 batch × 3 load/concurrency factorial before any
systems cell is generated.

Version 1.3 changes execution scheduling only. The frozen examples, seeds,
arms, budgets, statistics, and implementation digests are unchanged. Disjoint
seed/family groups may run in three same-accelerator worker processes under one
exclusive study lock; every worker writes a distinct artifact path, and a
single parent verifies exact Cartesian coverage and raw digests before emitting
the canonical matrix. The online learned-lookahead runner additionally reuses
one immutable checkpoint load across shards from the same scale and seed.

Version 1.4 binds the preregistered four-seed confirmatory extension to
executable artifacts before primary causal outcomes or any extension outcome
is inspected. It changes no arm, workload, budget, sample count, or success
criterion. Each new checkpoint must independently pass calibration-only
physical hot-memory matching and all-16-arm sequential/chunked equivalence.
The 7,200-shard extension and 9,000-shard primary causal cohorts remain
separately reportable; a 16,200-shard nine-seed analysis is emitted only after
their frozen contracts and base evaluator digest are proven identical.

Version 1.5 corrects the analysis implementation and prose before any primary
or extension outcome summary is inspected. Family-level and cross-contrast
Holm-Bonferroni adjustments use the exact seed-cluster paired-randomization
p-values, while cluster-bootstrap intervals remain the effect-uncertainty
summary. It also replaces the stale count of 12 causal contrasts with all 15
implemented preregistered contrasts; no arm, example, seed, or gate changes.

Version 1.6 expands the pre-outcome Qwen3-1.7B RULER baseline screen from 39
to 57 cells by adding pinned PyramidKV and Ada-KV-wrapped SnapKV implementations
at the same 25%, 50%, and 75% compression ratios. This adds layer- and
head-adaptive comparators without changing datasets, model snapshots, selection
criterion, natural-suite outcomes, or P2 execution.

Version 1.7 closes a pre-execution provenance gap in the P3 harness. The
evaluator previously resolved from the pinned KVPress checkout while the press
package could resolve from site-packages. Every P3 runner now fails closed unless
both modules resolve inside the clean pinned checkout and records their resolved
paths and SHA-256 digests. No P3 result cell or model prediction existed.

Version 1.8 adds a separate pre-outcome P4 adaptive-controller systems matrix.
The existing resident-vs-tiered matrices cannot measure the system cost of the
central causal contrast. The new 432 paired cells therefore cross both scales,
both 2x/4x budgets, all context/generation/load axes, and the exact
`fixed+pins`/`calibrated+pins` configurations. Every cell runs regardless of
the eventual P2 gate outcome, so system-cell selection cannot depend on quality
results. No adaptive P4 cell existed when this contract was frozen.

Version 1.9 adds a preregistered second-model-family transfer cohort before any
P3 prediction, generated RULER dataset, or baseline-selection outcome existed.
The frozen Phi-4-mini-instruct revision runs native and the single 50%-KV method
selected by the complete Qwen3-1.7B screen on 13 RULER tasks at 8K, 32K, and
128K with 100 paired examples per task-length (7,800 predictions). Phi-specific
reselection or tuning is prohibited. This cohort is reported separately from
the full Qwen3 natural suite and cannot be pooled into a larger apparent sample.

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

The legacy M3 learned-risk pilot is not this predictor: it observes final-query
native probes from a completed full pass and constructs an offline replay plan.
It is retained only as a digest-bound negative pilot. A qualifying online
lookahead must consume signals available no later than token *t*, choose the
token *t+1* residency plan before its sparse-value access, define first-token
fallback, and account for predictor/cache/transfer overhead.

That online contract is frozen separately in
`manifests/p1-online-learned-lookahead-v1.json`. It uses five checkpoint seeds,
both Tier-S scales, all nine workload families, five contexts, and both 2x/4x
budgets. Its conclusion is exploratory and mechanically excluded from the P2
primary causal gate; running it cannot retroactively select or replace a P2 arm.

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
   paths across all 16 arms, 2 budgets, 9 families, and 5 contexts; and
4. run the 9,000 held-out factorial shards (2,880,000 quality arm-conversations
   plus 720,000 direct physical arm-conversations for the central pair and two
   fixed top-p baselines).

No stage may consume 807-series quality to tune the fixed mixture. A failed
calibration-memory cell or equivalence audit blocks held-out execution for that
cell instead of permitting a post-hoc repair.

The resume-safe execution commands are:

```bash
python research/adaptive_v4_memory/scripts/run_p2_core_parallel.py --scale s151 --workers 3
python research/adaptive_v4_memory/scripts/run_p2_causal_parallel.py --workers 3
python research/adaptive_v4_memory/scripts/run_p1_online_lookahead_parallel.py --workers 3
```

The core parallel runner preserves already verified shards from the other
scale. The causal runner partitions the five training seeds without overlap and
rejects any missing or duplicate one of the 9,000 coordinates. The online
runner refuses to publish its progress matrix unless all 6,750 label shards,
20 policies, and 9,000 held-out test shards independently pass their original
implementation and dependency checks. Parallel scheduling is not system
performance evidence; shard wall times from concurrent execution are excluded
from P4 latency and throughput claims.

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
- compute conversation-paired 95% intervals with 10,000 resamples as
  within-seed descriptive uncertainty, and use the five independent training
  checkpoint seed means as the clusters for primary inference;
- report the paired absolute effect and relative effect where defined;
- apply Holm-Bonferroni correction across primary workload families;
- report the worst context-length and worst seed slice; and
- retain failures and timeouts in an explicit accounting table.

A controller passes the primary quality gate only if its mean regression from
native is at most 1 percentage point, no primary family regresses by more than
2 points, and the lower confidence bound of its improvement over the strongest
fixed policy is non-negative on at least two families at both scales.
All five training-seed effects must also be positive at each scale; the
family-level improvement count uses seed-cluster intervals; separately reported
Holm adjustments use exact seed-cluster paired-randomization p-values rather
than treating examples as independent. Holm p-values are not a success gate.

The central causal claim has a separate, stricter gate. At the same measured
hot-memory footprint, `calibrated+pins` must beat `fixed+pins` on both S55 and
S151: the pooled paired effect and its preregistered four-cell-corrected
cluster-bootstrap lower bound must both be greater than zero, and
all five seed-level effects must be positive at each scale. Any failed clause is
reported as a failed or bounded causal claim rather than averaged away. Every
clause is required separately at both 2x and 4x; one budget cannot rescue the
other. The four primary scale-by-budget cells use a Bonferroni-corrected 98.75%
training-seed-cluster bootstrap interval over the five independent seed means;
conversation-paired intervals remain descriptive within a seed. Family slices
use Holm-Bonferroni separately within each contrast, scale, and budget, and the
15 pooled preregistered contrasts form a second Holm family within each scale
and budget. The
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

The operational freeze for these requirements is
`manifests/p3-natural-suite-v1.json`. It selects the pinned 262,144-token
Qwen3-4B-Instruct-2507 snapshot as the primary compatible transfer model while
retaining the earlier Qwen3-1.7B run as a 32K-limited small-model tier. The
minimum matrix contains 45,289 predictions per arm: 32,500 RULER predictions,
10,286 SCBench turn predictions across both modes, all 503 LongBench v2 items,
all 500 LongMemEval_S-cleaned questions, and 1,500 MRCR items through 128K. The
dataset inventory contains 20 digest-bound files totaling 2,391,718,372 bytes.
Dataset acquisition is sequence-gated after P2 and the causal audit, so this
freeze does not modify or compete with the running synthetic core.

The cross-family transfer contract is frozen separately in
`manifests/p3-cross-family-ruler-transfer-v1.json`. It uses the ungated MIT
`microsoft/Phi-4-mini-instruct` snapshot at revision
`cfbefacb99257ffa30c83adab238a50856ac3083`, whose config declares 131,072
tokens and whose 21 top-level files are hash-bound. It adds 3 lengths × 13
tasks × 100 examples × 2 paired arms = 7,800 predictions. The selected 50%-KV
method is inherited unchanged from the Qwen3-1.7B screen; a failed transfer is
reported as negative evidence rather than triggering a Phi-specific search.

Official DeepSeek-V4 Flash evaluation is a separate evidence tier. If the
pinned checkpoint and supported runtime cannot be provisioned, the report must
provide the exact revision, launch command, minimum accelerator/storage
requirement, and estimated cost while withholding official-scale claims.

Natural-language comparisons always include native/dense and the strongest
memory-matched fixed method supported by the selected model. `fixed+pins` and a
synthetic-qualified calibrated arm are included only after an
architecture-preserving port passes prediction/cache equivalence. Qwen3 does
not expose the DeepSeek sparse-attention pin/indexer contract, so a generic
Qwen KV-compression method cannot be relabeled as the Adaptive V4 controller.
FlashMemory- and IndexCache-family baselines are likewise included only when
their pinned implementations support the selected model/runtime;
incompatibility is recorded explicitly and never replaced by a projected
number.

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

The repeated Cartesian matrices cover 8K, 32K, and 128K. The optional 500K
condition is a separate feasibility preflight: S55 and S151 each attempt one
500,000-token prefill plus 128-token decode for resident-native and tiered-native
at batch 1 and one active request. Success, OOM, timeout, and other errors are all
terminal evidence. With one attempt per scale-policy it cannot support latency,
throughput, tail, variance, or production-serving claims.

Reference PyTorch and fused production runtimes are separate result tables.
Projected kernel speedups are never mixed with measured results.

The resident-vs-tiered reference matrix does not answer the adaptive-quota
systems question. A separate 432-cell reference matrix crosses both scales and
both 2x/4x budgets for `fixed+pins` versus `calibrated+pins`, using identical
protected-prefix positions and the calibration-frozen physical memory-match
schedule. It runs whether the P2 causal result is positive, bounded, or negative;
that result changes interpretation rather than selecting systems cells.

The production matrix crosses all four batch sizes with actual concurrency
1/8/32. This yields 216 paired cells across two scales, three contexts, and
three generation lengths. Actual
concurrency is reconstructed from request-level scheduler admission,
first-token, and completion timestamps; all c8/c32 request lifetimes must share
a positive overlap window. Runtime/source/container-or-bare-metal, executable,
accelerator, and driver provenance are mandatory. Serial round-robin execution
is invalid even if its aggregate throughput is reported as concurrency.

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

The executable final gate is `manifests/p5-paper-package-v1.json`, implemented
by `scripts/build_p5_paper_package.py`. It requires the completed P2 core, P2
causal, complete five-benchmark P3 natural suite, P4 serial-interleaved reference,
and a separate P4 production audit with actual concurrency 1/8/32 at their
frozen counts. It regenerates
every primary table and the paper report, records all input/output SHA-256
digests in one artifact index, and classifies conclusions only as `success`,
`bounded-result`, `negative-result`, or `unverified`. A failed causal gate is
bounded to the tested controller, a single compatible-model result is not
promoted to official V4 evidence, reference interleaving cannot be classified
as production success, and failed or partial system cells remain in the primary
table.
