# Experimental protocol

| Field | Value |
| --- | --- |
| Protocol version | 0.1 |
| Status | Pre-implementation |
| Last updated | 2026-07-14 |

This protocol is the preregistered evaluation contract for Adaptive V4 Memory.
Changing a primary metric, benchmark, exclusion rule, or acceptance threshold
after seeing results requires a dated amendment and separate reporting of the
original protocol.

## 1. Experimental principles

1. Compare methods under the same model weights, prompts, generated-token
   limits, seeds, dtype, hardware, batching, and runtime.
2. Distinguish logical cache bytes, accelerator-resident bytes, host/disk bytes,
   and transferred bytes.
3. Report controller/indexer overhead and end-to-end latency; theoretical
   attention FLOPs are secondary.
4. Evaluate query shifts and repeated requests, not only the first known query.
5. Report mean quality, task-level quality, worst-case regression, and tail
   latency.
6. Keep a native V4 result for every workload and context length.
7. Treat tiny-model and official-checkpoint evidence as separate experimental
   tiers.

## 2. System under test

### Tier T: tiny deterministic model

Purpose: correctness, exhaustive replay, property tests, and oracle enumeration.

- current `DeepSeekV4Config`-scale models;
- FP32 CPU as the numerical reference;
- CUDA FP32/BF16/FP16 for instrumentation equivalence;
- short sequences where candidate block sets can be exhaustively enumerated.

Tier T establishes implementation correctness only. It cannot support serving
or model-quality claims.

### Tier S: small trained V4 models

Purpose: controlled research where attention structures have been learned.

Planned scales are approximately 50M, 150M, and—resources permitting—300M
parameters. At least two independently trained scales are required for a
positive research claim. Training data, tokenizer, token count, seeds, and all
checkpoints must be published or reproducibly referenced.

No controller result is interpreted before a native CSA/HCA model demonstrates
non-trivial long-context retrieval and language-model quality.

### Tier F: official Flash evidence

Purpose: architecture-realistic trace analysis and systems validation.

The official checkpoint cannot be treated as locally materialized merely
because its shards can be streamed. Tier F experiments must record the actual
runtime, device topology, parallelism, kernel versions, and whether scores were
captured from the official DeepSeek or Hugging Face implementation. Results
obtained only from index metadata or converted tensor shapes are not inference
results.

## 3. Policies and baselines

Every primary comparison includes:

1. **Native V4**: native CSA top-k, HCA, sliding state, and indexer execution.
2. **Recency**: fixed recent compressed blocks plus mandatory native state.
3. **Random**: random compressed blocks at the same hot budget, with seeds.
4. **Fixed top-k**: one global CSA hot/selection budget.
5. **Fixed top-p**: the smallest set reaching a fixed indexer score mass.
6. **Per-layer calibrated**: static layer budgets selected on calibration data.
7. **Squeeze-style**: prompt-derived layer allocation from a documented proxy.
8. **IndexCache-style**: fixed indexer-layer reuse patterns, uniform and
   calibrated.
9. **FlashMemory-style**: fixed refresh interval, fixed threshold, fixed
   predictor layers, and cold/hot residency.
10. **Oracle**: replay-derived minimum sufficient budget or best feasible set;
    offline only.
11. **Adaptive V4 Memory**: training-free and learned variants reported
    separately.

If a published baseline cannot be reproduced faithfully, label it `proxy`,
document the mismatch, and exclude it from claims of outperforming that work.

## 4. Workloads

### 4.1 Synthetic diagnostic suite

The repository will provide deterministic generators for:

- single passkey and variable-position Needle-in-a-Haystack;
- multiple needles with independent and overlapping queries;
- associative recall and multi-hop tracing;
- dense global aggregation where most blocks contribute;
- irrelevant-context queries requiring only local state;
- query shift: multiple unrelated questions over one shared context;
- instruction persistence with system instructions at beginning, middle, and
  end positions;
- adversarial distractors with high lexical similarity; and
- long generation where the relevant block changes over time.

Each generator records the exact evidence block IDs, number of relevant blocks,
context composition, seed, and answer. Synthetic tests are diagnostic and may
not be averaged into the primary natural-task score.

### 4.2 Established long-context benchmarks

- [RULER](https://arxiv.org/abs/2404.06654): controlled retrieval and context
  length scaling;
- [LongBench v2](https://arxiv.org/abs/2412.15204): realistic long-context
  understanding and reasoning;
- [LongMemEval](https://arxiv.org/abs/2410.10813): long-term interactive memory;
- [SCBench](https://arxiv.org/abs/2412.10319): multi-turn and multi-request
  shared-context cache reuse;
- [Michelangelo/MRCR](https://arxiv.org/abs/2409.12640): latent-structure and
  dense multi-range retrieval; and
- language-model validation corpora at multiple sequence lengths for NLL and
  perplexity.

Benchmark versions, dataset hashes, licenses, preprocessing, prompt templates,
and scoring scripts must be pinned in a manifest. Test labels may not be used
for controller training or budget calibration.

### 4.3 Security and instruction robustness

Following the failure mode documented by
[The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231), run a
separate suite for:

- system-prompt leakage;
- refusal/instruction retention across compression levels;
- conflicting instruction order;
- long irrelevant prefixes around protected instructions; and
- follow-up queries that require an instruction not used in the first answer.

Protected blocks may be pinned by a policy, but pinned bytes must be counted and
the policy must not use hidden benchmark labels.

## 5. Trace contract

The native trace is append-only and versioned. JSONL is acceptable for the first
prototype; Parquet or Arrow is preferred for large experiments. Every trace
contains enough information for deterministic policy replay without saving raw
private prompts.

### 5.1 Run metadata

- schema version and trace ID;
- repository commit and dirty-state flag;
- model/config/checkpoint content digest;
- tokenizer and dataset manifest digest;
- runtime, PyTorch, CUDA, driver, kernel, and accelerator details;
- dtype, batch size, seed, determinism settings, and warm-up policy;
- policy name and complete policy configuration.

### 5.2 Control-point event

- request, conversation, turn, and query IDs;
- prompt/decode phase, absolute position, generated-token index;
- layer index and layer type;
- logical compressed-block count and bytes;
- hot/cold block IDs or privacy-preserving stable hashes;
- native selected block IDs and rank;
- score count, entropy, top-p cardinalities, boundary margins, and quantiles;
- cross-layer and temporal selection-overlap statistics;
- controller action, confidence, fallback reason, and next refresh point;
- cold-to-hot and hot-to-cold block/byte counts;
- indexer, transfer, core-attention, and end-to-end timings; and
- cache hit, miss, and late-prefetch counters.

Raw indexer score vectors are optional because of trace size. If omitted, a
replayable top-N list and validated summary statistics are required. Score
summaries must be computed outside timed regions or their cost must be reported.

### 5.3 Numerical invariance gate

With the controller disabled, tracing must not alter:

- logits beyond zero tolerance in deterministic FP32 tests;
- native selected block IDs;
- cache tensors and position counters;
- RNG state used by generation; or
- generated sequences.

Any non-zero observer effect blocks M1.

## 6. Metrics

### 6.1 Quality

- native and policy task scores;
- absolute and relative score delta;
- NLL/perplexity delta;
- exact-match and evidence-block recall;
- instruction-retention and leakage rates;
- per-turn score and degradation slope;
- worst primary-task regression; and
- dense-memory fallback recovery relative to native.

Do not use a single macro average to hide a failed workload.

### 6.2 Memory

- logical cache bytes;
- peak and mean accelerator-resident cache bytes;
- state, HCA, CSA, and indexer bytes separately;
- cold host and disk bytes;
- pinned/protected bytes;
- fragmentation and reserved-but-unused bytes; and
- bytes per original context token.

Report absolute bytes and ratios. Percentage reduction without the fixed state
floor can exaggerate gains at short context lengths.

### 6.3 Compute and movement

- native and executed indexer calls;
- indexer FLOPs or measured kernel time;
- core sparse-attention selected entries;
- controller inference time;
- host-to-device and device-to-host bytes;
- transfer count, mean size, and useful-byte fraction;
- cache misses and late prefetches; and
- CPU/disk retrieval time when applicable.

### 6.4 Latency and throughput

- time to first token;
- inter-token latency mean, p50, p95, and p99;
- per-request and aggregate decode tokens/s;
- throughput under cache-saturated concurrency;
- controller and telemetry overhead; and
- energy if a stable measurement interface is available.

All latency experiments include warm-up, synchronization boundaries, sample
counts, and confidence intervals.

### 6.5 Controller diagnostics

- sufficient-budget prediction calibration;
- under-allocation and over-allocation rates;
- block precision/recall against the native/oracle set;
- uncertainty AUROC/AUPRC for dense-memory events;
- expected calibration error;
- budget regret relative to the replay oracle;
- action stability across adjacent decode steps; and
- fallback frequency and bytes recovered by fallback.

## 7. Comparison modes

Every method is compared at three matched operating conditions:

1. **memory-matched**: same mean hot-cache bytes;
2. **quality-matched**: within the preregistered native-quality tolerance; and
3. **latency-matched**: same p95 inter-token latency where feasible.

The full quality–memory, quality–transfer, and quality–latency Pareto curves are
primary results. A single hand-selected operating point is insufficient.

## 8. Statistical protocol

- Use at least three seeds for trained Tier S models and stochastic policies.
- Pair policies on identical examples, generation seeds, and native traces.
- Report bootstrap 95% confidence intervals over examples for quality deltas.
- Report confidence intervals over repeated timed runs for latency metrics.
- Correct for multiple comparisons in confirmatory ablations.
- Separate calibration, controller training, validation, and final test splits.
- Freeze thresholds before final benchmark execution.
- Report all failed/OOM runs and exclusion reasons.

Practical significance thresholds from the research proposal take precedence
over statistical significance alone.

## 9. Required ablations

1. score concentration only;
2. + boundary margin;
3. + cross-layer disagreement;
4. + temporal change;
5. + HCA signal;
6. fixed versus adaptive refresh;
7. fixed versus per-layer budget;
8. destructive eviction versus recoverable cold residency;
9. no fallback versus uncertainty fallback;
10. controller size and execution frequency;
11. synchronous versus asynchronous transfer; and
12. protected instruction pinning on/off with counted bytes.

For learned controllers, compare asymmetric under-allocation loss, regression,
classification, and ranking objectives. Do not tune all ablations on the final
test set.

## 10. Phase gates

### M0: instrumentation

Exit criteria:

- trace schema and digest validation implemented;
- trace-on/off numerical invariance passes;
- cache/accounting identities hold exactly in synthetic tests;
- no raw benchmark text is emitted by default; and
- observer overhead is measured.

### M1: replay oracle and fixed baselines

Exit criteria:

- native traces replay deterministically;
- recency, random, fixed top-k/top-p, per-layer, and index-reuse baselines run in
  one harness;
- oracle sufficient-budget search is verified exhaustively on Tier T; and
- H1 feature/target relationships are measured without controller training.

Stop the learned-controller track if native signals do not predict sufficient
budget better than context length and fixed layer identity.

### M2: training-free adaptive controller

Exit criteria:

- controller respects a global hot-memory budget;
- every action and fallback is replayable;
- at least one policy improves a fixed baseline on validation Pareto curves;
- no primary workload exceeds the quality limits; and
- movement cost is included.

### M3: learned risk controller

Exit criteria:

- controller training and test traces are disjoint;
- calibration and dense-memory detection improve over training-free rules;
- two Tier S model scales show consistent direction; and
- required ablations identify the source of gain.

### M4: tiered runtime

Exit criteria:

- real hot/cold blocks, asynchronous transfer, and batching are implemented;
- non-uniform residency reduces measured accelerator allocation;
- p95/p99 latency and throughput improve at a quality-matched point; and
- fragmentation/reservation overhead is reported.

### M5: official-scale validation

Exit criteria:

- official runtime or a clearly identified compatible implementation is used;
- FlashMemory-style and native V4 baselines are reproduced or discrepancies are
  documented;
- at least three benchmark families, including a dense-memory and a multi-turn
  workload, are evaluated; and
- all proposal falsification criteria are reported.

## 11. Planned artifact layout

The following layout is reserved but should be created only as implementations
land:

```text
research/adaptive_v4_memory/
├── configs/                 # immutable experiment configurations
├── manifests/               # dataset/model/runtime digests
├── schemas/                 # trace and result JSON schemas
├── scripts/                 # trace, replay, benchmark, and plot entry points
├── results/                 # small checked-in summaries, never raw private data
└── reports/                 # dated experiment and negative-result reports
```

Large traces, model checkpoints, and benchmark datasets are external artifacts
referenced by content digest. They must not be committed to Git.

## 12. First implementation issue

Implement a passive native-memory trace collector with the following scope:

1. typed trace configuration and event/result records;
2. optional hooks at CSA indexer selection and cache advancement;
3. stable compressed-block identifiers across chunked decoding;
4. JSONL output with an atomic manifest and SHA-256 digest;
5. zero raw token/prompt logging by default;
6. trace-on/off FP32 equality and cache equality tests;
7. chunked/full trace consistency tests; and
8. a tiny replay utility that reconstructs native selected sets.

The first issue must not implement pruning, residency changes, a controller, or
performance claims. Instrumentation correctness is the only M0 objective.
