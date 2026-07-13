# Research proposal: risk-aware adaptive residency for DeepSeek-V4

| Field | Value |
| --- | --- |
| Status | Draft v0.1 |
| Date | 2026-07-14 |

## Abstract

DeepSeek-V4 already reduces long-context cost through a heterogeneous attention
stack: Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), a
sliding-window branch, and a Lightning Indexer. Its native policy is nevertheless
largely fixed at inference time: compression rates, sparse-selection budgets,
indexer placement, and cache layout do not respond to the memory density of the
current query.

We propose Adaptive V4 Memory, a risk-aware controller for the *physical
residency* and *retrieval compute* of V4's compressed history. The full logical
compressed cache is retained in a cold tier. At runtime, the controller chooses
per-layer hot budgets, sparse-selection budgets, indexer refresh points, and
cold-to-hot transfers. It falls back toward native behavior when its confidence
is low or the workload requires dense global memory.

The target contribution is not another irreversible KV eviction heuristic. It
is a V4-native, reversible, multi-axis controller evaluated on quality, hot
memory, transfer traffic, indexer compute, latency, and worst-case failures.

## 1. Problem statement

For decoding step `t`, split physical accelerator memory into:

\[
M_{GPU}(t) = M_{state}(t) + M_{HCA}(t) + M_{CSA,res}(t) + M_{index,res}(t),
\]

where `state` includes the sliding window and incomplete compression tails,
`HCA` is the densely read heavily compressed stream, `CSA,res` is the set of
CSA compressed entries currently resident on the accelerator, and `index,res`
contains the index representations needed to select CSA entries.

The complete compressed history remains in a logical cache:

\[
\mathcal{M}_{logical}(t) = \mathcal{M}_{hot}(t) \cup \mathcal{M}_{cold}(t).
\]

No controller action may destroy the only recoverable copy of an entry. This
separates the research from eviction policies that permanently discard context
and then fail when later turns ask a different question.

At each control point, the action is

\[
a_t = \{b_{l,t}, k_{l,t}, r_t, \mathcal{L}_t, \mathcal{P}_t\},
\]

where `b` is the hot block budget per CSA layer, `k` is the sparse-selection
budget, `r` is the next refresh interval, `L` is the set of indexer layers to
execute, and `P` is the cold-to-hot prefetch set.

The research objective is a constrained Pareto optimization, not a single
weighted score:

\[
\min \{\Delta Q, M_{GPU}, B_{transfer}, C_{index}, T_{prefill}, T_{decode}\}
\]

subject to an explicit tail-risk constraint on quality regression. Weighted
Lagrangian objectives may be used for controller training, but all final results
must report the complete Pareto frontier.

## 2. Why this is specific to V4

V4 differs from the multi-head KV caches assumed by much of the compression
literature:

- shared K=V MQA leaves one primary KV stream per layer, so head-wise KV budget
  allocation is not directly transferable;
- CSA selects from entries that already summarize overlapping token blocks;
- HCA provides a separate, dense, 128x-compressed global path;
- every attention layer also contains a local sliding branch;
- indexer keys, primary compressed entries, tail state, and sliding state have
  different shapes and update rules;
- V4's native sparse indexer can itself dominate long-context compute.

The useful adaptive axes are therefore layer, time, compressed block, indexer
execution, and memory tier—not ordinary KV-head eviction.

## 3. Closest prior work and candidate novelty

The closest direct predecessor is
[FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079), which keeps a CPU
cold pool and periodically predicts a GPU working set. Its reported policy uses
a fixed refresh interval, fixed threshold, three fixed prediction layers, and
OR aggregation. The paper reports two diagnostic failures that define our main
targets: false-positive residency continues to grow on context-independent
queries, and dense-memory MRCR tasks regress sharply.

[IndexCache](https://arxiv.org/abs/2603.12201) shows that indexer output can be
reused across layers, but its inference pattern is fixed after calibration or
training. [Ada-KV](https://arxiv.org/abs/2407.11550) optimally redistributes a
fixed top-k eviction budget across attention heads under its loss bound, while
[SqueezeAttention](https://arxiv.org/abs/2404.04793) reallocates budgets across
layers from prompt-time statistics. Neither jointly controls V4 residency,
indexer compute, and cold-tier transfer.

The candidate novelty is the joint combination of:

1. query- and time-dependent per-CSA-layer residency;
2. adaptive sparse-selection budget rather than a global fixed top-k;
3. adaptive indexer layer/refresh selection rather than a static reuse pattern;
4. explicit cold-tier preservation for multi-turn reversibility;
5. uncertainty-triggered widening toward native V4 behavior; and
6. evaluation of GPU bytes, interconnect traffic, compute, latency, average
   quality, and worst-case quality under one protocol.

These are candidate claims only. The related-work matrix must be refreshed
before any paper submission, and a claim is removed if an equivalent prior
method is found.

## 4. Controller signals

The first controller must use signals already available from native execution:

- entropy and concentration of Lightning Indexer scores;
- number of entries required to reach a target cumulative score mass;
- score margin between the selected boundary and the next entry;
- cross-layer Jaccard overlap and rank correlation of selected blocks;
- disagreement between designated predictor layers;
- temporal change in selections since the previous refresh;
- query-state distance from recent control points;
- HCA attention entropy or concentration;
- request turn, prompt/decode position, and recent retrieval misses;
- cold-block reuse and transfer history.

Signals that require full dense attention are allowed only for offline oracle
labeling and analysis, never as a hidden online cost in the final method.

## 5. Hypotheses

### H1: demand predictability

Native indexer concentration, boundary margin, and cross-layer/temporal
disagreement predict the oracle number of CSA entries required to preserve
native output within a fixed tolerance.

### H2: adaptive residency

A per-query, per-layer budget achieves a strictly better quality–hot-memory
frontier than fixed threshold, fixed top-k, and uniform per-layer budgets.

### H3: reversible memory

Preserving the full logical compressed history in a cold tier substantially
reduces multi-turn and query-shift regressions compared with irreversible
eviction at matched hot-memory budgets.

### H4: risk-aware fallback

Score uncertainty and layer disagreement identify dense-memory cases early
enough that widening the active set bounds worst-case quality loss without
destroying the average memory benefit.

### H5: adaptive index execution

Temporal and cross-layer stability allow the controller to skip at least half
of indexer work at matched quality, while dynamic refresh outperforms a fixed
IndexCache-style layer pattern on distribution shifts.

## 6. Proposed method stages

### Stage A: trace and oracle

Run unmodified native behavior and record scores, selected compressed block
IDs, layer overlap, cache growth, and timing. Replay each example with controlled
budgets to estimate the minimum sufficient set and label dense-memory events.

### Stage B: training-free controller

Use calibrated rules based on score mass, margin, disagreement, and a global
budget constraint. This stage establishes interpretable baselines and tests
whether H1 holds before training a controller.

### Stage C: learned risk controller

Train a lightweight model against oracle sufficient-budget labels. Candidate
loss terms include budget regression, block-recall classification, transfer
cost, and an asymmetric penalty for under-allocation. The predictor must be
small enough that its latency is included rather than ignored.

### Stage D: systems integration

Implement hot/cold block tables, asynchronous prefetch, bounded lookahead, and
batch-aware scheduling. Logical compression gain does not count as a systems
result until a real kernel/runtime benefits from the non-uniform working set.

## 7. Falsification criteria

The project must report negative outcomes. The main hypothesis is rejected or
narrowed if any of the following persists after the planned ablations:

1. no controller Pareto-dominates the strongest fixed policy on at least two
   model scales and two benchmark families;
2. native-quality regression exceeds 1 percentage point on average or 2 points
   on any primary benchmark at the chosen operating point;
3. the MRCR/dense-memory fallback recovers less than 90% of native performance;
4. dynamic residency reduces GPU bytes but PCIe/NVLink traffic removes the
   end-to-end decode-latency benefit;
5. controller plus telemetry consumes more than 10% of decode time at the
   target context length;
6. multi-turn performance is not materially better than irreversible eviction
   at the same hot-memory budget;
7. adaptive index refresh cannot skip at least 50% of eligible indexer calls
   while staying within the quality bound; or
8. results depend on one synthetic retrieval benchmark and fail to transfer to
   natural, multi-turn, or global-information tasks.

For context-independent inputs, a stretch target is that quadrupling irrelevant
history increases the dynamic CSA hot set by no more than 1.5x. Failure to meet
this target is reported as a scaling limitation, not hidden by percentage-only
memory metrics.

## 8. Claims we will not make

- A tiny-model result proves official Flash serving performance.
- Lower logical cache size implies lower latency.
- Mean benchmark parity implies instruction or security robustness.
- Needle-in-a-Haystack success establishes general long-context memory.
- A fixed compression percentage is comparable across different cache layouts.

## 9. Deliverables

1. a versioned, replayable native V4 memory trace format;
2. an oracle budget and failure-analysis tool;
3. fixed-policy and training-free adaptive baselines;
4. a learned risk controller if the training-free study supports H1;
5. a hot/cold cache prototype with transfer-aware measurements;
6. benchmark manifests, raw run metadata, and Pareto plots; and
7. a paper or negative-results report with reproducible artifacts.
