# Adaptive V4 Memory

Adaptive V4 Memory is a research track for making DeepSeek-V4's heterogeneous
long-context memory responsive to the current query and decoding state.

The central design rule is **reversible residency, not irreversible eviction**:
the complete compressed history remains available in a cold tier, while a
risk-aware controller decides which CSA blocks and index computations should be
resident on the accelerator.

## Research documents

- [`research_proposal.md`](research_proposal.md) defines the problem, candidate
  novelty, hypotheses, objective, and falsification criteria.
- [`related_work.md`](related_work.md) maps the relevant KV-cache, sparse
  attention, index reuse, tiered-memory, and evaluation literature.
- [`experimental_protocol.md`](experimental_protocol.md) fixes the baselines,
  workloads, metrics, statistical analysis, artifacts, and phase gates before
  implementation begins.

## Current status

Status: **M0 Tier T instrumentation implemented**

No performance or quality claim has been made. The read-only collector records
native CSA selections and cache accounting. Controller work begins only after
the native trace has been exercised beyond deterministic Tier T tests.

## Working research question

Can native Lightning Indexer score statistics and cross-layer/temporal
disagreement predict the amount of long-range memory a query needs well enough
to jointly adapt:

1. per-CSA-layer resident block budgets;
2. sparse-attention selection budgets;
3. indexer refresh frequency and layer placement; and
4. cold-to-hot block transfers,

while preserving native V4 quality in dense-memory and multi-turn workloads?

## Non-goals

- claiming that every KV-compression workload is safely compressible;
- deleting the only copy of historical compressed entries;
- reporting memory reduction without end-to-end latency and transfer costs;
- optimizing only Needle-in-a-Haystack or only average benchmark accuracy;
- treating the tiny reference model as evidence for frontier-scale serving.

## M0 implementation

The first code change is implemented as instrumentation only:

- capture native CSA selected compressed blocks;
- assign stable request, query-position, layer, and block identifiers;
- record logical-cache, hot-residency, transfer, and latency counters separately;
- write a versioned trace without changing logits or cache contents;
- prove trace-on and trace-off numerical equivalence in tests.

The implementation lives in `nano_deepseek_v4/memory_trace.py`, with the
machine-readable contracts in [`schemas/`](schemas/). At M0 all logical cache
bytes are hot, cold residency and transfers are zero, and no controller or
eviction action exists. Indexer score summaries remain an M1 extension; M0
records the native selected sets needed to establish replay correctness first.
