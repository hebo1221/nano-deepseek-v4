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

Status: **protocol design**

No performance or quality claim has been made. The immediate implementation
target is a read-only trace collector for native CSA indexer behavior. Controller
work begins only after the native baseline and trace schema pass the M0 gates in
the experimental protocol.

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

## First implementation slice

The first code change must be instrumentation only:

- capture native CSA indexer score summaries and selected compressed blocks;
- assign stable request, turn, token, layer, and block identifiers;
- record logical-cache, hot-residency, transfer, and latency counters separately;
- write a versioned trace without changing logits or cache contents;
- prove trace-on and trace-off numerical equivalence in tests.

