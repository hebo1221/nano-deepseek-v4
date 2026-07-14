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
- [`paper_grade_protocol.md`](paper_grade_protocol.md) freezes the multi-seed,
  causal-ablation, natural-benchmark, statistical, and systems expansion.

## Current status

Status: **paper-grade expansion active; M5 one-token result is a pilot**

No performance or systems claim has been made. M1 adds deterministic baselines,
an exhaustive Tier-T oracle, differentiable CSA training probes, and two
trained Tier-S scales. Native score features reduced sufficient-budget MAE by
11.9% and 16.3% over the preregistered context+layer baseline, so the M2
training-free controller track may proceed. The evidence is one-seed synthetic
diagnostics only, and both scales missed the separate 85% training target.

M2 adds a deterministic training-free controller with global budgets,
stability-based refresh, protected-block pinning, movement accounting, and
uncertainty fallback. On held-out Tier-S associative recall it improved over a
memory-matched fixed top-k at both scales. These are logical selection results:
all blocks remain GPU-resident, so there is no physical memory claim.

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

The M0 instrumentation provides:

- capture native CSA selected compressed blocks;
- assign stable request, query-position, layer, and block identifiers;
- record logical-cache, hot-residency, transfer, and latency counters separately;
- write a versioned trace without changing logits or cache contents;
- prove trace-on and trace-off numerical equivalence in tests.

The implementation lives in `nano_deepseek_v4/memory_trace.py`, with the
machine-readable contracts in [`schemas/`](schemas/).

## M1 implementation and evidence

`nano_deepseek_v4/memory_replay.py` runs native, recency, random, fixed top-k,
fixed top-p, calibrated per-layer, and index-reuse policies from one v2 trace.
It also extracts score and overlap features and evaluates the predictive gate
with grouped holdout validation. `nano_deepseek_v4/memory_probe.py` exposes an
opt-in differentiable research probe without changing normal model outputs.

See the checked Tier-T and Tier-S reports in [`reports/`](reports/) and their
small machine-readable summaries in [`results/`](results/). Checkpoints and raw
training summaries remain ignored artifacts referenced by SHA-256.

## M2 implementation and evidence

`nano_deepseek_v4/memory_controller.py` implements the training-free rule,
action records, replay digest validation, protected pinning, dense fallback,
movement accounting, and immutable counterfactual selection plans. The M2
report records calibration/test isolation, memory-matched fixed baselines, a
forced fallback stress arm, and the first overhead-gate failure. M3 learned-risk
work may proceed, but actual hot/cold allocation remains an M4 requirement.

## M3 decision

The learned risk controller used disjoint train/calibration/test examples,
asymmetric budget loss, calibrated dense fallback, and feature/loss/size
ablations. At both scales it improved some predictive metrics but required
57–63% fallback and failed to improve the M2 quality/block Pareto. The checked
negative-result report freezes that outcome; M2 remains the controller
candidate rather than the learned controller.

## M4 implementation and evidence

`nano_deepseek_v4/tiered_memory.py` implements pinned-CPU cold storage, bounded
GPU hot residency, asynchronous CUDA prefetch, protected blocks, late-miss
recovery, and transfer accounting for CSA compressor values. Cache
clone/crop/select/stack and persistence preserve tier state. S55 and S151 CUDA
measurements show equal quality and about 91% physical reduction for the tiered
value component, but only about 1.9% of total measured cache allocation because
index and rollback state remain resident. Batch-1 throughput retained 94–95%;
no speedup was observed. See the checked M4 report and summary for the bounded
claim and negative strict-gate decision.

## M5 pilot decision

`nano_deepseek_v4/online_memory_controller.py` connects M2 actions to the real
tier fetch with deterministic lifecycle and persistence support. Across three
synthetic workload families and both Tier-S scales, it respected every budget
but lost substantial quality because a cross-layer global decision can only be
applied one token after its scores are complete. Total hot-cache reduction was
only 1.51–1.92%, p95 latency was 1.11–1.29x native, and no speedup was observed.

This falsifies only the tested one-token global interface, not adaptive memory
in general. The [`paper_grade_protocol.md`](paper_grade_protocol.md) freezes the
larger multi-seed, natural-benchmark, causal-ablation, and systems study before
new results are observed.

The pinned official Flash (159.62 GB) and Pro (864.72 GB) payloads cannot be run
on the available host, so official-scale execution remains explicitly
unverified. The checked M5 report records the pilot negative result, official
feasibility audit, minimum fused-kernel contract, and the retained bounded M4
fixed-top-k result.

## P1 causal-controller pilot

`nano_deepseek_v4/causal_memory_controller.py` now applies a layer-local action
to the same token that produced its scores, with static global-to-layer quotas
and optional earlier-layer signals. The controller governs both prefill and
decode, drives the physical tier fetch, validates serialized replay digests,
and supports the full cache lifecycle.

The clean two-scale pilot recovered the one-token controller's quality loss but
did not improve on fixed top-k. At the tested minimum quotas the no-fallback
controller produced the same prediction digests as fixed top-k; dense fallback
used 4--7x more hot blocks on multi-query workloads without an accuracy gain.
This is a directional single-seed result, not paper-grade evidence. See the
checked P1 report and summary. Calibration-only non-uniform quotas, protected
pinning, learned lookahead, and the preregistered five-seed matrix remain active
work.
