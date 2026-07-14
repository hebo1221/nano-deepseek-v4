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

## Legacy M3 offline pilot decision

The learned risk pilot used disjoint train/calibration/test examples,
asymmetric budget loss, calibrated dense fallback, and feature/loss/size
ablations. At both scales it improved some predictive metrics but required
57–63% fallback and failed to improve the M2 quality/block Pareto. The checked
negative-result report freezes that outcome. Its features came from final-query
native probes collected in a full pass and were then used to construct an
offline replay selection plan; this is not an online lookahead execution path.
M2 remains the controller candidate, while a deployable token-t to token-(t+1)
learned lookahead stays a separate P1 experiment.

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

The later P3 audit also pins the public FlashMemory paper, serving code, base
weights, and retriever snapshot. It found a separate release-contract blocker:
the serving path expects a 64-head/1024-rank PT checkpoint that is absent from
the published retriever snapshot, while the public safetensors model card
describes a 128-head/2048-rank artifact and provides no serving conversion.
The [P3 feasibility report](reports/2026-07-14-p3-official-flashmemory-feasibility.md)
and [execution manifest](manifests/p3-flashmemory-deepseek-v4-v1.json) therefore
freeze both the resource and checkpoint blockers without claiming an official
run.

The [P3 natural-suite freeze](reports/2026-07-14-p3-natural-suite-freeze.md)
pins the Qwen3-4B 262K transfer model, all five natural benchmark revisions,
20 dataset files, official scorer contracts, the 45,289-prediction-per-arm
minimum, and the sequence-gated acquisition path. Oversized examples are
reported as unsupported and are never head-tail truncated.

The supplemental safety-retention study adds 3,600 actual-model predictions:
four frozen families at 8K/32K/128K, 100 examples per slice, and three paired
arms. Its causal contrast compares the strongest fixed baseline with and
without an exact system-role protected prefix at the same KV budget. The audit
requires paired bootstrap uncertainty, an exact paired test, literal canary
leakage rates, worst-slice reporting, and layer-level proof that pinning neither
adds kept tokens nor increases resident KV bytes. This remains synthetic
transfer evidence, not a comprehensive safety certification.

The natural-safety extension adds 7,254 paired Qwen3-4B generations: 6,172
LongSafety front/end placements and 1,082 IFEval prompts across native and the
strongest memory-matched fixed arm. IFEval is scored with its pinned official
deterministic implementation. Its `nltk==3.10.0` runtime and the official
`punkt`/`punkt_tab` archives are revision-, archive-, and extracted-tree-bound,
so scoring never depends on an unrecorded user-level NLTK cache. LongSafety
generations are digest-bound and failure-accounted, but its official three-agent
paid judge is blocked pending explicit opt-in, so no comparative LongSafety
safety score is claimed.

The suite RULER evidence is regenerated separately with the pinned Qwen3-4B
tokenizer at 8K/16K/32K/64K/128K, then executed as 32,500 paired records per
required arm. The earlier Qwen3-1.7B RULER matrix remains the leakage-safe
fixed-baseline selection set and is not substituted for this 4B evidence.

LongBench v2 has an executable Qwen3-4B native/frozen-fixed runner with
example-level resume. It renders the pinned direct 0-shot prompt, scores invalid
answer formats as zero instead of excluding them, records every operational
failure, and measures exact post-prefill key/value cache bytes before question
decoding. Both arms must use identical example and prompt digests.

RULER and MRCR now share the same exact-token rule: each complete rendered chat
prompt is tokenized once and only then sliced into prefill and final-query
tensors, so a BPE merge at the boundary cannot change the evaluated input.
RULER additionally binds all five generated dataset manifests and the pinned
official scorer digest into each terminal arm artifact.

LongMemEval runs all 500 cleaned full-history questions for both required arms.
Without an explicitly enabled `gpt-4o-2024-08-06` judge, generations and
physical measurements are retained as `judge-blocked` terminal records and no
auxiliary score is substituted for the official metric.

The adaptive compatibility expansion separately pairs `fixed+pins` with the
causal layer-quota arm on Qwen3-4B RULER, SCBench, LongBench-v2, and MRCR
(89,578 predictions total). A fifth 1,000-generation LongMemEval cohort preserves
the same raw-response and physical-budget evidence but remains quality-unverified
while the official judge is blocked. Phi-4-mini runs the same adaptive contrast
on RULER and LongBench-v2 as a bounded two-model × two-benchmark transfer grid;
no Qwen/Phi scores or p-values are pooled.

SCBench executes all 12 tasks in both official multi-turn and multi-request
modes (10,286 turn predictions per arm). Multi-turn retains the official
golden-answer follow-up prompts while removing generated answer tokens;
multi-request restores one compressed shared-context cache after every query.
Task scorers, RepoQA thresholding, and the ROUGE metric script are digest-bound.

The P4 reference systems matrix freezes 216 scale/context/generation/load cells,
with five warmups and 30 measured repetitions for resident and tiered policies.
Policy failures are isolated: if resident OOMs, the surviving tiered policy is
still measured and the cell is reported as partial instead of being discarded.
Its c8/c32 profiles are serial round-robin active-request probes, not actual
concurrent serving; they remain bounded reference evidence even at 216/216.

Before that matrix, `p4-500k-context-preflight-v1.json` separately attempts a
500,000-token prefill plus 128-token decode on S55/S151 for resident-native and
tiered-native. It has one attempt per scale-policy, preserves OOM/timeout/error
outcomes, and is feasibility evidence only—not a latency or throughput result.

The separate P4 production manifest freezes another 216 cells behind an
external serving-adapter contract. Its c8/c32 cells pass only when raw request
admission, first-token, and completion timestamps reconstruct the requested
overlap and the per-request, per-token execution records prove either a
multi-request decode batch or genuinely overlapping decode execution. Merely
keeping request lifecycles open around a serial decode loop is rejected. The
adapter executable, runtime revision, deployment image or explicit
bare-metal mode, accelerator, driver, P3 audit, and every cell artifact are
digest-bound. A serial loop, projected metric, aggregate-only latency, or
unreported failure cannot satisfy the production gate.

The checked `p4_continuous_batch_adapter.py` is the first concrete backend for
that contract. It combines all admitted requests into the actual model batch at
every prefill chunk and decode step, and records every request-token coordinate
against that execution batch. Its claim remains static full-request batching;
it does not establish dynamic arrivals, continuous admission, a fused kernel,
or multi-GPU serving.

After all audits finish, the strict P5 package can be regenerated with:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/build_p5_paper_package.py
```

The command requires a clean tree and complete 4,500-shard P2 core,
9,000-shard causal, 370,500-prediction/57-cell small-model RULER, five-benchmark natural,
3,600-prediction safety-retention, 7,254-generation natural-safety, 216-cell
reference-system, four terminal 500K feasibility attempts, and separate
216-cell actual-concurrency production audits. It writes digest-indexed CSV
tables and a paper-style report under
`artifacts/adaptive_v4_memory/paper_grade/p5/`; missing evidence is never
imputed.

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

The online learned-lookahead runtime is now implemented separately from that
legacy pilot. A frozen 16-hidden two-head policy consumes only token-t CSA
rankings, is applied at token t+1, uses native selection for bootstrap, and is
serialized with its calibration, split digests, parameters, selections, and
replay digest. Cache clone/select/stack/crop/persistence and physical tiering
are covered by executable tests. Its full 5-seed × 2-scale × 9-family ×
5-context × 2-budget protocol is frozen in
`manifests/p1-online-learned-lookahead-v1.json`; it runs after, and cannot alter,
the P2 primary causal gate.

The follow-up calibration matrix fits checkpoint-specific same-token layer
quotas from disjoint 707-series seeds using all nine P2 families and five context
lengths. All 5 seeds × 2 scales completed at 256 conversations per family:
23,040 conversations and 286,720 query-layer observations. The digest-bound
audit revalidated every raw file, checkpoint, family/context slice, budget, and
leakage guard. S55 produced five distinct quota vectors at both 2x and 4x;
S151 produced two at 2x and three at 4x. This checkpoint variation rules out
reusing one seed's quota as a universal policy. The 1x point remains the uniform
fixed-policy floor, and protected end positions are inherited by the physical
tier store. This is still calibration-only evidence: held-out quality was not
inspected. See the checked five-seed calibration report and summary.

The first paired held-out pilot now covers all nine families and five contexts
on S55 seed 6071401 with 13 policies. Hierarchical calibration versus fixed was
+2.32 pp at 1x, -0.18 pp at 2x, and +0.18 pp at 4x over 560 queries. The 1x
recoveries occurred entirely on instruction persistence, despite an identical
uniform quota, so a protected-pin ablation is required before attribution.
Dense fallback changed no correctness outcome while increasing H2D traffic;
it is an ablation negative result, not a core full-matrix arm. Every policy,
including native, failed the irrelevant-local slice. These are directional
single-checkpoint findings only; see the checked held-out pilot report and
digest-bound paired analysis.

The exact-example pin ablation has since isolated that effect: no-pins
hierarchical-1x was prediction-identical to fixed-1x (44/80), while pins reached
57/80, with all 13 recoveries at context 256+. The 1x gain is therefore a
protected-retention result, not a calibrated-quota result. Native full-sequence
forward also reproduced the irrelevant-local 0/20 result, and strengthened
generator invariants verified the local target/key construction; that slice is
a checkpoint generalization failure.

For the large quality matrix, chunk=2 causal-cache evaluation has been accepted
after exact comparison with the token-by-token pilot: all 1,260 core-policy
conversation records and 3,920 predictions matched. Full-sequence and larger
chunks showed at least one mismatch and are rejected. This optimization applies
only to quality; physical tier, transfer, and latency claims remain on the
sequential runtime path.

The quality matrix is also frozen at batch 4. A direct S55 comparison rejected
batch 20 after a prediction changed on `fixed-2x` at context 128, despite its
short-prefix speedup. The checked counterexample is in
`results/p2-batch20-equivalence-s55.summary.json`; the matrix runner rejects
larger batch sizes and validates this field when resuming shards.

Paper-grade execution now has three resume-safe parallel entry points:
`run_p2_core_parallel.py`, `run_p2_causal_parallel.py`, and
`run_p1_online_lookahead_parallel.py`. They hold the same exclusive study GPU
lock as the sequential runners, divide only disjoint seed/family coordinates,
and preserve the original shard implementation digests. The parent process
publishes a canonical matrix only after checking complete Cartesian coverage,
dependency hashes, and raw-artifact hashes. This is an execution-throughput
change, not a protocol, sample-size, or reported systems-performance change.

Before any held-out causal-factorial shard was generated, the execution
contract was amended to reuse a forward only when the complete frozen
`SameTokenControllerConfig` has both the same canonical SHA-256 digest and exact
dataclass equality within the same paired batch. Every arm-conversation record
is retained, while executed and reused forwards are counted separately and a
digest collision fails the shard. The amendment and its claim boundary are
recorded in
[`2026-07-14-p2-causal-exact-config-reuse.md`](reports/2026-07-14-p2-causal-exact-config-reuse.md).
The same pre-execution amendment adds fixed top-p 0.5 and 0.8 to every causal
cell, bringing the full factorial to 16 arms and 2.88 million quality
arm-conversations. A target-aware best-registered-arm result is computed per
complete conversation as an offline upper bound and is explicitly excluded
from calibration and the primary causal gate.

The S151 paired pilot is also complete. Hierarchical versus fixed was +5.89 pp
at 1x, +2.68 pp at 2x, and 0 pp at 4x; the 1x difference again came from
protected instruction retention. Exact no-pin predictions matched fixed 1x
(30/80 instruction queries), while pins reached 63/80. At 4x calibrated and
fixed both scored 245/560, with calibrated using 86.7% of fixed H2D traffic.
S151 rejected chunk=2 on an exact-prediction counterexample, so its large-scale
quality path uses token-sized cache steps. Chunk=1 was subsequently validated
on all 1,260 core-policy records and 3,920 predictions with exact agreement to
the physical-tier token reference. The frozen P2 quality contract is therefore
S55 chunk=2 and S151 chunk=1.

The matching S151 held-out pilot is also complete. Hierarchical versus fixed
was +5.89 pp at 1x, +2.68 pp at 2x, and tied at 4x; all net 1x/2x gains came
from instruction persistence, while native remained best overall. Dense
fallback again had zero net accuracy benefit. Unlike S55, S151 failed exact
chunk=2 validation; the clean-source chunk=1 audit subsequently matched all
1,260 core-policy records and 3,920 predictions exactly. Its full quality
matrix therefore uses token-by-token cache evaluation.

## P2 training matrix

The preregistered 5-seed × 2-scale Tier-S training matrix is complete. All ten
runs reached exactly 1,000 steps from one clean source commit, and an independent
audit revalidated every checkpoint byte count and SHA-256 digest. No low-quality
seed was excluded. Final native validation varied substantially (S55 mean
0.7542, sample SD 0.0975, range 0.2552; S151 mean 0.8135, sample SD 0.1016,
range 0.2604), confirming that the previous one-seed evidence was not adequate
for a paper-level claim. See the checked P2 training report and summary.

This completes only the training prerequisite. The nine-family, 1,000-example,
multi-context, multi-budget held-out policy matrix and its paired statistical
analysis remain active P2 work.
