# Experiment-scale audit against representative papers

## Decision

The expanded protocol is no longer small by number of experimental units. It
contains 4,500 core shards (630,000 policy-example evaluations), 9,000 causal
shards (2.88 million policy-example evaluations across 16 arms), and a separate
post-P2 online learned-lookahead study with 6,750 label shards, 20 fitted
policies, and 9,000 held-out shards (1.08 million arm-conversations across six
arms). It also contains 253,500 small-model RULER
predictions, 90,578 Qwen3-4B natural-benchmark predictions, 3,600 paired safety
predictions, 7,254 paired natural-safety generations, and two 216-cell systems
matrices with 30 measured repetitions per cell. A separate 500K-token feasibility
preflight contributes four terminal scale-policy attempts but no performance sample.
The natural-safety suite covers
6,172 LongSafety generations and 1,082 IFEval generations with official
deterministic scoring; the paid LongSafety judge remains blocked. These units
are reported separately and are never combined into a
misleading headline total.

This density does not turn all examples into independent replications. The
primary trained-model inference unit is the checkpoint seed, with five
independent seeds per scale. A paired two-sided sign-flip test therefore has
only 32 possible assignments and a minimum attainable p-value of 0.0625.
The analysis enumerates those assignments exactly, reports multiplicity-adjusted
p-values as resolution-limited descriptive evidence, and never treats the
10,000 within-seed bootstrap draws as 10,000 independent models. The positive
gate instead requires a corrected seed-cluster interval above zero, matched
physical memory, and a positive effect in all five seeds. This is a real
inferential limitation even though the within-seed sample count is large.

Execution is scheduled with three disjoint same-accelerator workers for the
remaining S151 core scale, the causal factorial, and the online learned
lookahead study. This does not multiply experimental units: coordinates remain
exactly those frozen above. It reduces idle accelerator time during the
long-generation slices and eliminates per-shard checkpoint reloads in the
online study. Canonical matrices are emitted only after exact-count,
no-overlap, implementation-digest, dependency-digest, and raw-artifact checks;
concurrent shard wall time is not used as P4 performance evidence.

The remaining weaknesses are breadth and independent-seed resolution, not raw
count. The primary natural and safety evidence uses one compatible model family,
actual-model evaluation stops at 128K, and the 500K preflight uses only the two
Tier-S reference scales. Official DeepSeek-V4 execution remains blocked by the frozen
weights/runtime contract. The paper must therefore be framed as a deep,
digest-bound single-compatible-model study unless a later cross-family
replication is completed.

## Comparator audit

| Work | Reported breadth | Position of this study |
| --- | --- | --- |
| [SCBench](https://arxiv.org/abs/2412.10319) | 12 tasks, two shared-context modes, four capability categories, eight LLMs | Our example accounting and safety contrast are deeper, but model breadth is narrower. |
| [SnapKV](https://arxiv.org/abs/2404.14469) | 16 long-sequence datasets and a 380K NIAH demonstration | Our lifecycle and failure analysis are broader; its dataset count and maximum demonstrated length are stronger. |
| [FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079) | RULER, LongBench-v2, LongMemEval, and 500K physical-cache evidence on V4 | It remains the stronger direct-architecture comparator. We cannot substitute Qwen3 evidence for it. |
| [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) | Multi-instruction degradation and system-prompt leakage across eviction choices | Our four-family synthetic stress and paired LongSafety/IFEval generation expand coverage, but LongSafety remains generation-only until the official paid judge runs. |
| [Benchmarking KV-Cache Optimizations across Task Quality and System Performance](https://arxiv.org/abs/2607.05399) | Two model families, four workload categories, quality, TTFT, throughput, realized compression | Our tail repetitions and metric closure are stronger; cross-family coverage is weaker. |

## Frozen response

1. Finish the preregistered primary study before adding an open-ended model
   sweep.
2. Report every operational failure as zero in conservative quality aggregates
   and retain worst family×context slices.
3. Retain fixed top-p 0.5/0.8 on the full causal grid and report the
   target-aware registered-arm oracle only as a non-causal descriptive upper
   bound.
4. Keep the 3-arm safety contrast same-budget and verify physical KV bytes,
   paired inputs, bootstrap uncertainty, and the exact paired test.
5. Execute the two-arm natural-safety suite, score IFEval with pinned official
   code plus revision-bound `punkt`/`punkt_tab` data, and report LongSafety only
   as digest-bound generation evidence while its paid judge is blocked.
6. Keep the single-attempt 500K feasibility preflight, repeated reference PyTorch,
   checked static continuous batching, and unavailable external production runtime
   as separate evidence tiers.
7. After primary completion, add a second instruction-tuned model family only
   when the causal result is positive or scientifically ambiguous. A clearly
   negative primary causal result should be published as bounded negative
   evidence instead of multiplying compute to search for a favorable model.

The machine-readable contract is
[`experiment-scale-audit-v1.json`](../manifests/experiment-scale-audit-v1.json).
