# Experiment-scale audit against representative papers

## Decision

The expanded protocol is no longer small by number of experimental units. It
contains 4,500 core shards (630,000 policy-example evaluations), 9,000 causal
shards (2.52 million policy-example evaluations), 253,500 small-model RULER
predictions, 90,578 Qwen3-4B natural-benchmark predictions, 3,600 paired safety
predictions, 7,254 paired natural-safety generations, and two 108-cell systems
matrices with 30 measured repetitions per cell. The natural-safety suite covers
6,172 LongSafety generations and 1,082 IFEval generations with official
deterministic scoring; the paid LongSafety judge remains blocked. These units
are reported separately and are never combined into a
misleading headline total.

The remaining weakness is breadth, not raw count. The primary natural and
safety evidence uses one compatible model family, actual-model evaluation stops
at 128K, and official DeepSeek-V4 execution remains blocked by the frozen
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
3. Keep the 3-arm safety contrast same-budget and verify physical KV bytes,
   paired inputs, bootstrap uncertainty, and the exact paired test.
4. Execute the two-arm natural-safety suite, score IFEval with pinned official
   code, and report LongSafety only as digest-bound generation evidence while
   its paid judge is blocked.
5. Keep reference PyTorch, checked static continuous batching, and unavailable
   external production runtime as separate evidence tiers.
6. After primary completion, add a second instruction-tuned model family only
   when the causal result is positive or scientifically ambiguous. A clearly
   negative primary causal result should be published as bounded negative
   evidence instead of multiplying compute to search for a favorable model.

The machine-readable contract is
[`experiment-scale-audit-v1.json`](../manifests/experiment-scale-audit-v1.json).
