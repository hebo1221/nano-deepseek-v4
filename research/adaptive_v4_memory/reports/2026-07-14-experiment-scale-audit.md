# Experiment-scale audit against representative papers

## Decision

The expanded protocol is no longer small by number of experimental units. Its
immutable five-seed primary cohort contains 4,500 core shards (630,000
policy-example evaluations) and 9,000 causal shards (2.88 million
policy-example evaluations across 16 arms). A disjoint four-seed extension adds
3,600 core and 7,200 causal shards, yielding contract-audited nine-seed totals
of 8,100 core and 16,200 causal shards. The study also contains a separate
post-P2 online learned-lookahead study with 6,750 label shards, 20 fitted
policies, and 9,000 held-out shards (1.08 million arm-conversations across six
arms). It also contains 370,500 small-model RULER predictions across a 57-cell
screen that includes token-, layer-, and head-adaptive cache baselines, 90,578
Qwen3-4B natural-benchmark predictions, 7,800 preregistered Phi-4-mini
cross-family RULER predictions, 3,600 paired safety
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
independent seeds per scale. Its paired two-sided sign-flip test has only 32
assignments and a minimum attainable p-value of 0.0625. The separately reported
confirmatory analysis pools the four extension seeds only after an
identical-contract audit, giving nine seeds, 512 assignments, and a minimum
two-sided p-value of 0.00390625. Neither analysis treats the 10,000 within-seed
bootstrap draws as independent models. The positive gate additionally requires
a corrected seed-cluster interval above zero, matched physical memory, and a
positive effect in every seed of the relevant cohort.

Execution is scheduled with three disjoint same-accelerator workers for the
remaining S151 core scale, the causal factorial, and the online learned
lookahead study. This does not multiply experimental units: coordinates remain
exactly those frozen above. It reduces idle accelerator time during the
long-generation slices and eliminates per-shard checkpoint reloads in the
online study. Canonical matrices are emitted only after exact-count,
no-overlap, implementation-digest, dependency-digest, and raw-artifact checks;
concurrent shard wall time is not used as P4 performance evidence.

The remaining weakness is breadth beyond a bounded two-family transfer, not raw
count. The full natural and safety suite uses one compatible model family,
actual-model evaluation stops at 128K, and the 500K preflight uses only the two
Tier-S reference scales. Official DeepSeek-V4 execution remains blocked by the frozen
weights/runtime contract. The paper must therefore be framed as a deep,
digest-bound Qwen3 natural study with a separately reported Phi-4 RULER
transfer replication, not a broad model-population study.

## Comparator audit

| Work | Reported breadth | Position of this study |
| --- | --- | --- |
| [RULER](https://arxiv.org/abs/2404.06654) | 13 representative tasks across 17 long-context models | Our five-length, 500-sample-per-task matrix is denser per compatible model, but cannot support the same cross-model generalization. |
| [SCBench](https://arxiv.org/abs/2412.10319) | 12 tasks, two shared-context modes, four capability categories, eight LLMs | Our example accounting and safety contrast are deeper, but model breadth is narrower. |
| [SnapKV](https://arxiv.org/abs/2404.14469) | 16 long-sequence datasets and a 380K NIAH demonstration | Our lifecycle and failure analysis are broader; its dataset count and maximum demonstrated length are stronger. |
| [PyramidKV](https://arxiv.org/abs/2406.02069) | LongBench plus NIAH, including 12% and 0.7% retained-cache regimes | Our three matched compression ratios and lifecycle stresses give a broader operating grid; its 70B NIAH and extreme-compression evidence remain stronger. |
| [Ada-KV](https://arxiv.org/abs/2407.11550) | 13 RULER and 16 LongBench datasets in both question-aware and question-agnostic settings | Our screen includes an Ada-KV-wrapped SnapKV arm, but its dataset and query-regime breadth is substantially stronger. |
| [FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079) | RULER, LongBench-v2, LongMemEval, and 500K physical-cache evidence on V4 | It remains the stronger direct-architecture comparator. We cannot substitute Qwen3 evidence for it. |
| [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) | Multi-instruction degradation and system-prompt leakage across eviction choices | Our four-family synthetic stress and paired LongSafety/IFEval generation expand coverage, but LongSafety remains generation-only until the official paid judge runs. |
| [Benchmarking KV-Cache Optimizations across Task Quality and System Performance](https://arxiv.org/abs/2607.05399) | Two model families, four workload categories, quality, TTFT, throughput, realized compression | We now match its two-family count on bounded RULER transfer and retain deeper tail accounting, but its multi-workload coverage on both families remains broader. |

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
7. Execute the preregistered Phi-4-mini transfer cohort regardless of the Qwen
   or Phi result. It inherits the Qwen-selected 50%-KV operating point without
   Phi-specific tuning; a failed transfer remains bounded negative evidence.

The machine-readable contract is
[`experiment-scale-audit-v1.json`](../manifests/experiment-scale-audit-v1.json).

## Independent-seed confirmatory extension

Before inspecting any P2 policy accuracy, paired effect, family result, seed
result, or quality-gate output, the study added a separate four-seed extension.
The original five-seed matrices remain immutable and independently reportable.
The extension uses training seeds 6071406--6071409 with disjoint calibration
and evaluation seed namespaces, the same two scales, all nine families, all
five contexts, ten replicates per context, and no outcome-dependent early
stopping.

This changes the meaningful inferential quantity, not just the headline row
count. Nine independent training seeds provide 512 exact sign assignments and
a minimum two-sided seed-level p-value of 0.00390625. Even the theoretical
Holm floor across nine primary family hypotheses becomes 0.03515625 instead of
being structurally above 0.05. P-values remain only one part of the gate: effect
size, seed-cluster uncertainty, physical memory matching, direction across
seeds, and worst-slice regressions remain mandatory.

The additive extension contains 3,600 core shards (504,000 policy-example
evaluations) and 7,200 causal shards (2.304 million policy-example evaluations).
If and only if the implementation, workload, policy, budget, pairing, and audit
contracts are identical, the combined study contains 8,100 core shards and
16,200 causal shards across nine independent checkpoints per scale. The frozen
contract is
[`p2-independent-seed-extension-v1.json`](../manifests/p2-independent-seed-extension-v1.json).
