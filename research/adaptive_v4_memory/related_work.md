# Related work and novelty boundary

Last reviewed: 2026-07-14

This document tracks the literature that can invalidate, constrain, or inspire
Adaptive V4 Memory. It is deliberately broader than KV eviction because V4
combines compression, sparse retrieval, hybrid cache state, and tiered storage.

Legend used below:

- **write**: controls which entries enter long-term memory;
- **read**: controls which existing entries attention reads;
- **residency**: moves recoverable entries between memory tiers;
- **reuse**: shares KV or sparse indices across layers/queries;
- **precision**: changes bytes per entry rather than entry count.

## 1. Native architecture

| Work | Main mechanism | Adaptation | Key result or limitation | Implication for this project |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | CSA 4x overlapping compression + Lightning Indexer, HCA 128x dense compression, sliding branch, heterogeneous state cache | Mostly fixed after training | Reports about 2% of BF16 GQA8 KV size at 1M context; cache policies and layouts vary by layer | The optimization unit is a compressed block/layer/tier, not an ordinary KV head |
| [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | Hardware-aligned compression, selection, and sliding branches | Learned during training | Sparse design must be co-designed with kernels to produce real speedups | A logical budget policy without runtime support is insufficient |

V4's shared K=V MQA is an important boundary condition. Methods that allocate
independent cache entries across MHA/GQA heads are useful evidence for
heterogeneous demand, but their action space does not transfer directly.

## 2. Token retention and eviction

| Work | Action | Adaptation granularity | Training | Main lesson for V4 |
| --- | --- | --- | --- | --- |
| [H2O](https://arxiv.org/abs/2306.14048) | write/evict | token, decode-time | none | Recent tokens and accumulated heavy hitters complement one another, but retained importance is query-history dependent |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | write/evict | fixed sinks + recent window | none | A small stable state prevents catastrophic streaming collapse; V4 already incorporates a learnable sink and sliding state |
| [SnapKV](https://arxiv.org/abs/2404.14469) | evict | head/token from prompt observation window | none | Future attention can often be predicted from the end of the prompt, but a later unrelated query can invalidate the decision |
| [RocketKV](https://arxiv.org/abs/2502.14051) | evict + sparse read | coarse eviction followed by fine sparse selection | none | Two-stage pruning can improve the memory/compute frontier; both stages need separate accounting |
| [EMS](https://arxiv.org/abs/2412.08521) | evict then merge | head/token, global-local importance | none | Merging can retain information discarded by pure eviction; representation error must be measured, not only token recall |
| [NestedKV](https://arxiv.org/abs/2605.26678) | evict | token across global/block/local time scales | none | A single importance signal is brittle; multi-timescale novelty and adaptive budgets are useful controller features |
| [KVzip](https://arxiv.org/abs/2505.23416) | evict | query-agnostic token importance via context reconstruction | none | A reusable cache should not depend on one known question; reconstruction provides an offline oracle candidate |
| [KVzap](https://arxiv.org/abs/2601.07891) | evict | fast input-adaptive approximation of KVzip | none | An expensive oracle can supervise a practical controller, but approximation error must be stress-tested |
| [Self-Pruned KV Attention](https://arxiv.org/abs/2605.14037) | learned write | token/layer/KV-head utility threshold | continued pretraining | Learned utility exposes true non-uniform memory demand; irreversible write suppression remains risky under query shift |
| [IndexMem](https://arxiv.org/abs/2605.25475) | learned index + latent residual memory | token, periodic decode refresh | learned indexer and memory | Future-importance prediction can work, but lossy latent recovery and exact cold-tier recovery are different evidence tiers |
| [KVReviver](https://arxiv.org/abs/2512.17917) | evict + reconstruct | compressed sketches | none | Approximate reversibility is a useful baseline; reconstruction error must be separated from residency misses |

The relevant negative lesson is that deletion and residency are not equivalent.
Adaptive V4 Memory keeps compressed entries recoverable and treats an omitted
GPU entry as a cache miss, not permanent information loss.

## 3. Layer-, head-, and task-adaptive budgets

| Work | Budget axis | Decision signal | Online behavior | Transfer to V4 |
| --- | --- | --- | --- | --- |
| [PyramidKV](https://arxiv.org/abs/2406.02069) | layer | pyramidal information funneling | fixed layer schedule after profiling | Supports non-uniform layer budgets; V4 CSA/HCA types need separate policies |
| [SqueezeAttention](https://arxiv.org/abs/2404.04793) | layer | prompt-time input/output cosine similarity + clustering | task/prompt adaptive, then fixed during decode | Provides a training-free layer allocator baseline, but does not react to later turns |
| [Ada-KV](https://arxiv.org/abs/2407.11550) | attention head | globally top-ranked attention mass under a total budget | prompt/query-aware head budgets | Its loss-bound idea is useful; primary KV-head allocation does not directly apply to MQA |
| [DynamicKV](https://arxiv.org/abs/2412.14838) | layer/token | task-specific layer activation patterns | periodically updates earlier layer budgets | Closest baseline for time-varying layer budgets, though designed for conventional caches |
| [ZigZagKV](https://arxiv.org/abs/2412.09036) | layer | layer uncertainty | dynamic layer allocation | Motivates uncertainty as a controller signal; uncertainty calibration must be validated |
| [BaKLaVa](https://arxiv.org/abs/2502.13176) | head/layer | one-time per-cache importance profiling | profiled once, then fixed | Direct precedent for non-uniform calibrated quotas; its static profile is a baseline rather than evidence for later query shifts |
| [ReFreeKV](https://arxiv.org/abs/2502.16886) | input/global budget | attention-derived full-cache preservation risk | threshold-free input-adaptive pruning | The closest conventional-cache analogue to a risk-triggered budget; irreversible pruning and V4 residency remain distinct action spaces |
| [LAVa](https://arxiv.org/abs/2509.09754) | head/layer | attention-output and cross-layer residual-stream loss | training-free dynamic head and layer budgets | Makes cross-layer output loss a required comparator signal; head allocation does not transfer directly to V4 MQA |
| [WindowKV](https://arxiv.org/abs/2503.17922) | group/layer/window | task-adaptive semantic windows | layer groups share indices | Contiguous semantic windows and group-wise reuse may reduce metadata and transfer fragmentation |
| [RazorAttention](https://arxiv.org/abs/2407.15891) | head | retrieval-head identification | full cache only for retrieval heads | The retrieval/local specialization principle may apply to V4 indexer heads, not primary KV entries |
| [DuoAttention](https://arxiv.org/abs/2410.10819) | head | optimization on synthetic retrieval data | full retrieval heads + constant streaming heads | Demonstrates data-driven hybrid layouts and the need for retrieval-focused calibration |
| [CompressKV](https://arxiv.org/abs/2508.02401) | head/layer/token | semantic retrieval heads and layer-wise eviction error | task-sensitive retained set | Supports layer-specific error budgets and retrieval-head signals |
| [KV-CoRE](https://arxiv.org/abs/2602.05929) | layer/data/domain | normalized effective rank | evaluation framework | Shows compressibility is data- and layer-dependent; useful as an analysis metric, not a controller alone |
| [PolyKV](https://arxiv.org/abs/2606.15157) | layer, phase, policy | calibration perturbation and entropy/PPL sensitivity | calibrated prefill/decode policies | Policy heterogeneity can matter more than budget heterogeneity; very small non-uniform budgets can starve layers |
| [ARKV](https://arxiv.org/abs/2603.08727) | layer/precision/retention | prefill attention entropy, variance, and kurtosis | prompt-calibrated tri-state action | Supports jointly measuring retain/quantize/evict choices, but precision is deferred here to avoid a residency confound |

## 4. Sparse reads and index computation

| Work | Read/index mechanism | Adaptivity | Main result or risk | V4 relevance |
| --- | --- | --- | --- | --- |
| [MInference](https://arxiv.org/abs/2407.02490) | per-head A-shape, vertical-slash, or block-sparse pattern | offline pattern + dynamic indices | Large prefill gains with model-specific patterns | Pattern discovery and hardware kernels are useful baselines for prefill, not V4 residency |
| [SeerAttention](https://arxiv.org/abs/2410.13276) | learned block gate | query/block adaptive | Learned sparsity beats fixed heuristics but needs training and custom kernels | A learned block selector is an alternative to score-statistic control |
| [RetrievalAttention](https://arxiv.org/abs/2409.10516) | CPU ANN over KV vectors | query adaptive | Preserves full logical history but query/key OOD complicates retrieval | Strong precedent for reversible cold retrieval and explicit transfer costs |
| [IndexCache](https://arxiv.org/abs/2603.12201) | reuse Lightning Indexer top-k across layers | calibrated static pattern or training-aware sharing | Removing 75% of indexers can preserve quality; uniform removal fails without adaptation | Direct baseline for indexer-compute reduction; our policy must add time/query adaptation |
| [You Only Index Once](https://arxiv.org/abs/2606.06467) | share KV and one routing index across cross-decoder layers | architectural/static | Amortizes fine-grained routing in a KV-sharing model | Defines the upper end of architectural sharing but requires retraining a different backbone |
| [LaProx](https://arxiv.org/abs/2605.07234) | output-aware cache scoring | query/layer adaptive | Attention mass alone can mis-rank entries whose projected values dominate the output | Add projected value/output contribution as a baseline signal instead of relying only on index scores |
| [Expected Attention](https://arxiv.org/abs/2510.00636) | expected future-query importance | training-free future-query distribution | Combines future attention probability with output contribution | Provides a training-free lookahead baseline between current-score control and a learned predictor |
| [SAGE-KV](https://arxiv.org/abs/2503.08879) | one-shot head/token selection after prefill | prompt observation only | Avoids decode-time updates but cannot react to query shift | Useful static post-prefill baseline; multi-turn stress is required |

Sparse-read work establishes that storing a logical cache and reading a physical
working set are separate decisions. Adaptive V4 Memory focuses on their joint
control under the native V4 architecture.

## 5. Direct V4 memory predecessor and serving systems

| Work | Mechanism | Fixed choices | Reported boundary | Open question retained here |
| --- | --- | --- | --- | --- |
| [FlashMemory-DeepSeek-V4](https://arxiv.org/abs/2606.09079) | CPU cold pool, lookahead Memory Indexer, GPU working set, native second-stage top-k | refresh interval 64, threshold 0.5, layers 10/12/20, OR aggregation | False positives grow with irrelevant context; dense MRCR memory fails | Can uncertainty-controlled budgets, refresh, and fallback handle both sparse and dense demand? |
| [Tangram](https://arxiv.org/abs/2606.06302) | static reservation, ragged paging, ahead-of-time load balancing for non-uniform budgets | calibrated head ranking and bounded ratios | Dynamic heterogeneity otherwise causes fragmentation, reclamation, and load imbalance | How should V4 block classes and adaptive layer budgets map to real pages and batches? |
| [NOSA](https://arxiv.org/abs/2510.13602) | query-aware and query-agnostic locality for CPU offload | learned transfer selection | Targets long generation by reducing unnecessary transfers | Strong P4 transfer-aware comparison; useful bytes and late misses must accompany quality |

FlashMemory is the minimum direct baseline. A method that only replaces its
threshold with another fixed threshold is not a research contribution.

## 6. Quantization and representation size

| Work | Mechanism | Lesson |
| --- | --- | --- |
| [KIVI](https://arxiv.org/abs/2402.02750) | asymmetric 2-bit quantization: keys per-channel, values per-token | Key and value distributions require different quantizers |
| [KVTuner](https://arxiv.org/abs/2502.04420) | offline layer-wise mixed-precision search | Layer sensitivity can be calibrated and reused online with low overhead |

V4 already uses mixed FP8/BF16 entries. Precision adaptation is an optional
later axis; combining it with residency in the first study would confound the
source of gains.

## 7. Evaluation, failure, and reproducibility literature

| Work | What it exposes | Required response in our protocol |
| --- | --- | --- |
| [SCBench](https://arxiv.org/abs/2412.10319) | Single-query compression can collapse in later turns; importance distributions shift during long generation | Evaluate shared-prefix multi-turn and multi-request workloads; preserve cold logical memory |
| [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) | Compression can selectively erase instructions and increase system-prompt leakage | Pin and separately score instructions; report worst instruction and leakage results |
| [Key, Value, Compress](https://arxiv.org/abs/2503.11816) | Existing comparisons mix models, data, batch sizes, hardware, and incomplete latency metrics | Run all core policies in one harness and report memory, throughput, and quality together |
| [The Risk of KV Cache Compression](https://arxiv.org/abs/2607.01520) | Causal masking creates workload-dependent intrinsic compressibility and minimax risk | State bounded or negative claims and report worst slices instead of assuming universal safe compression |

For conventional-model baselines, the official
[KVPress](https://github.com/NVIDIA/kvpress) library supplies a common interface
for more than twenty compression methods, RULER/NIAH evaluation, per-layer
compression, and threshold policies. P3 will pin a repository revision and use
it rather than reimplementing selected baselines with incompatible harnesses.
The frozen Qwen3 screen evaluates StreamingLLM, SnapKV, Expected Attention,
Critical Expected Attention, PyramidKV, and Ada-KV-wrapped SnapKV at 25%, 50%,
and 75% compression. This covers token-, layer-, and head-adaptive retention
before selecting the 50%-compression baseline for the natural benchmarks.

## 8. Novelty boundary matrix

`Partial` means the work handles the property only in a fixed, indirect, or
limited form.

| Method | V4-native | Full cold history | Query/time budget | Per-layer budget | Adaptive index execution | Risk fallback | Multi-turn reuse target | Transfer-aware |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native V4 | Yes | Partial (prefix storage) | No | Fixed by layer type | No | Native full path | Partial | Partial |
| Ada-KV | No | No | Partial | No | No | No | No | No |
| SqueezeAttention | No | No | Prompt only | Yes | No | No | No | No |
| BaKLaVa | No | No | Profile only | Yes | No | No | No | No |
| ReFreeKV | No | No | Input-adaptive | Partial | No | Risk threshold | No | No |
| LAVa | No | No | Query-adaptive | Yes | No | No | No | No |
| IndexCache | Sparse-index model | N/A | No | Indexer on/off | Static/calibrated | No | No | No |
| SP-KV | No | No | Per-token write gate | Yes | N/A | Threshold only | No | Kernel-aware |
| FlashMemory-V4 | Yes | Yes | Fixed threshold/interval | Three fixed predictors | Periodic fixed | No | Limited | Yes |
| Adaptive V4 Memory (planned) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

The final row is a research specification, not an achieved feature list.

## 9. Literature-derived design rules

1. **Do not conflate logical compression and physical residency.** A cold entry
   remains recoverable and must be counted separately from GPU bytes.
2. **Do not assume head-wise cache freedom.** V4 primary attention uses MQA;
   indexer-head statistics are signals, not independently stored KV heads.
3. **Allocate for tail risk, not only mean attention mass.** Dense global tasks
   and instructions can be rare but catastrophic when missed.
4. **Treat query awareness as a cache-reuse hazard.** A cache optimized for the
   current question may be wrong for the next turn.
5. **Measure causal necessity when possible.** Attention score, effective rank,
   and cosine similarity are proxies; replay ablations establish whether an
   entry was actually needed.
6. **Account for controller and movement cost.** Extra index passes, host-device
   transfers, ragged pages, and scheduling can erase theoretical gains.
7. **Keep a native fallback.** The controller must widen rather than confidently
   discard when score density or disagreement signals a hard case.
8. **Avoid layer starvation at the minimum budget.** Keep the 1x allocation
   uniform and test calibration-only non-uniform quotas at 2x and 4x, where a
   floor can be preserved for every CSA layer.
9. **Score output consequence, not attention alone.** Compare Lightning Indexer
   concentration with projected value/output contribution and expected-future
   attention before attributing gains to a learned controller.
10. **Separate phase-specific policy from budget.** Prefill policy, decode
    policy, and layer budget are distinct ablations; a gain from one cannot be
    credited to the others.

## 10. Reading queue

The following work is relevant but not yet part of the core comparison table.
It should be reviewed before expanding the controller action space:

- [MiniCache](https://arxiv.org/abs/2405.14366): depth-wise KV merging;
- [ClusterKV](https://arxiv.org/abs/2412.03213): semantic-space cache clusters;
- [Twilight](https://arxiv.org/abs/2502.02770): hierarchical top-p pruning;
- [FastKV](https://arxiv.org/abs/2502.01068): separating context reduction from
  decode cache compression;
- [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574): retrieval-head causal analysis;
- [The Sparse Frontier](https://arxiv.org/abs/2504.17768): sparse-attention
  trade-offs; and
- [Understanding the Physics of KV Cache Compression](https://arxiv.org/abs/2603.01426): attention-dynamics analysis.
