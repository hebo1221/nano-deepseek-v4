# Adaptive V4 Memory: 2026-07-17 deep-research swarm decision

Snapshot: 2026-07-17 (Asia/Seoul)

Status: literature-backed strategy decision; hypotheses and proposed kill gates,
not an empirical claim or a change to the frozen P2 execution

## 1. Decision first

The earlier primary bet, **HCA-guided same-token physical CSA prefetch**, is
demoted from a main contribution to a hypothesis that must earn its place in a
trace-only study. Coarse-to-fine routing, cross-layer same-token prefetch,
lossless speculative recall, V4-native cold residency, and summary-to-detail
recovery are already substantially covered by
[NSA](https://arxiv.org/abs/2502.11089),
[InfiniGen](https://arxiv.org/abs/2406.19707),
[ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda),
[FlashMemory-V4](https://arxiv.org/abs/2606.09079), and
[SeKV](https://arxiv.org/abs/2606.31145). Calling their intersection a new
prefetch hierarchy would be too broad.

The strongest remaining research question is narrower and more scientific:

> **How much conditional information about later CSA relevance remains in
> DeepSeek-V4's already-computed HCA path, beyond causal adjacent-CSA anchors
> and cheap second-order block statistics; and can HCA--CSA disagreement decide
> which queries require a higher statistical-order read or the unmodified
> native CSA scan?**

The candidate contribution is therefore a **V4-native, query-adaptive
statistic-acquisition cascade**, not another eviction score:

1. reuse a causal prior-CSA anchor and the existing HCA path at zero additional
   predictor-model cost;
2. treat HCA primarily as a sufficiency/failure sentinel, not assume that it is
   a reliable directory;
3. read a CSA-side second-order sketch only for HCA-ambiguous spans; and
4. fall through to the unmodified Lightning Indexer when uncertainty remains.

"Exact" in this proposal can only mean equivalence to **native V4 CSA
semantics**: the original compressed-block sparse top-k. It does not mean dense
full-KV equivalence. Without a final full index scan or a guaranteed-recall
procedure, the work must not use lossless or exact wording.

This is a conditional **GO for H0 instrumentation only**. It is a **NO-GO for
new offload/runtime construction** until the signal, sufficiency, and timing
gates below pass. The running frozen P2 synthetic core is unchanged.

## 2. How this audit was performed

Three parallel reviews were combined and then adversarially cross-checked:

- a frontier scan of July 2026 arXiv, ICML 2026 primary pages, and official
  paper/project pages;
- a novelty red team that searched for the closest mechanism, not merely the
  closest title; and
- a repository/systems red team that checked causal layer ordering, transfer
  slack, current M4 behavior, and implementability in this codebase.

The scan covered selection and indexing, representation compression,
hierarchical recovery, offload and prefetch, verified execution, learned
memory architectures, workload reuse, risk theory, and evaluation protocol.
Reported numbers below are **author-reported** unless explicitly labeled as a
repository observation. They have not been independently reproduced here.
A failure to find an identical paper is not evidence that none exists.

## 3. What changed at the frontier

| Frontier result | What it establishes | Consequence for Adaptive V4 |
| --- | --- | --- |
| [ECHO, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) | exact final sparse top-k with speculative intra-query/inter-query KV recall and a graph-friendly GPU cache manager; up to 2.1x generation throughput is reported | strongest system baseline; generic score-guided physical prefetch and guaranteed final recall are already occupied |
| [LiteTopK](https://arxiv.org/abs/2607.11976) | exact fused indexer-top-k avoids much of selection overhead | skipping or approximating the index is not automatically better than optimizing the exact path |
| [IndexCache](https://arxiv.org/abs/2603.12201) and [CLSA](https://arxiv.org/abs/2606.06467) | adjacent/cross-layer routing reuse can remove repeated index work | prior-CSA anchors are a mandatory causal baseline, not a new idea |
| [AsyncTLS](https://arxiv.org/abs/2604.07815) and [HiLS](https://arxiv.org/abs/2607.02980) | hierarchical block-to-token selection and asynchronous or learned coarse-to-fine retrieval | HCA-to-CSA hierarchy alone has weak novelty |
| [COBS](https://arxiv.org/abs/2607.09052) | first-order single-vector block summaries can miss within-block covariance; a second-order descriptor recovers much of the reported RULER gap | the assumption that a 128-token HCA summary is a sufficient 4-token CSA directory must be tested, not asserted |
| [Uncertainty-gated selection](https://arxiv.org/abs/2607.07724) | widens sparse selection when the top-k boundary is ambiguous | margin-based adaptive quota/fallback is no longer a contribution by itself |
| [Query Visibility Audit](https://arxiv.org/abs/2607.11942) | query-aware compression can degrade sharply when reused query-agnostically; backend and overflow choices can materially change scores | durable reusable memory and query-specific overlay require separate contracts and a factorial protocol |
| [MosaicKV](https://arxiv.org/abs/2607.00760) | jointly adapts sequence and channel compression with GPU/CPU management | generic multi-axis adaptive compression is crowded |
| [Lynx](https://arxiv.org/abs/2607.01831), [JoLT](https://arxiv.org/abs/2607.12550), and [VeriCache](https://arxiv.org/abs/2605.17613) | progressive precision transfer, low-rank/JL residual representation, and compressed-draft/exact-verify execution | a broad anchor-residual-detail ladder is not novel; the remaining question is selective acquisition in the native V4 cross-path |
| [Risk of KV Cache Compression](https://arxiv.org/abs/2607.01520) | minimax risk depends on query distribution and intrinsic response compressibility | "intrinsic compressibility" itself is prior art; a controller must declare its distribution and failure regime |
| [MiMo-V2.5](https://arxiv.org/abs/2607.13095), [SPIN](https://arxiv.org/abs/2604.26837), and [PEEK](https://arxiv.org/abs/2607.02525) | production systems already combine layerwise prefetch, hierarchical memory, prefix demand, placement, or scheduling | a generic hardware-aware cache controller is not a defensible headline |

Two apparently positive paths are thus also threats. Exact-kernel work can
make approximate index skipping unnecessary, while second-order descriptors
can make HCA's coarse first-order proposal insufficient. Adaptive V4 has to
beat both directions at matched bytes and native-CSA recall.

## 4. Field map: where the academic paths now lead

| Path | Current opportunity | Likely saturation or failure | Role here |
| --- | --- | --- | --- |
| attention/semantic eviction and pins | protects instructions, values, and structural roles | proxy scores fail under task and format shift; the axis is crowded | safety control and ablation only |
| future-utility/write prediction | can anticipate later use | future labels, checkpoint adaptation, and query shift are expensive | secondary baseline; not the headline |
| adaptive quota and uncertainty | spends budget on dense or ambiguous queries | controller cost and generic margin widening are prior art | only useful with risk and physical contracts |
| exact/fused sparse indexing | preserves semantics while removing selector overhead | does not reduce stored state by itself | strongest competing fast path |
| first/second-order block descriptors | improves routing information per block | descriptor bytes and reads can cost as much as the native scan | central scientific comparator |
| representation compression | directly reduces bytes | reconstruction, transient memory, and custom kernels can erase gains | optional recovery action |
| hierarchy/offload/recompute | increases effective capacity | PCIe/RDMA deadline misses and host synchronization dominate tails | only after trace/timing H0 passes |
| verification/certification | contains long-rollout divergence | verification can approach the cost of the exact path | reliability fallback |
| workload/prefix reuse | exploits real request demand | singleton traffic and policy/scheduler coupling limit generality | motivates durable/ephemeral split |
| memory-native training | can teach locality and compressibility | needs continued training and new weights | highest long-term ceiling, deferred |
| theory/sequential risk | identifies when compression is unsafe | worst-case bounds can be vacuous and calibration breaks under shift | possible contribution after causal labels exist |
| kernel/page/scheduler co-design | turns logical sparsity into real HBM and latency gains | fragmentation, launch, compaction, and transient state are hidden costs | mandatory systems proof, not first step |

The common direction is no longer "keep fewer tokens." It is **detect the
current information regime, acquire only the evidence needed to decide, and
retain an exact or verified escape path whose cost is measured physically**.

## 5. Claim-threat matrix

| Proposed claim | Closest prior art | Defensible residue | Current verdict |
| --- | --- | --- | --- |
| HCA-guided same-token CSA prefetch | InfiniGen, ECHO, AsyncTLS, FlashMemory-V4 | existing HCA as a zero-extra-model V4 proposal signal | low--medium; demoted |
| HCA as a coarse directory | NSA, HiLS, SeKV, COBS | empirical conditional-information characterization across native HCA/CSA paths | medium only if H0 wins |
| adaptive moment/residual ladder | COBS, SeKV, JoLT, Lynx, MosaicKV | per-query selective statistical-order acquisition with native fallback | medium; architecture-heavy |
| calibrated adaptive quota | uncertainty-gated selection, CONF-KV, DBudgetKV, runtime certification | finite-sample risk control over a predefined causal-damage event | medium; requires careful statistical scope |
| future option-value residency | ForesightKV, SP-KV, KVpop | time-to-critical-use jointly with transfer/recompute deadline | low--medium; defer |
| semantic pins | MemDecay, structural-role work, EpiCache | role-protected safety control at exactly matched bytes | low as novelty; retain as control |
| resource-priced action market | PEEK, SPIN, MosaicKV, ECHO, serving schedulers | V4-specific causal risk plus exact recovery under concurrency | low and costly; defer |

## 6. Revised hypotheses

### H0-A: native cross-path conditional information

At a causally valid HCA-to-later-CSA pair, HCA features contain held-out
information about native CSA selected blocks, attention mass, or causal output
damage **after conditioning on** previous-token and previous-CSA anchors,
recency/locality, query shift, and matched-cost block summaries.

This is not equivalent to claiming HCA is a block mean. HCA and later CSA use
different hidden states and learned compressors. COBS's theorem for a
query-independent single-vector selector cannot simply be transferred to HCA;
the relationship must be measured.

### H0-B: HCA as a failure sentinel

Cross-resolution disagreement between HCA and a cheap CSA-side descriptor is a
better predictor of incompressible/dense-memory queries, native-CSA misses, and
causal output damage than score margin, temporal instability, or query shift
alone.

This is stronger and more defensible than assuming HCA says *where* to fetch.
HCA may be more useful for saying *when the cheap representation is
insufficient*.

### H0-C: selective statistical-order acquisition

For non-ambiguous spans, HCA/prior-anchor evidence is enough. Only ambiguous
spans read a low-rank covariance or other second-order sketch. Remaining
uncertainty falls through to the unmodified native CSA indexer. At matched
native-CSA candidate recall, selective acquisition must read fewer descriptor
bytes and create fewer late misses than fixed COBS-style second-order reads.

### H1: dual-contract memory

A reusable query-agnostic durable base and a non-reusable query-aware ephemeral
overlay beat a single policy when the same prefix receives multiple queries.
The two contracts must be scored separately; query-visible compression cannot
silently be reused as query-agnostic state.

### H2: sequential causal-risk fallback

On whole-conversation calibration units, a predefined causal-damage event can
be controlled at a declared marginal risk target by widening or invoking the
native CSA path. This is not conditional or worst-case correctness, and OOD
workloads require an explicit fallback.

## 7. H0 experiment: learn before building

### 7.1 Instrumentation

Add an opt-in, trace-only research probe that does not alter logits or cache
state:

- expose HCA span scores/probabilities and output contribution;
- align each HCA span with the contained four-token CSA compressed entries,
  including boundary metadata;
- record previous-token and causally earlier-CSA selected indices;
- record later native CSA top-k, attention mass, rank margin, and index cost;
- label causal damage through frozen patch/replay measurements such as output
  KL, top-logit change, or task-specific loss; and
- timestamp signal availability, issue time, native need time, and measured
  transfer completion.

No future CSA feature may enter a same-token predictor before its decision
point. Alignment, query position, request identity, and layer order must be
digest-bound in the trace.

### 7.2 Required baselines

1. random and local/recency;
2. previous-token attention or selection;
3. prior-CSA exact selected anchors / IndexCache-style reuse;
4. ECHO-style moving threshold or score history;
5. HCA-only and HCA plus query-shift gate;
6. mean/centroid or Mosaic-style first-order summary;
7. fixed COBS-style low-rank covariance descriptor;
8. unmodified native Lightning Indexer; and
9. an oracle used only to define the ceiling.

Every comparison must match candidate count, descriptor bytes, or measured
read/compute cost. Comparing a free-looking HCA signal against a much smaller
baseline without counting alignment metadata would be invalid.

### 7.3 Measurements

- selected-block recall, attention-mass recall, and worst-head/layer recall;
- incremental log loss/deviance or conditional-information proxy over the
  strongest causal baseline;
- precision-recall for native miss and causal-damage events;
- calibration error, risk--coverage, fallback frequency, and worst slice;
- descriptor HBM bytes, index bytes, metadata, and transient allocation;
- available causal lead time, p50/p95 transfer time, useful-transfer fraction,
  late-miss rate, and rollback cost; and
- query-aware versus query-agnostic reuse as an explicit factorial.

### 7.4 Proposed preregistration gates

The exact constants should be frozen after a baseline-only pilot and before
the candidate arm is revealed. The following are minimum design targets, not
observed results:

| Gate | Continue only if |
| --- | --- |
| conditional signal | HCA features improve the prespecified primary recall/damage metric over the strongest causal matched-cost baseline on both scales, with a conversation/seed-clustered 95% interval excluding zero |
| selector recall | with coarse expansion `alpha <= 4`, mean native-block recall is at least 99%, every major workload family is at least 97%, and complete-hit events are at least 95% |
| anchor lift | anchor+HCA reduces late-miss blocks by at least 50% or improves complete-hit by at least 5 percentage points over the prior-CSA anchor alone |
| sentinel | the HCA risk gate reaches at least 0.80 AUROC on anchor-failure events and reduces p99 misses by at least 50% for no more than 25% additional bytes |
| adaptive order | selective second-order reads dominate fixed second-order on the descriptor-bytes versus native-CSA-recall Pareto; provisional target: at least 20% fewer descriptor bytes at at least 99% native-CSA top-k recall |
| timing | at B1 and B4/B8, at least 99% of predicted transfers finish inside the measured causal window, late-miss events are at most 1%, and p95 signal-to-use slack exceeds p95 transfer plus dispatch time |
| movement | false bytes do not exceed useful bytes, total H2D is at most twice oracle delta movement, and staging-inclusive HBM is at most 1.25x the logical hot budget |
| system lift | projected p95 stall falls by at least 20% versus reactive delta fetch; at matched HBM, goodput improves at least 10% or tail stall at least 20% over the ECHO-style baseline |
| quality | any approximate arm is compared with native-CSA fallback; lossless wording is allowed only after bitwise/native-selection equivalence is demonstrated |

Failure of the conditional-signal gate kills HCA as a directory. Failure of the
adaptive-order gate kills the moment/residual cascade. Failure of timing kills
physical prefetch even if offline recall is good. These failures do not block
publication of a careful negative characterization study.

## 8. The only plausible runtime path after H0

If H0 passes, the near-term runtime candidate is **Causal Anchor-and-Gate**:

1. reuse exact indices from the latest causally prior CSA layer as the early
   anchor across the intervening HCA layer;
2. use HCA/query-shift/disagreement only to accept, widen, or reset that anchor;
3. issue any cold movement early enough to overlap with remaining HCA and
   FFN/MoE work; and
4. finalize with the native CSA indexer or an ECHO-style guaranteed-recall path.

The full multi-horizon ordering is therefore:

```text
prior-CSA exact anchor
    -> HCA query-shift / risk gate
    -> ECHO-style current-indexer rescue
    -> guaranteed delta recall
```

This mechanism does not claim that index reuse, prefetch, or exact recall is
new. Its possible contribution is a V4-specific empirical result: native HCA
cross-path evidence can identify when adjacent-CSA reuse is unsafe, reducing
unnecessary index/descriptor reads or late recovery at matched native
semantics.

The longer-term candidate is **Adaptive-Order Sentinel**:

```text
prior CSA anchor + native HCA
              |
       sufficient / stable? ------ yes ------> candidate + native check
              |
              no
              v
  read second-order sketch only for ambiguous spans
              |
       sufficient now? ----------- yes ------> candidate + native check
              |
              no
              v
       unmodified native CSA scan / exact recovery
```

The second-order sketch cannot be fetched indiscriminately from HBM or host
memory. If its bytes, dispatch, or reconstruction cost reaches the native scan,
the architecture has no systems case even if its prediction is accurate.
As a provisional stop rule, selector traffic above 1.3x the first-order path or
indexer latency above 1.1x ends this branch unless it buys a separately
preregistered quality gain.

A second systems candidate is an **HCA-triggered residency epoch**: accumulate
promotions over a longer horizon instead of issuing tiny per-layer copies.
This may provide more overlap and amortize launch/remap cost, but it remains
behind the same H0 signal gate and must count false residency over the whole
epoch.

## 9. Evaluation contract if the signal survives

The existing proof plan remains necessary but gains new axes:

- fixed, fixed+pins, calibrated-no-pins, calibrated+pins, shuffled quota,
  local/hierarchical, prior-CSA anchor, HCA-only, fixed second-order, and
  adaptive-order arms;
- at least five seeds and both trained Tier-S scales for the causal core;
- RULER 8K--128K, SCBench, LongBench v2, LongMemEval, and MRCR;
- WILDTRACE or an equivalent naturally distributed evidence benchmark,
  structured JSON/code/tool generation, dense aggregation, and adversarial
  block geometry;
- query-before-compression versus query-after-compression and same-prefix
  multi-query reuse;
- context, batch, concurrency, generation length, and load matrices; and
- total HBM including index/rollback/controller state, H2D/D2H, useful and late
  transfers, latency, throughput, p95/p99, OOM, and quality failures.

Strong system comparators must include ECHO, exact fused LiteTopK, IndexCache or
CLSA-style reuse, FlashMemory-V4, MosaicKV, and the resident native CSA path.
Strong semantic comparators must include start+recent and a robust
query-agnostic method as required by the query-visibility audit.

The original primary empirical gate remains unchanged:

> At identical actual hot memory, `calibrated+pins > fixed+pins` must reproduce
> across five seeds and both scales.

That gate tests the existing adaptive quota/pin story. It is independent of the
new HCA sufficiency study; neither result may be used as a substitute for the
other.

## 10. Repository reality check

The current implementation supports instrumentation but not a credible
physical-prefetch claim yet:

- `_core_attention` computes HCA scores/probabilities but currently returns
  only the context, so an opt-in probe is required;
- an HCA span covers roughly 128 tokens while a CSA entry represents roughly
  four, making the directory about 32-way coarse before boundary effects;
- the causal overlap window is only the remainder of the HCA block plus
  intervening FFN/MoE work before the target CSA layer;
- the M4 tier store copies the whole hot set, flushes hot slots on append,
  unions batch selections, and host-resolves CUDA events, so its latency cannot
  validate a new prefetch mechanism; and
- the checked M4 measurements moved only kilobytes per token yet added roughly
  millisecond-scale latency and reduced total measured cache allocation by only
  about 1.9%, indicating launch/gather/synchronization and non-value state are
  first-order costs.

Official V4 layer schedules do contain HCA-to-later-CSA causal pairs, so the
trace question is implementable. In the checked configurations, Flash has 21
CSA and 20 HCA layers, with every CSA after the first preceded by HCA; Pro has
30 CSA layers with a preceding HCA opportunity. A prior-CSA anchor is available
earlier than HCA relevance, however: it can overlap the prior CSA attention and
FFN, the intervening HCA attention and FFN, and the next index scan. HCA-only
movement has only the tail of HCA, its FFN, and the next index scan. That does
not establish enough lead time for offload. The timing gate must use a
prototype that avoids the known M4 whole-set copy and host-synchronization
path.

Before physical H1, the tier store needs persistent paged hot slots, delta
promotion/eviction, append-in-place, nonblocking event dependencies,
per-request ragged page tables, GPU-side gather/remap, and separate useful,
false, late, and redundant-byte counters. The current store remains a
correctness harness.

The final system gate is stricter than the trace gate: total HBM must fall by at
least 20% or admission/concurrency rise by at least 25%, p95 latency must not
regress, and throughput/goodput must reach at least 1.10x without a tail
collapse on query-shift or dense slices. A 91% reduction in one CSA value
component with roughly 2% total-cache reduction remains a no-go.

## 11. Priority and stop decisions

| Priority | Action | Decision |
| --- | --- | --- |
| P0 | let the frozen P2 core finish untouched | continue |
| P1 | implement HCA--CSA conditional-residual trace instrumentation | next |
| P2 | run causal matched-cost HCA/prior-CSA/COBS/native comparisons | next after instrumentation |
| P3 | test adaptive-order sentinel and grouped causal-risk gate offline | only if P1/P2 pass |
| P4 | build Anchor-and-Gate residency/prefetch prototype | only if signal and timing pass |
| P5 | run natural, transfer, and production systems matrices | only after the two independent core gates pass |
| defer | learned future utility, hardware action market, new memory-native training | not on the critical path |
| reject as headline | adaptive quota alone, semantic pins alone, generic hierarchy, generic uncertainty widening, generic hardware-aware controller | components/baselines only |

## 12. Bottom line

There is still a publishable possibility, but it is not "we invented adaptive
KV memory" or "HCA can prefetch CSA." The defensible core is a measurement and
mechanism claim about **native V4 cross-path sufficiency and its failure
boundary**:

> characterize when HCA plus a causal CSA anchor is sufficient, acquire
> second-order evidence only where it is not, and retain the native CSA path as
> the semantic reference.

The fastest way to make the project more valuable is therefore to resist a
large runtime build, expose the missing HCA trace, and try to kill this claim
against ECHO-style history, IndexCache-style adjacent-layer reuse, COBS-style
second-order summaries, and the exact fused index path. If it survives, the
project has a narrow but credible research identity. If it fails, the resulting
cross-path sufficiency map is still more informative than another underpowered
adaptive-quota experiment.
