# M5 final validation: online M2 negative result

Status: **final negative result for Adaptive V4 Memory's online M2 hypothesis**.
The M4 fixed-top-k tiered store remains a valid bounded memory result, but the
adaptive controller is not quality-safe in causal serving.

## What was implemented

`OnlineTrainingFreeController` connects the replay-validated M2 policy to cached
decode. Each CSA layer records native index scores; the globally budgeted action
is deterministically applied to the next token and therefore drives the actual
hot/cold block fetch. Clone, crop, batch select/stack, dense fallback, and cache
persistence are covered by deterministic tests. Native trace collection remains
unchanged, so adaptive action does not contaminate the evidence stream.

The one-token delay is not merely an implementation choice. A global action
needs scores from all CSA layers, while each layer needs its selected values to
produce hidden state for the next layer. Without a fused hierarchical policy,
the complete current-token signal is only available after the forward pass.

## Official checkpoint boundary

The pinned official metadata audit found 46 Flash safetensor files totaling
159,617,149,040 bytes and 64 Pro files totaling 864,721,029,744 bytes. This host
had 142,335,508,480 bytes of available RAM plus swap and 290,384,891,904 free
disk bytes. Flash exceeds executable memory, Pro exceeds both memory and disk,
and no supported `transformers`, vLLM, SGLang, or FlashInfer runtime is installed.
No 160 GB download was performed and no official-weight execution is claimed.
The audit script resolves exact official repository revisions and records the
host boundary in an ignored, digest-bound artifact.

## Tier-S falsification experiment

The final experiment used both trained scales, 32 generated examples per
family/mode, identical seeds, and four arms: native, memory-matched fixed-top-k,
online M2, and dense. Families were single remote retrieval, four-query shift,
and an eight-query dense-memory stress. Raw generated tokens were not persisted.

| Scale / family | Native | Fixed | Online M2 | Dense |
|---|---:|---:|---:|---:|
| S55 single retrieval | 81.25% | 81.25% | **28.13%** | 81.25% |
| S55 query shift | 61.72% | 50.78% | **14.06%** | 60.94% |
| S55 dense memory | 62.50% | 54.30% | **14.45%** | 62.89% |
| S151 single retrieval | 84.38% | 78.13% | **9.38%** | 84.38% |
| S151 query shift | 38.28% | 33.59% | **12.50%** | 37.50% |
| S151 dense memory | 35.94% | 28.91% | **9.77%** | 37.50% |

The controller produced zero action-budget violations and zero tier late misses.
Actual hot blocks converged to six on S55 and five on S151. Thus the failure is
not a store consistency bug: the previous token's ranking is the wrong signal
when a new query arrives. Natural fallback was 0% on single retrieval and at
most 15.63% under the longer stresses, far too late or too rare to recover.

Online M2 reduced total hot-resident cache by only 1.51–1.92%, because CSA value
blocks are a small part of the reference cache while rollback history, index
state, local KV, and HCA remain resident. Its p95 was 1.11–1.29x native and
controller CPU time grew from 242 microseconds to 2.33 milliseconds per control
point. It therefore fails both the quality gate and the strict speedup gate.

## Final interpretation

M1 established predictive budget signal and M2 found a useful offline policy.
M3 showed that the learned-risk replacement did not improve it. M4 established
correct physical value-block tiering with roughly 91% compressor-value reduction
but only about 2% total-cache reduction and no speedup. M5 now falsifies the
assumption that the offline M2 decision can be shifted one token and remain
quality-safe.

The supported outcome is narrow: fixed-top-k CPU-cold/GPU-hot CSA value storage
is correct and memory-bounded in this reference runtime. Adaptive online claims,
official-scale performance claims, and production speedup claims are rejected.

The minimum credible next systems design is a fused same-token contract that
(1) computes layer-local uncertainty before value attention, (2) allocates a
hierarchical request/layer budget without waiting for later layers, (3) issues
asynchronous prefetch into paged slots, and (4) exposes rollback/history as a
separate residency class. Until that exists, further controller tuning would
optimize an invalid causal interface.
