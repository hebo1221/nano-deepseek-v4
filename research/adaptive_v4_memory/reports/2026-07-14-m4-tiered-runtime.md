# M4 GPU-hot/CPU-cold tiered runtime

Status: reference runtime complete; quality and competitive-runtime gates pass,
but the preregistered strict speedup gate does not.

## Implementation

`nano_deepseek_v4/tiered_memory.py` adds a canonical pinned-CPU CSA value-block
store with a bounded GPU hot set, a dedicated CUDA transfer stream, explicit
prefetch, synchronous late-miss recovery, protected blocks, and transfer/useful
byte counters. The CSA indexer stays resident so its native selection is
unchanged; only selected compressor value blocks are gathered for attention.

Tier state participates in cache clone, crop, batch select, stack, and
safetensors persistence. Budget overflow is an error. Tests cover CPU state
operations, multi-turn resident/tiered decode equality, persistence, and actual
CUDA pinned-host/hot-device placement.

## Protocol

The checked [manifest](../manifests/m4-tiered-runtime.json) fixes 512-token
batch-1, batch-4, 1024-token query-shift, dense fallback, and two independent
round-robin request scenarios. S55 and S151 checkpoints are checked by SHA-256.
Each scenario uses 16 teacher-forced decode tokens after CUDA warm-up. The
quality arm uses 32 generated length-80 associative-recall examples per scale;
no tokens are persisted.

## Results

All runtime scenarios preserved greedy tokens and met the declared BF16
allclose tolerance. S55 was bitwise equal throughout. One S151 query-shift
scenario had maximum logit error 0.0625 because compact selection changes the
BF16 softmax reduction length; greedy output was unchanged. Quality was exactly
equal: 0.78125 at S55 and 0.875 at S151.

At batch 1, physical compressor value allocation fell by about 91.3%. Because
index vectors, local cache, and rollback history remain on GPU, total measured
cache allocation fell only 1.89–1.93%:

| Scale/context | GPU bytes reduced | Total cache reduction | p95 ratio | Throughput ratio |
|---|---:|---:|---:|---:|
| S55 / 512 | 49,152 | 1.888% | 1.062x | 0.952x |
| S55 / 1024 | 98,304 | 1.891% | 1.077x | 0.954x |
| S151 / 512 | 81,920 | 1.928% | 1.074x | 0.949x |
| S151 / 1024 | 163,840 | 1.931% | 1.074x | 0.943x |

Two independent request caches retained 94.4–94.9% throughput with p95 at
1.056–1.064x and doubled the absolute memory reduction. Batch 4 retained
81.2–84.0% throughput but p95 worsened to 1.54–1.66x because the reference
implementation unions per-example selections and performs Python-level gathers.
Dense fallback preserved correctness but moved roughly the full logical value
set repeatedly, as expected; it is a safety path, not a memory-saving point.

Allocated and reserved bytes plus allocator fragmentation, TTFT, p50/p95/p99,
H2D/D2H bytes and counts, useful bytes, evictions, and late misses are retained
in ignored raw results. The checked [summary](../results/m4-tiered-runtime.summary.json)
references them by digest.

## Decision and limits

M4 meets the project goal's quality-matched memory-reduction and competitive
runtime criterion, so the reference implementation is complete. It does not
improve latency or throughput and therefore fails the experimental protocol's
stricter positive-speedup gate. No production-serving or frontier-scale claim
is warranted.

This run isolates native-top-k tier mechanics; the offline M2 controller was
not executed inside the CUDA decode loop. Its previously measured CPU overhead
was 98–189 microseconds per example, but online action scheduling remains a
separate integration task. The next systems work should fuse index remapping,
attention, and H2D staging; tier rollback history as well as compressor values;
and evaluate an online M2 controller on official-compatible workloads.
