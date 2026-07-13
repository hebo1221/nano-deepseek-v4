# M2 training-free controller report

Status: the preregistered offline M2 gate passed at both Tier-S scales. This is
not an online serving or physical-memory result.

## Method

The controller combines normalized indexer entropy, top-p cardinality, boundary
margin, temporal overlap, and cross-layer overlap. It assigns requests under one
global block budget, derives a stability-dependent refresh interval, prioritizes
protected blocks, records load/evict bytes, and switches to a larger dense
fallback budget when uncertainty is high. Every action is digest-bound and is
recomputed from the same score trace during replay validation.

Eight rule configurations were compared on 384 calibration examples per scale.
Only the selected configuration was evaluated on 768 disjoint test examples.
The model forward uses native masks for unplanned context positions and applies
the frozen controller plan at the final answer control point.

## Results

- S55 selected six blocks per example. At the same logical block count,
  controller accuracy was 67.4% versus fixed top-k-2 at 64.3%. Controller CPU
  time was 98.4 µs/example and planned forward time was 37.1 ms/batch versus
  34.6 ms for fixed top-k-2.
- S151 selected five blocks. Accuracy was 74.2% versus fixed top-k-1 at 67.8%.
  Controller CPU time was 189.1 µs/example and planned forward time was 64.2
  ms/batch versus 62.5 ms for fixed top-k-1.
- All quality-regression, worst-length, movement, overhead, fallback, and Pareto
  checks passed. Cold-start movement/selection bytes were exactly 1.0.

Natural held-out fallback rate was zero at both scales. A separate forced
uncertainty stress arm activated fallback on every example and recovered 5.9
percentage points at S55 and 1.4 points at S151 relative to fallback-disabled
allocation, reaching native-level quality at the cost of 48 and 80 mean blocks.

## Failed implementation path

The first S55 implementation applied a Python plan loop at every token and
timed a second complete replay-validation pass. Quality already dominated fixed
top-k-1 at the same three blocks (64.7% versus 52.6%), but controller overhead
was about 9.85 ms/example and planned forward time about 266 ms/batch, so the
overhead gate failed. The final implementation tensors the plan mask, times
replay validation separately, and invokes the controller only at the answer
control point. The original failure is not used in the positive gate result.

## Limits

The block and byte reductions are logical selection accounting. All compressed
blocks remain allocated on GPU, so M2 makes no GPU-memory reduction claim.
There is one task, one model-training seed per scale, no natural dense-fallback
event, and no multi-turn query-shift test yet. M3 may proceed, but M4 must
implement actual hot/cold residency before any systems claim.
