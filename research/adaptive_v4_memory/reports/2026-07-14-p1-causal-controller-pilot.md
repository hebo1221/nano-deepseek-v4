# P1 causal same-token controller pilot

Status: **directional pilot, not claim-eligible**. This experiment checks the
causal interface and rejects obviously weak controller settings before the
preregistered five-seed study.

## Implementation and controls

The same-token controller ranks the current layer's causally visible CSA blocks
before value attention. A frozen quota is assigned to every CSA layer, and a
layer may use only its own current score signal plus decisions from earlier
layers at the same token. It never uses a later layer or future token. Cache
clone, crop, batch select/stack, checkpoint restore, quota enforcement, replay
digest integrity, protected positions, prefill control, and tiered fetch are
covered by tests.

The initial pilot runner had an invalid comparison: fixed top-k governed
prefill, while same-token control began only after prefill. The clean run fixes
this by attaching the controller to an empty cache before prefill. A separate
test establishes that the controller with adaptive components disabled is
bitwise equivalent to fixed top-k from prefill onward. The tier-store ceiling is
also fixed to the controller's layer quota rather than logical context size.

## Clean directional result

Both scales use evaluation seed 8071401 and 32 examples per family and arm. All
raw artifacts record clean source commit `d7341d6` and checkpoint digests.

| Scale / family | Fixed | One-token M2 | Same local | Same hierarchical |
|---|---:|---:|---:|---:|
| S55 single retrieval | 81.25% | 28.13% | 81.25% | 81.25% |
| S55 query shift | 50.78% | 14.06% | 50.00% | 50.00% |
| S55 dense memory | 54.30% | 14.45% | 54.30% | 54.30% |
| S151 single retrieval | 78.13% | 9.38% | 78.13% | 78.13% |
| S151 query shift | 33.59% | 12.50% | 33.59% | 33.59% |
| S151 dense memory | 28.91% | 9.77% | 28.91% | 28.91% |

Same-token control recovers the catastrophic loss caused by the one-token
interface, but it does not beat the memory-matched fixed policy. With fallback
disabled, prediction digests match fixed top-k on every S151 family and every
S55 family. The current minimum quotas cap requested blocks at exactly fixed
top-k, so score, refresh, and cross-layer terms have little room to alter the
quality-relevant action.

Dense fallback is actively harmful to the memory Pareto. Same-local residency
rose from the fixed 6 blocks to 29.06/40.88 blocks on S55 query/dense workloads
and from 5 blocks to 23.41/30.53 on S151, without an accuracy gain. Removing
fallback restored the fixed resident budget and prediction digest. The current
fallback thresholds therefore fail the pilot gate and are not carried forward
unchanged.

The reference Python controller adds latency: same-token p95 is approximately
19.2--21.0 ms on S55 and 30.5--32.9 ms on S151 versus fixed ranges of
16.9--17.3 ms and 26.2--27.0 ms. This is a reference-path measurement, not a
fused-kernel result.

## Decision

The causal interface is retained; the current uniform-quota policy is not a
positive result. P1 continues with calibration-only non-uniform layer quotas,
fallback threshold calibration, a protected-prefix ablation, and a separately
trained lookahead arm. No paper-level effectiveness claim is made from this
single-seed pilot. In parallel, the frozen five-seed training matrix begins so
that subsequent controller conclusions are not tied to the original two
checkpoints.
