# DSpark prefix scheduler

`nano-deepseek-v4 dspark-scheduler` checks the CPU arithmetic and decision
trace for the hardware-aware prefix scheduler described in the
[DSpark paper](https://arxiv.org/abs/2607.05147v1). It uses calibrated
conditional probabilities and a supplied steps-per-second (SPS) table. It does
not run a draft or target model, profile hardware, or download a checkpoint.

Run the wheel-packaged vectors:

```bash
nano-deepseek-v4 dspark-scheduler
```

Retain the deterministic receipt for automation or an issue report:

```bash
nano-deepseek-v4 dspark-scheduler \
  --json \
  --output dspark-scheduler.json
```

The same check is a separate required entry in the offline `dspark` profile:

```bash
nano-deepseek-v4 conformance \
  --profile dspark \
  --json \
  --output conformance-dspark.json
```

## What the scheduler computes

The input rows contain calibrated conditional probabilities `c[r, k]`, not the
raw confidence-head logits emitted by the official model. For request `r`, the
survival probability of a prefix through position `j` is

```text
a[r, j] = product(c[r, i] for i = 1..j)
```

For selected draft lengths `ell[r]`, the target batch size and expected token
count are

```text
B   = R + sum(ell[r])
tau = R + sum(a[r, j] for every selected prefix position j)
```

The scheduler evaluates the paper's throughput proxy `Theta = tau * SPS(B)`.
The SPS mapping is an input to this checker. A passing result therefore shows
that the arithmetic used the supplied table correctly; it says nothing about
whether that table measures a real deployment.

### Algorithm 1: causal greedy mode

`causal_greedy` starts from the no-draft baseline `R * SPS(R)`. Candidates are
ordered by survival probability, highest first. The local deterministic tie
break is one-based draft position ascending, then request index ascending.
Prefix closure follows from the non-increasing cumulative products.

Each candidate is added tentatively. It is retained only when its throughput is
strictly greater than the previous best. The first non-improvement, including
an exact tie, stops the scan. The returned allocation is the last improving
prefix and never includes the candidate that caused the stop. This follows
[Algorithm 1](https://arxiv.org/html/2607.05147v1#alg1) and its throughput
definition in [Equation 7](https://arxiv.org/html/2607.05147v1#S3.E7).
The paper's global-optimum argument assumes the objective is unimodal along
this candidate order. This checker deliberately runs Algorithm 1 on any valid
SPS table, but it does not infer or certify unimodality; on a jagged table the
causal result is an algorithm trace, not a global-optimum claim.

### Section 5.2: two-step-lagged mode

`lagged_topk` models the production adaptation in
[Section 5.2](https://arxiv.org/html/2607.05147v1#S5.SS2). It uses confidence
from two decoding steps earlier to choose one global draft-token capacity `K`.
That search checks the full positive-survival candidate curve, rather than
stopping at the first dip, so a later recovery in a jagged synthetic SPS curve
can still win. The current step's cumulative survival values then rank and
allocate exactly `K` candidates.

The API treats `historical_confidence_probabilities` as caller-supplied
capacity history. It cannot prove that the rows were produced exactly two
steps earlier, remained aligned to the same request slots, or handled arrivals
and departures correctly. The receipt therefore reports
`decision_source=caller_supplied_history`; temporal provenance and slot
continuity remain integration responsibilities.

The paper does not specify every adapter-level tie policy. The packaged receipt
therefore records the deterministic rule above as a local canonicalization,
rather than attributing it to the paper.

## Scheduling one local request

Pass `--input` to schedule a strict JSON request instead of running the fixed
vectors. A causal request has exactly these fields:

```json
{
  "mode": "causal_greedy",
  "confidence_probabilities": [[0.9, 0.8], [0.7, 0.6]],
  "steps_per_second": {"2": 10.0, "3": 9.0, "4": 8.0, "5": 7.0, "6": 6.0}
}
```

A lagged request supplies aligned current and two-step historical rows:

```json
{
  "mode": "lagged_topk",
  "current_confidence_probabilities": [[0.9, 0.8], [0.7, 0.6]],
  "historical_confidence_probabilities": [[0.8, 0.7], [0.6, 0.5]],
  "steps_per_second": {"2": 10.0, "3": 9.0, "4": 8.0, "5": 7.0, "6": 6.0}
}
```

Run it with:

```bash
nano-deepseek-v4 dspark-scheduler \
  --input request.json \
  --json \
  --output schedule.json
```

Probability matrices must be non-empty, rectangular, finite, and within
`[0, 1]`. Current and historical matrices must have identical shapes and aligned
request slots. For `R` requests and maximum draft length `L`, the SPS object
must contain every integer batch size from `R` through `R + R * L`, encoded as
canonical decimal JSON keys, with finite positive rates.

## Packaged conformance vectors

The wheel carries four fixed cases:

- a smooth Algorithm 1 trace;
- the high- and low-confidence examples based on
  [Appendix A](https://arxiv.org/html/2607.05147v1#A1);
- a two-step-lagged case with a jagged SPS curve.

The native implementation must agree with both an exhaustive, separately
written oracle and the packaged golden summaries. Additional checks cover the
strict-equality stop, deterministic ties, cumulative rather than marginal
confidence, the paper's retrospective counterexample, fixture integrity, and
implementation/source digests. The JSON report records the vector identity,
case summaries, hashes, environment, tie policy, and claim boundary.

Exit `0` means the fixed vectors passed, or that one valid input request was
scheduled. Exit `1` means a fixed conformance check failed, `2` is command-line
usage error, and `4` covers invalid input, fixture or harness errors,
serialization failure, and an unwritable output path.

The paper provenance is pinned to `arXiv:2607.05147v1`. The confidence-head
source is also pinned at the official
[`DeepSeek-V4-Flash-0731` revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L807-L815),
where the model emits raw logits. Converting or calibrating those logits into
the probability inputs used here remains outside this checker.

## Claim boundary

A pass establishes CPU arithmetic and causal-trace conformance for DSpark
Algorithm 1 and Section 5.2 on the supplied calibrated probability and SPS
tables. It does **not** establish:

- confidence calibration or STS quality;
- temporal provenance, the claimed two-step lag, or request-slot alignment;
- the validity of a measured hardware profile;
- draft or target execution, rejection sampling, or target-token acceptance;
- end-to-end losslessness or distribution preservation;
- optimized kernels, speed, throughput, or serving capacity;
- model quality or official-weight runtime parity.

Use the separate [DSpark semantic vectors](dspark.md) for the bounded draft
equations, the pinned metadata replay for checkpoint-schema evidence, and a
serving backend's end-to-end hardware tests for runtime claims.
