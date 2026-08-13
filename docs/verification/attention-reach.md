# Attention-path reachability

This deterministic conformance check asks whether a source-token change can
influence the final-position logits through the actual native attention graph.
It requires no training or downloaded weights.

```bash
nano-deepseek-v4-attention-reach
```

## Boundary under test

With a causal window of 32, one local layer can add 31 prior positions to the
receptive field. Three local layers therefore have a maximum source lag of:

```text
3 * (32 - 1) = 93
```

The protocol's distance is one-based. Distance 94 has lag 93 and remains
reachable; distance 95 has lag 94 and is structurally outside the local-only
graph.

The runner evaluates six pinned distances: 16, 64, 94, 95, 128, and 224. For
each probe it changes exactly one source token and compares final-position
logits for three models initialized from fixed seeds:

1. the native sliding → CSA → HCA hybrid stack;
2. a near-parameter-matched three-layer local control;
3. the same control with a full attention window.

The local and full-window controls have identical initialized state and differ
only in window size. They each contain 930,345 parameters. The hybrid contains
930,405, a difference of 60 parameters, or `0.00645%` of the hybrid.

## Pass contract

The discrete perturbation check uses a pinned `1e-6` L∞ influence tolerance.
Every probe must exceed that tolerance at every distance for the hybrid and
full-window variants. The local control must exceed it through distance 94 and
remain at or below it for distances 95, 128, and 224.

Exact-change counts remain diagnostic only. MoE expert batching can introduce
sub-tolerance floating-point drift even when the mathematical local graph has no
dependency.

As an orthogonal witness, the runner detaches each positional embedding output
and computes a fixed-seed Rademacher vector-Jacobian product from the final
logits. The source gradient must be nonzero on the same reachable paths and
exactly zero outside the local boundary. This derivative holds hash routing and
CSA top-k choices fixed, so it complements rather than replaces the discrete
token perturbation.

The command returns nonzero if the structural contract fails and can emit
canonical JSON with `--json`. The checked-in
[`attention-reach-conformance.json`](../../references/attention-reach-conformance.json)
binds the protocol, configurations, source digests, probe inputs, cotangent,
perturbations, gradients, and observed metrics. CI regenerates the check instead
of trusting the receipt alone.

## Claim boundary

This establishes structural numerical influence for fixed, randomly initialized
nano models through the combined hybrid path. It does not isolate CSA from HCA,
prove that every possible signal propagates, or measure learned retrieval,
language quality, optimized kernels, efficiency, pretrained checkpoints, or
general long-context performance.

A nonzero local derivative is evidence about the pinned execution graph, not a
benchmark score or an effect-size estimate.
