# Transformers differential parity

Architecture self-tests can agree with themselves while implementing the wrong
equation. This verifier constructs the same deterministic tiny configuration in
this package and pinned Transformers 5.15.0, maps the reference tensors into the
readable native layout, and compares independent execution.

## Run

Select the CPU or accelerator PyTorch build with the
[installation guide](../guides/installation.md), then install the pinned parity
extra:

```bash
python -m pip install -e ".[parity]"
nano-deepseek-v4-parity
```

Expected result:

```text
reference: transformers 5.15.0
weights: 104 native tensors from 89 packed reference tensors
main RoPE inverse frequencies: max=0, mean=0, allclose=True
compressed YaRN inverse frequencies: max=0, mean=0, allclose=True
main RoPE: max=5.96e-08, mean=3.1e-10, allclose=True
compressed YaRN RoPE: max=5.96e-08, mean=3.1e-10, allclose=True
full forward: max=0, mean=0, allclose=True
cached decode: max=2.98e-08, mean=3.49e-09, allclose=True
reference cache/full: max=4.47e-08, mean=4.25e-09, allclose=True
native cache/full: max=4.47e-08, mean=5.1e-09, allclose=True
backward gradients: max=4.66e-10, mean=2.75e-12, allclose=True
MTP residual streams: max=1.49e-08, mean=1.16e-10, allclose=True
MTP hidden states: max=1.49e-08, mean=7.76e-11, allclose=True
parity: PASS
```

## Coverage

The model fixture includes:

- sliding attention, CSA, and HCA;
- hash and learned MoE routing;
- mHC residual streams;
- official compressed-branch YaRN;
- irregular cached chunks that cross compression boundaries;
- gradients for all participating parameters.

A separate rotary oracle uses the official 64-dimensional slice and checks main
and compressed inverse frequencies plus positions 0, 1, 127, 65,535, 65,536,
and 1,048,575. It catches both a dropped `rope_scaling` configuration and an
accidental generic YaRN magnitude scale; V4 keeps rotary vectors
unit-normalized.

Transformers does not expose the official MTP wrapper. The verifier therefore
composes MTP fusion and stream-collapse equations independently from a
revision-pinned DeepSeek inference source while executing the intervening
decoder block in Transformers.

The checked-in
[`transformers-v5.15.0-parity.json`](../../references/transformers-v5.15.0-parity.json)
records source revisions and SHA-256 digests, input and configuration identity,
weight mapping, tolerances, and every observed error. CI reruns the comparison
rather than accepting the receipt as sufficient evidence.

## Why the dependency is pinned

Transformers is part of the oracle. An upgrade can change packed layouts,
configuration normalization, cache behavior, or equations. Updating the pin
therefore requires reviewing and replacing the receipt rather than silently
accepting a new dependency version.

## Claim boundary

The result verifies eager floating-point tensor equations on a tiny model and
rotary vectors at official dimensions and boundary positions. It does not
execute a million-token sequence or establish parity for official FP4/FP8
kernels, optimized serving, distributed execution, long-context performance, or
pretrained checkpoint quality.
