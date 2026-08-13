# DSpark semantic vectors

`nano-deepseek-v4 dspark` checks a small, deterministic implementation of the
DSpark draft equations against packaged golden vectors and an independent dense
oracle. It runs on CPU, makes no network requests, and does not download the
DeepSeek-V4 checkpoint.

Run the standalone check after installing the current source:

```bash
nano-deepseek-v4 dspark
```

For automation or an issue report, retain the receipt:

```bash
nano-deepseek-v4 dspark \
  --json \
  --output dspark.json
```

The same check is available inside the conformance harness:

```bash
nano-deepseek-v4 conformance \
  --profile dspark \
  --json \
  --output conformance-dspark.json
```

The `dspark` profile is CPU-only and offline. The standalone command is the
shorter path when DSpark semantics are the only question; the profile is useful
when one receipt must carry the surrounding package and network-policy evidence.

## What is checked

The fixture uses the official three-stage layout and block size of five, scaled
down to tiny full-precision tensors. Fixed inputs, weights, intermediate values,
and outputs make changes reviewable without a large checkpoint or accelerator.

The native path and the independent oracle must agree on:

1. target hidden states concatenated in configured layer order, followed by the
   first-stage projection and RMS normalization,
   `H_ctx = RMSNorm(W_c [H^(l1); ...; H^(lm)])`;
2. context keys and values injected alongside draft-block keys and values, with
   bidirectional attention inside the draft block;
3. the anchor token followed by four noise positions producing five parallel
   hidden states and base-logit vectors `U_k`;
4. left-to-right greedy proposals from
   `U_k + W_1[x_(k-1)] W_2`, where the Markov lookup uses the previously
   proposed token;
5. confidence values
   `sigmoid(w^T [h_k; W_1[x_(k-1)]])` from the same previous-token embedding;
6. shared target embeddings and language-model head remaining unchanged while
   the draft path runs.

Token IDs are compared exactly. Floating-point vectors use the tolerance stated
in the receipt. Mutation checks cover the easy-to-miss failures: introducing a
causal draft mask, removing the Markov bias, looking up the current token,
reversing target-layer order, or changing the confidence input or sigmoid.

## Provenance

The vector contract is derived from these immutable references:

- [`DeepSeek-V4-Flash-0731` revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062),
  especially `config.json` (SHA-256
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`)
  and `inference/model.py` (SHA-256
  `c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f`);
- [*DSpark: Confidence-Scheduled Speculative Decoding with
  Semi-Autoregressive Generation*, arXiv:2607.05147v1](https://arxiv.org/abs/2607.05147v1),
  equations 2, 3, 5, and 7.

The receipt binds those source identities to the packaged fixture, oracle, and
native implementation digests. A new upstream revision or regenerated vector
set is therefore a new evidence object, not an invisible update to an old pass.
The fixture digest covers the canonical semantic config as well as every input
and weight tensor, so changing an epsilon, block dimension, or noise-token ID
also changes the fixture identity.

## Reading the receipt

A pass means that the installed package reproduced the packaged token and
floating-point vectors within the recorded tolerance. Keep the status, vector
and implementation digests, source revision, dependency versions, platform,
and `claim_boundary` together when archiving the result. A matching digest
without the boundary does not widen the claim.

The check establishes tiny eager CPU conformance for the draft equations above.
It does **not** establish:

- official FP4, FP8, or BF16 weight or kernel parity;
- official-checkpoint loading or payload integrity;
- rejection sampling, target-token acceptance, confidence calibration, or the
  hardware-aware prefix scheduler;
- cache layout, distributed or long-context behavior, latency, throughput, or
  serving capacity;
- training equivalence, model quality, or lossless end-to-end speculative
  decoding.

Use the pinned metadata replay for checkpoint-schema evidence and a serving
backend's hardware tests for runtime claims beyond this boundary.
The separate [DSpark scheduler checker](dspark-scheduler.md) covers only the
paper's CPU scheduler arithmetic and causal trace on supplied calibrated
probabilities and synthetic SPS tables; it does not widen this equation
receipt's claim.

## Reusing the vectors

The packaged fixture is intended as an adapter target for other runtimes. A
downstream implementation can load the same config, inputs, weights, and golden
outputs, map its internal names to the documented tensor roles, and compare the
same checkpoints: context projection, non-causal draft attention, base and
Markov-adjusted logits, proposals, and confidence scores.

Report the vector-set digest and any dtype or tolerance change with downstream
results. Passing a converted or relaxed fixture is useful evidence, but it is
not the same conformance result as the unmodified packaged vectors.
