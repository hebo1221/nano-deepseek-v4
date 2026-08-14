# Research and reproducibility

The repository separates quick correctness checks from experiments that consume
substantial compute. A passing smoke test is useful evidence, but it is not a
training result or an architecture comparison.

## Evidence ladder

| Question | Command or receipt | Cost and scope |
| --- | --- | --- |
| Does the package import and execute? | `nano-deepseek-v4-demo` | Seconds on CPU; architecture and cache smoke test |
| Does the installed fixed learning path pass? | `nano-deepseek-v4 reproduce` | About a minute on CPU; fixed 20-step learning contract and verified native bundle |
| Can a custom tiny experiment train and save? | `nano-deepseek-v4 train --steps 20 --save-directory ./checkpoints/tiny-text` | Configurable corpus, recipe, device, and sampling; not the fixed reproduction contract |
| Can information cross the attention schedule? | [`../verification/attention-reach.md`](../verification/attention-reach.md) | Focused Jacobian/VJP conformance test |
| Do the equations match an independent implementation? | [`../verification/transformers-parity.md`](../verification/transformers-parity.md) | Deterministic float64 parity test |
| Do two trained bundles differ on the same held-out windows? | `nano-deepseek-v4-compare` | Paired evaluation; does not measure training-seed variation |
| Can compatible paired reports be summarized? | `nano-deepseek-v4-aggregate` | Rejects protocol drift before aggregation; run labels remain caller-supplied |

The checked-in JSON files under [`../../references`](../../references) are
machine-readable receipts. They bind measurements to corpus, configuration,
implementation, bundle, and evaluator digests where applicable.

## Tiny Shakespeare study

[`tiny-shakespeare.md`](tiny-shakespeare.md) contains the complete reproduction
commands and the observed NVIDIA GB10 results for:

- a 5,000-step hybrid-attention training baseline;
- a matched all-sliding attention control;
- paired evaluation on 256 shared held-out byte windows.

The result is deliberately modest: its descriptive loss interval crosses zero,
and one matched run cannot characterize seed-level variation.

## Interpreting receipts

Receipts are evidence about the exact checked-in experiment, not a substitute
for broader evaluation. In particular:

- overlapping byte windows do not form independent samples;
- one training seed cannot support an architecture-superiority claim;
- GB10 eager-PyTorch throughput does not establish optimized-kernel or
  long-context performance; and
- the official DeepSeek-V4 benchmark results have not been reproduced here.

For the package boundary, see [`../architecture.md`](../architecture.md). For
native checkpoint integrity, see
[`../guides/native-bundles.md`](../guides/native-bundles.md).
