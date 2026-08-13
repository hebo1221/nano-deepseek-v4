# Native model bundles

Native bundles store a `DeepSeekV4ForCausalLM` configuration, sharded
safetensors weights, an index, and a canonical SHA-256 manifest. They are
distinct from official DeepSeek snapshots, whose key layout follows the
upstream checkpoints.

## Model-only v1 bundle

```python
import torch
from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM

model = DeepSeekV4ForCausalLM(DeepSeekV4Config())
model.save_pretrained("./tiny-model", max_shard_size_bytes=256 * 1024**2)

# Construction starts on the meta device and materializes one shard at a time.
restored = DeepSeekV4ForCausalLM.from_pretrained(
    "./tiny-model",
    device="cpu",
    dtype=torch.float32,
)
```

Calling `save_pretrained` without a tokenizer writes the backward-compatible v1
format. It is suitable for model loading and inspection, but it deliberately
does not claim that token IDs have a known text mapping.

## Generation-ready v2 bundle

When a model was trained with this project's canonical byte-v1 mapping, bind
that mapping to the weights during the same atomic save:

```python
from nano_deepseek_v4 import ByteTokenizer

tokenizer = ByteTokenizer()
tokenizer.validate_config(model.config)
model.save_pretrained("./byte-model", tokenizer=tokenizer)
```

The v2 manifest adds `nano_deepseek_v4_tokenizer.json` to the same checksum
inventory as `config.json`, `model.safetensors.index.json`, and every shard. The
sidecar fixes the complete byte mapping:

- PAD, BOS, and EOS are 0, 1, and 2;
- byte `b` maps to token `b + 3`;
- vocabulary size is 259;
- text uses UTF-8 without normalization;
- invalid UTF-8 is replaced only in the text view, while `decode_bytes` remains
  exact.

The generation CLI rejects v1 bundles. `vocab_size == 259` is not proof that a
model was trained with byte-v1, and guessing would make generated text
semantically unauditable. Both v1 and v2 remain valid model bundles.

## Inspect before loading

```bash
nano-deepseek-v4-inspect --bundle ./tiny-model
nano-deepseek-v4-inspect --bundle ./byte-model --json
```

The verifier checks:

- manifest format and every declared model-file entry;
- every declared SHA-256 when checksum verification is enabled;
- canonical configuration and tokenizer/config compatibility;
- index metadata and total payload size;
- every safetensors header;
- the exact index-to-shard tensor mapping;
- missing tensors, unexpected tensors, and unindexed safetensors shards.

Additional non-model files such as a model card or license may live beside the
bundle. They are outside the native manifest and are neither loaded nor covered
by its integrity claim.

The inspection path does not construct a model or materialize tensor payloads.
For v2, its report exposes the tokenizer type and digest and marks the bundle as
generation-ready only after all tokenizer checks pass.

## Publication and loading behavior

`save_pretrained` assembles the complete bundle in a sibling temporary directory
before exposing it. A non-empty destination is rejected rather than mixed with a
new checkpoint generation. Shared-storage tensors, including tied embeddings,
are cloned before safetensors serialization.

`from_pretrained` verifies the bundle before allocating runnable weights, creates
the model on the meta device, and materializes at most one shard at a time. The
shard-size limit is soft when one tensor is individually larger than the
requested size. Stop optimizer updates before saving so every shard represents
one logical training step.

Checksum verification can be disabled with `verify_checksums=False` only for a
trusted local bundle. Format, inventory, and tokenizer/config compatibility
remain structural requirements; disabling hashing does not turn an unknown
tokenizer into byte-v1.

## Trust boundary

The manifest detects storage corruption, incomplete copies, and accidental
mixing of bundle generations. It does not authenticate the publisher: an
attacker able to replace both a file and the manifest can recompute the digest.
Pin an external revision or manifest digest obtained through a trusted channel.

For revision-pinned generation from a user-owned Hugging Face repository, see
[Train and generate](train-and-generate.md). Official DeepSeek checkpoints must
follow the separate [official checkpoint guide](official-checkpoints.md).
