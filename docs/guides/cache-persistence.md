# Cache persistence

The native inference-cache format stores a `DeepSeekV4Cache` as safetensors plus
a JSON manifest. It is intended for controlled reuse of cache state produced by
the same configuration and exact model identity.

## Save and reload

```python
import torch
from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)

config = DeepSeekV4Config()
model = DeepSeekV4ForCausalLM(config).eval()
ids = torch.tensor([[1, 2, 3, 4]])
prefill = model(ids, use_cache=True)
assert prefill.past_key_values is not None

revision = "sha256:<trusted-checkpoint-digest>"
save_deepseek_v4_cache(
    prefill.past_key_values,
    "./cache/session-1",
    model_revision=revision,
)

cache = load_deepseek_v4_cache(
    config,
    "./cache/session-1",
    model_revision=revision,
)
```

The caller supplies `model_revision` because a model configuration alone does
not identify the trained weights. Use an immutable checkpoint revision or a
trusted digest rather than a moving branch name.

## Validation contract

Cache files are written atomically. Before returning cache state, the loader
checks:

- cache format version;
- canonical model-configuration digest;
- exact caller-supplied model revision;
- safetensors payload SHA-256;
- tensor inventory, names, and dtypes;
- layer count and expected cache components;
- local, buffer, history, compressed, and overlap shapes;
- stored token counts and position relationships.

Malformed metadata is rejected before it can silently alter model state. Legacy
cache formats without explicit model identity are not accepted.

## Trust boundary

The config digest, revision label, and payload checksum protect against
accidental mismatch, corruption, and incomplete writes when the expected values
come from a trusted channel. They are not digital signatures. An attacker able
to replace both cache files can rewrite the label and checksum, and the loader
does not derive a digest from the live model parameters.

Do not reuse cache artifacts across unknown publishers or model revisions. For
the model bundle that should supply the trusted identity, see
[Native model bundles](native-bundles.md).
