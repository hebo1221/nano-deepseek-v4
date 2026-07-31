# nano-deepseek-v4

[![CI](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml)
[![CodeQL](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/hebo1221/nano-deepseek-v4/blob/main/LICENSE)

A compact, readable PyTorch reference implementation of the **DeepSeek-V4**
architecture, in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT).
The model code in `nano_deepseek_v4/modeling.py` is a single ~1,300-line file
that covers the major architectural components described in the
[DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348):

- hybrid attention with **sliding attention**, **Compressed Sparse Attention (CSA)**,
  and **Heavily Compressed Attention (HCA)**
- shared K=V **multi-query attention with partial RoPE** and learnable attention sinks
- **Manifold-Constrained Hyper-Connections (mHC)** residual streams with
  Sinkhorn-Knopp doubly-stochastic projection
- **hash-routed bootstrap MoE** layers and **learned top-k routed MoE** layers
  with `sqrt(softplus)` affinity scoring
- **Lightning Indexer** for sparse top-k selection
- **Multi-Token Prediction (MTP)** module aligned to official `e_proj`/`h_proj` keys
- official Flash/Pro `config.json` ingestion and checkpoint key conversion
- **FP4 E2M1** and **FP8 E4M3FN** scale-sidecar dequantization
- **Muon-style** momentum optimizer with Newton-Schulz matrix orthogonalization

The defaults are deliberately tiny so the model runs on CPU. The library can
inspect official Flash/Pro snapshots and convert them into this implementation's
layout; materializing the full 284B/1.6T models still requires correspondingly
large memory and is not the quickstart path.

This is meant to be **executable, readable, and hackable** — not a production
training framework.

Unlike the Hugging Face Transformers implementation, which optimizes for broad
ecosystem integration, this repository keeps the architecture compact enough to
read end-to-end and modify in small experiments.

## Sixty-second CPU demo

Install a checkout (this also works before the first PyPI publication):

```bash
git clone https://github.com/hebo1221/nano-deepseek-v4.git
cd nano-deepseek-v4
python -m pip install -e .
```

After the v0.2.0 publish workflow completes, the equivalent pinned install is:

```bash
python -m pip install "nano-deepseek-v4==0.2.0"
```

Then run the no-download demo:

```bash
nano-deepseek-v4-demo
# The module form is equivalent:
# python -m nano_deepseek_v4.demo
```

Expected structure:

```text
nano-deepseek-v4 0.2.0
parameters: 1,023,364
attention: sliding -> sliding -> compressed_sparse -> heavily_compressed
mlp: hash_moe -> hash_moe -> hash_moe -> moe
logits: (1, 8, 512)
cache tokens: 5 -> 8
cached/full match: True
router layers: 4
```

It constructs the complete tiny architecture and checks that chunked cached
inference matches a full CPU forward pass. The weights and token IDs are random:
this is an architecture/cache smoke test, not a text-generation quality demo.

For checkpoint-download helpers or development tools:

```bash
pip install -e ".[official]"   # editable + huggingface_hub for checkpoint download
pip install -e ".[dev]"        # tests, coverage, lint, typing, and packaging tools
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.4.

## Quickstart — tiny model on CPU

```python
import torch
from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM

config = DeepSeekV4Config()                    # 1M-param toy default
model = DeepSeekV4ForCausalLM(config)
ids = torch.randint(0, config.vocab_size, (1, 16))
out = model(ids, labels=ids)
print(out.logits.shape, out.loss.item())       # tiny CPU-runnable smoke path
```

## Inspecting an official Flash checkpoint

```python
from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_official_checkpoint,
    verify_deepseek_checkpoint_snapshot,
)

# Download the official Flash snapshot first:
#   hf download deepseek-ai/DeepSeek-V4-Flash \
#       --local-dir ./checkpoints/flash

snapshot = "./checkpoints/flash"

# 1) Verify shard/key/shape integrity without loading tensor payloads:
report = verify_deepseek_checkpoint_snapshot(snapshot)
assert report.is_complete

# 2) Read the official-format fields supported by this implementation:
config = DeepSeekV4Config.from_official_json(f"{snapshot}/config.json")

# 3) Full loading is intentionally explicit and memory-heavy:
model = DeepSeekV4ForCausalLM(config)
report = load_deepseek_official_checkpoint(model, snapshot)
print(len(report.conversion.converted_keys), "tensors loaded")
```

`load_deepseek_official_checkpoint` materializes the official tensors and the
converted state dictionary; it is not a sharded serving loader. For snapshots
too large to materialize, use
`build_deepseek_official_checkpoint_streaming_load_report` to scan every tensor
payload and emit conversion evidence without constructing a runnable model.

The checked-in
[`DeepSeek-V4-Flash-validation.summary.json`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/references/DeepSeek-V4-Flash-validation.summary.json)
records a complete 46-shard official Flash preflight and streaming payload scan,
including content digests, dtype/shape coverage, and logical parameter counts.

## Persisting an inference cache

```python
import torch
from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)

cache_config = DeepSeekV4Config()
cache_model = DeepSeekV4ForCausalLM(cache_config).eval()
cache_ids = torch.tensor([[1, 2, 3, 4]])
prefill = cache_model(cache_ids, use_cache=True)
assert prefill.past_key_values is not None
revision = "sha256:<trusted-checkpoint-digest>"
save_deepseek_v4_cache(
    prefill.past_key_values,
    "./cache/session-1",
    model_revision=revision,
)
cache = load_deepseek_v4_cache(
    cache_config,
    "./cache/session-1",
    model_revision=revision,
)
```

Cache files are written atomically and bound to the exact model configuration
plus a caller-supplied model revision or digest label. The loader checks that
label for exact equality and verifies the manifest version, payload SHA-256,
tensor inventory, dtypes, shapes, layer count, and position relationships before
returning a cache.

The label and checksum guard against accidental mismatch and storage corruption
when their expected values come from a trusted channel. They are not signatures:
an attacker who can replace both cache files can rewrite the label and checksum.
The loader does not derive a model digest or authenticate cache provenance.

## What's inside

```
nano_deepseek_v4/
├── config.py         # DeepSeekV4Config, flash() / pro() / from_official_json()
├── modeling.py       # CSA + HCA + sliding attention, mHC, MoE, MTP head — single file
├── optim.py          # Muon + AdamW group splitter
├── paged_cache.py    # block-table KV cache with append / read / crop / evict
├── checkpoint.py     # safetensors I/O + official key conversion + FP4/FP8 dequant
├── data.py           # CLM packing, SFT batch builder with label masking
├── training.py       # single-step training loop, GRPO loss, distillation loss
├── evaluation.py     # next-token perplexity, multiple-choice scoring
└── demo.py           # self-checking CPU architecture/cache demo
```

The stable package intentionally remains small enough to read. Large experimental
harnesses and unpublished research results are kept off the default branch.

## Architecture tour

`nano_deepseek_v4/modeling.py` follows the DeepSeek-V4 report top-to-bottom:

| section | report § | classes |
| --- | --- | --- |
| Manifold-Constrained Hyper-Connections | §2.2 | `HyperConnection`, `HyperHead` |
| Sliding / CSA / HCA hybrid attention | §2.3 | `DeepSeekV4Attention`, `CSACompressor`, `HCACompressor` |
| Lightning Indexer for sparse top-k | §2.3.1 | `CSAIndexer` |
| Shared K=V MQA with partial RoPE | §2.3.3 | `apply_partial_rope`, `attention_sink`, `GroupedLinear` |
| Hash-MoE bootstrap + routed MoE | §2.1 | `DeepSeekV4MoE`, `SwiGLUExpert` |
| Multi-Token Prediction modules | §2.1 | `DeepSeekV4MTPModule` |
| Muon optimizer | §2.4 | `Muon`, `zeropower_via_newton_schulz` (in `optim.py`) |

Flash's sliding bootstrap layers and the Flash/Pro CSA/HCA schedules are encoded
in `DeepSeekV4Config.flash()` and `DeepSeekV4Config.pro()`.

## What it does not do

- It does not load the official Pro 1.6T checkpoint and serve it at production
  latency. That requires GPU paged-attention kernels, EP all-to-all, multi-node
  NCCL, FP4 hardware paths — out of scope for nano-style readability.
- It does not provide a memory-efficient sharded loader for the full official
  checkpoints; the streaming report is a verifier, not an inference runtime.
- It does not implement training infrastructure (DualPipe, fine-grained EP,
  activation checkpointing) at frontier scale.
- It does not reproduce the published benchmark numbers.

It does provide a compact executable interpretation of the major components so
you can read, modify, and experiment with them.

## Citing

If you found this useful in research, please cite the DeepSeek-V4 report itself
and use the metadata in
[`CITATION.cff`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CITATION.cff)
for this implementation.

## Project policies

See
[`PRODUCTION_READINESS.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/PRODUCTION_READINESS.md),
[`CONTRIBUTING.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CONTRIBUTING.md),
[`SECURITY.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/SECURITY.md),
and [`CHANGELOG.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CHANGELOG.md).
Maintainers can follow
[`RELEASING.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/RELEASING.md)
for the tokenless, provenance-attested PyPI release path. The readiness document
defines the supported production boundary and the required release gates.

## License

Apache-2.0. See
[`LICENSE`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/LICENSE).
