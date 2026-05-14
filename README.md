# nano-deepseek-v4

A compact, readable PyTorch reference implementation of the **DeepSeek-V4**
architecture, in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT).
The model code in `nano_deepseek_v4/modeling.py` is a single ~1,300-line file
that implements every architectural idea from the
[DeepSeek-V4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf):

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

The defaults are deliberately tiny so the model runs on CPU. Loading the
official 284B/1.6T checkpoint is supported via `load_deepseek_official_checkpoint`.

This is meant to be **executable, readable, and hackable** — not a production
training framework.

## Install

```bash
pip install -e ".[official]"   # editable + huggingface_hub for checkpoint download
pip install -e ".[dev]"        # + pytest
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
print(out.loss.item())                         # works on CPU in < 1s
```

## Quickstart — official Flash checkpoint

```python
from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_official_checkpoint,
    verify_deepseek_checkpoint_snapshot,
)

# Download the official Flash snapshot first:
#   huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
#       --local-dir ./checkpoints/flash --local-dir-use-symlinks False

snapshot = "./checkpoints/flash"

# 1) Verify shard/key/shape integrity without loading tensor payloads:
report = verify_deepseek_checkpoint_snapshot(snapshot)
assert report.is_complete

# 2) Build the model from the official config:
config = DeepSeekV4Config.from_official_json(f"{snapshot}/config.json")
model = DeepSeekV4ForCausalLM(config)

# 3) Convert and load the official safetensors shards:
model, conversion = load_deepseek_official_checkpoint(model, snapshot)
print(conversion.converted_key_count, "tensors loaded")
```

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
└── demo.py           # 20-line forward demo
```

Total: ~5,000 lines of code. Read it.

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

The Flash and Pro layer schedules (CSA ↔ HCA interleave with sliding bootstraps)
are encoded in `DeepSeekV4Config.flash()` and `DeepSeekV4Config.pro()`.

## What it does not do

- It does not load the official Pro 1.6T checkpoint and serve it at production
  latency. That requires GPU paged-attention kernels, EP all-to-all, multi-node
  NCCL, FP4 hardware paths — out of scope for nano-style readability.
- It does not implement training infrastructure (DualPipe, fine-grained EP,
  activation checkpointing) at frontier scale.
- It does not reproduce the published benchmark numbers.

It does implement every architectural idea cleanly enough that you can read,
modify, and experiment.

## Citing

If you found this useful in research, please cite the DeepSeek-V4 report itself
and link this repository. A standalone `CITATION.cff` will be added with the
first tagged release.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
