# nano-deepseek-v4

[![CI](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml)
[![CodeQL](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

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
#   hf download deepseek-ai/DeepSeek-V4-Flash \
#       --local-dir ./checkpoints/flash

snapshot = "./checkpoints/flash"

# 1) Verify shard/key/shape integrity without loading tensor payloads:
report = verify_deepseek_checkpoint_snapshot(snapshot)
assert report.is_complete

# 2) Build the model from the official config:
config = DeepSeekV4Config.from_official_json(f"{snapshot}/config.json")
model = DeepSeekV4ForCausalLM(config)

# 3) Convert and load the official safetensors shards into `model`:
report = load_deepseek_official_checkpoint(model, snapshot)
print(len(report.conversion.converted_keys), "tensors loaded")
```

For snapshots too large to materialize with a model, use
`build_deepseek_official_checkpoint_streaming_load_report` to scan every tensor
payload and emit conversion evidence without constructing the full model.

The checked-in
[`DeepSeek-V4-Flash-validation.summary.json`](references/DeepSeek-V4-Flash-validation.summary.json)
records a complete 46-shard official Flash preflight and streaming payload scan,
including content digests, dtype/shape coverage, and logical parameter counts.

## Persisting an inference cache

```python
from nano_deepseek_v4 import load_deepseek_v4_cache, save_deepseek_v4_cache

prefill = model(ids, use_cache=True)
assert prefill.past_key_values is not None
save_deepseek_v4_cache(prefill.past_key_values, "./cache/session-1")
cache = load_deepseek_v4_cache(config, "./cache/session-1")
```

Cache files are written atomically and bound to the exact model configuration.
The loader verifies the manifest version, payload SHA-256, tensor schema, layer
count, shapes, and position ranges before returning a cache.

## Research: Adaptive V4 Memory

The first research track studies risk-aware, reversible hot/cold residency for
V4's heterogeneous long-context memory. It asks whether native Lightning
Indexer statistics can adapt per-layer CSA budgets, indexer refresh, and block
prefetch while retaining the complete compressed history for later turns.

Start with the [`Adaptive V4 Memory` research index](research/adaptive_v4_memory/README.md).
The proposal, related-work matrix, and preregistered experimental protocol make
no performance claim. M0 instrumentation and the M1 replay/predictive-signal
gate and M2 offline training-free controller gate are complete. M3 learned-risk
evaluation is a negative result. M4 established correct fixed-top-k tiering;
M5 then falsified the online adaptive M2 hypothesis.

M0 tracing is an observer-only API. It records native CSA block selections and
cache byte accounting without logging token IDs or changing cache residency:

```python
from nano_deepseek_v4 import AdaptiveMemoryTraceCollector, MemoryTraceConfig

trace = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="experiment-001"))
output = model(ids, use_cache=True, memory_trace=trace)
trace.write("./artifacts/experiment-001")
```

The output directory contains append-ordered `events.jsonl` and an atomic
`manifest.json` with the payload SHA-256. `load_memory_trace` validates the
schema version, identity, sequence, count, and digest before replay.

Schema v2 traces also carry complete causal indexer rankings and resident bytes.
The `adaptive-v4-replay` CLI runs deterministic native, recency, random, top-k,
top-p, per-layer, and index-reuse policies. The checked M1 Tier-S result shows
native features improving sufficient-budget MAE by 11.9% and 16.3% at two model
scales, while explicitly withholding a research claim because only one seed and
one synthetic task were evaluated.

The M2 controller combines score concentration and overlap signals under a
global block budget, supports protected pinning and uncertainty fallback, and
emits digest-replayable actions. At memory-matched operating points it improved
associative-recall accuracy over fixed top-k at both tested scales. This remains
offline logical-selection evidence: physical GPU residency does not change.

The M3 learned risk controller reduced some prediction errors but used 57–63%
dense fallback and failed to improve the M2 quality/block Pareto at both scales.
It remains a reproducible negative baseline rather than the M4 default.

M4's pinned-CPU/GPU-hot CSA value store preserved quality and reduced that
component by about 91%, but reduced total cache allocation by only about 2% and
did not improve speed. M5 connected M2 to actual cached decode; its one-token
lookahead respected every budget but failed quality on both scales and all
three final workload families. Official Flash/Pro execution was infeasible on
the audited host. The research index links the digest-bound final report.

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
├── memory_trace.py   # M0/M1 passive CSA/cache trace and digest validation
├── memory_replay.py  # deterministic replay policies, oracle, and signal analysis
├── memory_probe.py   # opt-in differentiable CSA research objectives
├── memory_controller.py # M2 global-budget rules, fallback, plans, and replay
├── learned_memory_controller.py # M3 calibrated learned-risk negative baseline
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
and use the metadata in [`CITATION.cff`](CITATION.cff) for this implementation.

## Project policies

See [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md),
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and
[`CHANGELOG.md`](CHANGELOG.md). The readiness document defines the supported
production boundary and the required release gates.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
