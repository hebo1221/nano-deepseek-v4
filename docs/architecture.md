# Architecture tour

`nano_deepseek_v4/modeling.py` follows the DeepSeek-V4 report top-to-bottom and
keeps the model equations in one readable source file. Checkpoint, experiment,
and verification tooling lives in separate modules so it does not obscure that
path.

| Section | Report section | Main implementation surfaces |
| --- | --- | --- |
| Manifold-Constrained Hyper-Connections | §2.2 | `HyperConnection`, `HyperHead` |
| Sliding / CSA / HCA hybrid attention | §2.3 | `DeepSeekV4Attention`, `CSACompressor`, `HCACompressor` |
| Lightning Indexer for sparse top-k | §2.3.1 | `CSAIndexer` |
| Shared K=V MQA with partial RoPE | §2.3.3 | `apply_partial_rope`, `attention_sink`, `GroupedLinear` |
| Hash-MoE bootstrap and routed MoE | §2.1 | `DeepSeekV4MoE`, `SwiGLUExpert` |
| Multi-Token Prediction | §2.1 | `DeepSeekV4MTPModule` |
| DSpark checkpoint dialect | 0731 official source and arXiv:2607.05147v1 | `DeepSeekV4Config.flash_0731()` metadata, `dspark.py`/`dspark_oracle.py` tiny semantic vectors, and `dspark_scheduler.py`/`dspark_scheduler_oracle.py` CPU scheduler traces |
| Muon optimizer | §2.4 | `Muon`, `zeropower_via_newton_schulz` in `optim.py` |

Flash's sliding bootstrap layers and the Flash/Pro CSA/HCA schedules are encoded
in `DeepSeekV4Config.flash()` and `DeepSeekV4Config.pro()`. The presets preserve
the official compressed-branch YaRN settings and the MTP suffix of the attention
schedule. `DeepSeekV4Config.flash_0731()` models the newer checkpoint metadata
without rewriting its raw `num_nextn_predict_layers=1`: the three trailing
compression entries are represented as a separate DSpark stage schedule.

DSpark is not routed through `DeepSeekV4MTPModule`. The architecture and
checkpoint reports expose its stage layout and parameter inventory. A separate
tiny dense fixture checks the outer proposal equations without constructing the
official model. A second fixture checks Algorithm 1 and Section 5.2 scheduler
arithmetic using calibrated probability and synthetic SPS inputs. Official-
checkpoint model construction still raises an explicit unsupported-runtime
error until weight loading, acceptance, scheduler integration, cache, and
kernel paths have independent hardware evidence.

## Package layout

```text
nano_deepseek_v4/
├── _receipts/        # wheel-bundled fixed-run and metadata contracts
├── __main__.py        # `python -m nano_deepseek_v4` entry point
├── cli.py             # lazy unified dispatcher; legacy entry points remain
├── config.py          # native config, official JSON ingestion, Flash/Pro/0731 presets
├── modeling.py        # CSA, HCA, sliding attention, mHC, MoE, and MTP
├── optim.py           # Muon and AdamW group splitter
├── paged_cache.py     # block-table KV cache operations
├── checkpoint.py      # native/official safetensors, DSpark metadata, cache persistence
├── data.py            # CLM packing and SFT label masking
├── training.py        # train step, GRPO, and distillation losses
├── evaluation.py      # language-model and multiple-choice evaluation
├── architecture.py    # allocation-free inspection CLI
├── conformance.py     # fixed profiles and aggregate machine receipts
├── dspark.py          # packaged vectors, separate native path, focused CLI
├── dspark_oracle.py   # independent dense FP32 equation oracle
├── dspark_scheduler.py # CPU scheduler, packaged traces, and focused CLI
├── dspark_scheduler_oracle.py # exhaustive independent scheduler oracle
├── attention_reach.py # structural attention-path conformance
├── official_parity.py # pinned independent equation comparison
├── verify_flash_0731_receipt.py # installable pinned metadata replay
├── tokenizer.py       # byte-v1 tokenizer and strict sidecar schema
├── train_text.py      # deterministic byte-text training CLI
├── reproduce.py       # fixed CPU training, acceptance, and bundle publication
├── generate_text.py   # verified local/Hub bundle generation
├── compare_bundles.py # sequential paired held-out evaluation
├── aggregate_comparisons.py # cross-run contract validation
└── demo.py            # self-checking CPU architecture/cache demo
```

`reproduce.py` loads
`_receipts/tiny-text-training-baseline.json` through package resources, so the
fixed learning contract travels with an installed wheel or source distribution.
`dspark.py` similarly loads `_receipts/dspark-semantic-v1.json`; the JSON carries
fixed inputs, weights, intermediates, outputs, and source digests rather than a
runtime RNG recipe. `dspark_scheduler.py` loads
`_receipts/dspark-scheduler-v1.json`, whose calibrated probabilities, synthetic
SPS tables, expected allocations, and traces are independent of model weights.

## Scope boundary

The readable implementation does not provide production tensor-parallel or
expert-parallel serving, DualPipe training, multi-node communication, or
hardware-specific FP4 attention kernels. The checkpoint streaming path verifies
payloads and conversion coverage; it is not a production inference runtime.

For executable fidelity evidence, use the
[conformance profiles](verification/conformance.md) or the focused
[Transformers parity protocol](verification/transformers-parity.md). For the
combined attention graph boundary, use the
[attention reachability protocol](verification/attention-reach.md).
For the bounded three-stage draft-equation boundary, use the
[DSpark semantic vectors](verification/dspark.md); they are not evidence that
the official checkpoint can be loaded or served.
For the separate Algorithm 1 and Section 5.2 CPU arithmetic boundary, use the
[DSpark prefix scheduler](verification/dspark-scheduler.md). It does not validate
calibration, a hardware profile, speculative acceptance, losslessness, speed,
or serving behavior.
