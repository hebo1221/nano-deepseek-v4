# Documentation

The root README is the short installation and first-run path. This index routes
to the detailed contracts, commands, and claim boundaries that are useful after
the first successful run.

All commands in these documents assume the repository root as the working
directory and the current source installed by the
[PyTorch-first installation guide](guides/installation.md), unless a document
says otherwise.

## Guides

- [Installation](guides/installation.md): CPU-safe PyPI and editable installs,
  accelerator setup, optional extras, and the first no-download check.
- [Read the model in ten minutes](guides/modeling-walkthrough.md): run
  `nano-deepseek-v4 tour` and follow one real tensor through mHC, sliding,
  CSA/HCA memory, MoE routing, MTP, and cached decoding.
- [Train and generate](guides/train-and-generate.md): verify the fixed CPU
  learning contract, customize byte-text training, save a generation-ready
  bundle, and use a local or user-owned Hugging Face repository.
- [Native bundles](guides/native-bundles.md): v1/v2 formats, tokenizer binding,
  checksum verification, and loading.
- [Official checkpoints](guides/official-checkpoints.md): allocation-free
  Flash/Pro inspection, pinned 0731 DSpark metadata checks, revision-pinned Hub
  inspection, and local snapshot verification.
- [Cache persistence](guides/cache-persistence.md): save and reload inference
  cache state with configuration and model-revision binding.
- [CPU-first interactive tutorial](../notebooks/01_flash_inference.ipynb): run
  the tiny architecture, train and verify a tokenizer-bound bundle, generate
  from it, and inspect the official model boundaries before any optional
  checkpoint download.

## Verification

- [Conformance profiles](verification/conformance.md): one fixed
  `core`/`dspark`/`offline`/`live` run with explicit network policy, exit codes,
  and a deterministic machine receipt.
- [DSpark semantic vectors](verification/dspark.md): portable CPU-offline
  checks for the tiny draft equations, their pinned provenance, and the exact
  boundary of the resulting receipt.
- [DSpark prefix scheduler](verification/dspark-scheduler.md): Algorithm 1 and
  Section 5.2 CPU arithmetic and causal-trace checks on calibrated probability
  and synthetic SPS inputs.
- [Attention reachability](verification/attention-reach.md): deterministic
  hybrid/local/full structural conformance at the exact local receptive-field
  boundary.
- [Transformers parity](verification/transformers-parity.md): pinned full,
  cached, backward, YaRN, and MTP differential checks.

These are correctness and conformance tools. They are not substitutes for
pretrained-model quality or optimized-kernel benchmarks.

## Research reproduction

- [Research index](research/README.md): choose between fast receipt checks,
  evaluation replay, and full training replay.
- [Tiny Shakespeare control](research/tiny-shakespeare.md): the 8.50M byte
  model, one matched all-sliding run, paired evaluation, exact commands, and a
  deliberately narrow claim boundary.

## Source tour

- [Architecture](architecture.md): report-to-code map and package layout.
- [`PRODUCTION_READINESS.md`](../PRODUCTION_READINESS.md): supported boundary and
  release gates.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md): development setup and change
  expectations.
- [`SECURITY.md`](../SECURITY.md): supported versions and private vulnerability
  reporting.
