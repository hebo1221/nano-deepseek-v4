# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Fixed, wheel-packaged DSpark semantic vectors and a no-download
  `nano-deepseek-v4 dspark` checker. A separately written native path and dense
  FP32 oracle reproduce 39 intermediate/output tensors and exact proposals for
  the official three-stage, five-token outer draft equations. Six deliberate
  mutations guard target-layer order, non-causal within-block attention,
  previous-token Markov bias, and confidence input/sigmoid behavior. The
  focused receipt keeps official weights, speculative acceptance, scheduling,
  kernels, performance, serving, training, and quality explicitly out of scope.
- A separate no-download `nano-deepseek-v4 dspark-scheduler` checker for the
  CPU arithmetic and causal traces in DSpark Algorithm 1 and Section 5.2. Fixed
  calibrated-probability and synthetic-SPS vectors plus invariant checks
  exercise strict greedy stopping, deterministic ties, Appendix A confidence
  cases, and caller-supplied lagged capacity selection on a jagged curve against
  an exhaustive oracle. The
  `dspark` conformance profile requires both DSpark checks while keeping STS
  quality, measured hardware, rejection sampling, end-to-end losslessness,
  speed, serving, model quality, and official-weight runtime out of scope.
- Cumulative `nano-deepseek-v4 conformance` profiles for local CPU execution,
  pinned offline parity, and an explicit live metadata/header replay.
  Each run reports every selected or skipped check, dependency and network
  state, implementation and nested-receipt digests, claim boundaries, and
  stable exit codes. It can write one deterministic JSON receipt; no profile
  loads official weights. Clean-wheel receipts for all four profiles are
  hash-bound into the release manifest, provenance-attested, and attached to the
  GitHub release. Offline checks account for blocked standard-library socket
  attempts even when the caller swallows the exception; CI and release
  `core`/`offline` receipts additionally require a capability-dropped Linux
  network namespace and record its enforcement scope.
  Cumulative live runs also restore Hugging Face's import-time offline and
  telemetry flags with Hub-compatible precedence, discard cached Hub HTTP
  sessions, preserve Python/NumPy/Torch RNG state, and contain lazy
  TorchInductor cache setup before the live phase.
- A machine-readable CUDA release gate that dynamically collects every
  GPU-marked accelerator test, requires an all-pass zero-skip run, records exact
  node IDs and CUDA/device/runtime provenance, and binds the result to a clean
  commit plus the complete CUDA-sensitive source inventory. The CPU-only tag
  workflow verifies current bytes and tested-commit ancestry, hashes the receipt
  into the release manifest, and publishes and attests the exact JSON evidence.
- A standalone release-source preflight that validates version surfaces,
  transition claims, real calendar dates, changelog ordering and links, and the
  requested tag before a protected tag is pushed. The tag workflow runs the
  same tested executable instead of maintaining an inline copy.
- A structured GitHub bug-report form that requires the affected surface,
  install provenance, exact version or commit, environment, minimal
  reproduction, expected and actual behavior, complete sanitized output, and
  immutable Hub revision when relevant. The issue picker routes questions,
  feature proposals, and vulnerabilities to their dedicated public or private
  channels instead of accepting unstructured bug reports.
- A discoverable `nano-deepseek-v4` command, mirrored by
  `python -m nano_deepseek_v4`, that lazily routes to every installed demo,
  training, generation, inspection, verification, comparison, and aggregation
  tool while preserving all legacy entry points. Root help groups workflows
  and `--version` reports the package version without loading optional Hub or
  Transformers dependencies.
- A unified `nano-deepseek-v4 reproduce` first-run command that executes the
  wheel-bundled, fixed 20-step CPU learning contract, independently verifies a
  generation-ready native v2 bundle, and only publishes a requested bundle after
  portable acceptance passes. Its machine receipt keeps implementation-bound
  exact historical losses, sample bytes, and artifact digests informational
  instead of making them cross-runtime pass conditions.
- A single PyTorch-first installation guide for CPU, accelerator, source, PyPI,
  and optional-extra environments. Every source and contributor path now
  installs CPU PyTorch before the package, and the Flash-0731 replay explicitly
  installs its required `official` extra.
- A `nano-deepseek-v4-generate` CLI for deterministic or seeded text generation
  from verified native bundles. Native bundle format v2 binds an explicit
  byte-v1 tokenizer sidecar into the existing SHA-256 inventory; v1 model-only
  bundles remain loadable but are never guessed to be text-compatible. Local
  paths and revision-pinned Hugging Face repositories share the same verifier,
  and JSON output records immutable source provenance, config/tokenizer/manifest
  digests, and exact continuation token IDs.
- A deterministic `nano-deepseek-v4-attention-reach` conformance CLI that
  perturbs source tokens across the mathematically derived three-layer
  window-32 boundary and compares the native hybrid stack with near-parameter-
  matched local-only and full-window controls. Its digest-bound report measures
  discrete numerical influence at a pinned tolerance and a complementary
  embedding-output vector-Jacobian product, not learned retrieval or model
  quality.
- Native `save_pretrained` / `from_pretrained` model bundles with configuration
  round-tripping, deterministic size-based safetensors sharding, SHA-256 file
  inventory, staged publication, and shard-at-a-time meta-device loading.
- A no-extra-dependency `nano-deepseek-v4-train` experiment that trains the complete
  tiny hybrid stack on built-in or user-provided byte text, reports deterministic
  validation metrics and experiment-identity digests, and verifies an optional
  saved model bundle. The CLI also provides an 8.50M mini preset, native config
  loading, seeded temperature/top-p generation, and reproducible bundle digests.
  Checked-in CPU-smoke and Tiny Shakespeare/GB10 receipts record both acceptance
  contracts without shipping generated weights. A matched all-sliding control
  records the short-context throughput tradeoff and an explicit null quality
  result instead of implying hybrid-attention superiority from one run.
- An allocation-free `nano-deepseek-v4-inspect` CLI for exact official-schema
  total/activated parameter counts, layer schedules, configuration digests, and
  model-free native-bundle integrity verification down to shard headers and keys.
  Official checkpoints can also be audited by key namespace with a canonical
  dtype/shape inventory digest and logical parameter accounting.
- Commit-pinned Hugging Face checkpoint inspection that downloads only the
  official config and safetensors index as files, uses bounded header ranges only
  for shards in the requested namespace, and reports that full-snapshot payload
  integrity remains out of scope. The pinned Flash MTP digest matches the
  independent 159.6 GB local checkpoint audit.
- A pinned `flash-0731` architecture preset and metadata-only DSpark checkpoint
  contract. The official `num_nextn_predict_layers=1` field remains intact while
  the three checkpoint stages are derived separately, validated across shards
  46–48, and recorded in a checked-in 4,705-tensor inventory receipt. Native
  model construction fails closed because official-checkpoint DSpark execution
  is not implemented.
- An installable `nano-deepseek-v4-verify-flash-0731` receipt verifier and
  wheel-bundled canonical receipt/config. It pins and hashes four allowlisted
  Hub files, compares a fresh three-shard metadata report, and rejects any run
  that leaves a cached safetensors file in its dedicated cache.
- A pinned `nano-deepseek-v4-parity` differential verifier and checked-in
  receipt covering full forward, irregular chunked cache decode, and backward
  gradients against Transformers 5.15.0, plus independently composed MTP
  fusion and stream collapse against a revision-pinned official inference
  source. It also checks official-dimension main/YaRN inverse frequencies and
  boundary-position rotary vectors without downloading model weights.
- A post-publish release gate that verifies PyPI's public version API exposes
  exactly the wheel and source distribution SHA-256 digests recorded during the
  build before creating the immutable GitHub release.
- A `nano-deepseek-v4-compare` CLI that checksum-verifies two native bundles,
  evaluates them sequentially on identical held-out byte windows, separates
  causal-LM and MTP losses, and emits evaluator-bound descriptive paired metrics
  without treating overlapping windows as independent samples.
- A `nano-deepseek-v4-aggregate` CLI that rejects evaluator, corpus, config,
  runtime, and batching drift before summarizing paired effects across distinct
  bundle manifests. Run labels remain explicitly user-supplied rather than being
  treated as proof of independent training seeds.

### Changed

- Wheel and source-distribution smoke tests now run the base-install unified
  `reproduce` → `generate` → DSpark conformance journey before optional extras
  are installed. They verify that the generated receipt is bound to the newly
  reproduced bundle's manifest, config, tokenizer, and parameter count while
  retaining a lightweight compatibility check for the legacy generation entry
  point.
- Release documentation now distinguishes enforced CPU CI from manual CUDA
  validation, mirrors the sdist-to-rebuilt-wheel parity gate locally, and
  documents how to verify published conformance receipts against the release
  manifest and GitHub attestation.
- The required type-check gate now covers developer scripts and tests in
  addition to the installed package.
- The `official` extra no longer installs the unused Hugging Face `tokenizers`
  package; it contains `huggingface_hub` plus the lightweight PEP 440 version
  parser used to gate the live conformance client.
- Flash-0731 verification now separates semantic drift, transient source or
  selected-header unavailability, missing optional dependencies, and local
  execution failures. Stable JSON outcomes and exit codes no longer leak local
  paths or raw transport details, and header-request transport failures can no
  longer be flattened into metadata drift.
- Architecture and namespace JSON reports distinguish conventional MTP from
  DSpark and expose speculative parameter counts, verification scope, schema
  compatibility, payload-integrity status, and native-runtime support.
- Local checkpoint inspection now requires `safetensors>=0.6.1`, the first
  supported release in this project's tested range that parses both the
  `F8_E4M3` weight and `F8_E8M0` scale metadata used by Flash-0731.
- The byte-text trainer can atomically persist its complete JSON receipt with
  `--output`, so long runs do not depend on captured standard output.
- Autoregressive generation can sample with an explicit generator and can keep
  producing fixed-length corpus probes after EOS, while greedy EOS-stopping
  behavior remains the default.
- The low-level safetensors loader now offers an opt-in progressive path that
  validates the complete key and shape inventory before mutating a model.
- Training labels are validated before model/cache execution, and MTP loss with
  a pre-existing KV cache is rejected until prediction-depth cache state is
  implemented instead of silently using incomplete context.

### Fixed

- The interactive tutorial now follows the no-download v0.3 path before any
  official-checkpoint work, uses the current index-coverage report fields,
  describes official packed FP4 storage as signed `I8` raw bytes where
  applicable, and is discoverable from the README and source distribution.
  CI executes both its default cell sequence and a real Jupyter-kernel smoke
  inside an OS-isolated network namespace.
- Both public native model constructors now reject DSpark configurations before
  allocating parameters; the lower-level `DeepSeekV4Model` could previously
  bypass the metadata-only runtime boundary.
- Checkpoint preflight now rejects scale sidecars attached to full-precision
  tensors instead of treating arbitrary shapes as verified quantization
  metadata.
- mHC now follows report equations 6-8 and the official inference kernel:
  `2 * sigmoid` output weights, exponential/softmax Sinkhorn initialization,
  and the checkpoint-defined transposed residual-map consumption direction.
- CSA and HCA layers now apply the compressed-branch RoPE base to their local
  queries and KV entries, matching the official implementation. The previous
  path incorrectly used the sliding-only RoPE base for every layer.
- Official Flash/Pro `rope_scaling` is preserved during config ingestion and
  round-tripping. CSA/HCA attention, compressors, and the Lightning Indexer now
  use the checkpoint's YaRN factor, correction range, and fixed unit magnitude;
  the setting was previously discarded and compressed positions used plain RoPE.
- MoE blocks now persist only the routing state they actually use: hash layers
  keep `tid2eid`, learned layers keep their correction bias, and MTP blocks use
  the learned-routing layout found in official checkpoints. This removes about
  254 MB of unused Flash state and 366 MB of unused Pro state.
- MTP now fuses future-token embeddings into every incoming mHC residual stream,
  carries that stream state through successive prediction depths, and trains
  each depth against its correct future target. The previous implementation
  collapsed the backbone state before MTP and then duplicated it.
- Official `compress_ratios` suffix entries now define the MTP attention
  schedule instead of being discarded as output sentinels. This also corrects
  Pro's allocation-free total and activated parameter counts.
- Sharded safetensors writing now clones shared-storage tensors, including tied
  input/output embeddings, before serialization.

## [0.2.0] - 2026-07-31

### Added

- CI coverage, lint, type-check, packaging, security scanning, and release
  artifact workflows.
- A no-download CPU demo available as `nano-deepseek-v4-demo` and
  `python -m nano_deepseek_v4.demo`.
- Provenance-attested GitHub and PyPI publishing through OpenID Connect, with a
  maintainer release checklist.
- Typed package marker and public result/data types.
- Optional CUDA FP32/BF16/FP16, cache, and optimizer smoke tests.
- Versioned, checksummed KV-cache persistence bound to both configuration and a
  caller-supplied immutable model revision.
- Checkpoint streaming, path-safety, collision, shape, and metadata validation.
- Reproducible validation evidence for the complete 46-shard official Flash
  checkpoint and an explicit production-readiness support boundary.
- Security and contribution policies.

### Changed

- Installation and official-checkpoint documentation now distinguish the tiny
  runnable reference path from memory-heavy materialized checkpoint loading.
- Persisted-cache format v2 requires an explicit `model_revision`; legacy v1
  manifests are rejected rather than loaded without model-identity binding.
- SFT truncation now preserves supervised response tokens and EOS.
- Top-p sampling retains the token that crosses the probability threshold.
- Configuration and model input validation now fail early with actionable
  errors.
- Cached attention masks must cover the complete past-plus-current key sequence;
  invalid floating-point configuration values are rejected.

### Fixed

- Official checkpoint examples now match the loader's return type.
- The official-checkpoint notebook no longer describes the materialized loader
  as lazy or links to notebooks that do not exist.
- Shared-storage cache tensors are cloned before safetensors serialization.
- Cache tensor names are rejected from safetensors metadata before tensor
  payloads are loaded.
- Non-integer compression ratios are rejected instead of silently truncated.
- Official compression schedules normalize their output-layer sentinel.
- Signed `I8` official FP4 expert weights now decode from their raw packed
  nibble representation.
- FP4 checkpoint statistics now report logical rather than packed-storage
  parameter counts.
- Corrupt or truncated safetensors headers are reported as failed preflight
  checks instead of aborting snapshot verification.

## [0.1.0] - 2026-05-14

- Initial public release of the compact DeepSeek-V4 Flash/Pro reference
  implementation.

[Unreleased]: https://github.com/hebo1221/nano-deepseek-v4/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hebo1221/nano-deepseek-v4/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hebo1221/nano-deepseek-v4/tree/v0.1.0
