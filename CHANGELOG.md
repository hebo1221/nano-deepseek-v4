# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
