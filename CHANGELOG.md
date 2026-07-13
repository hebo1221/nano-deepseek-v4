# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI coverage, lint, type-check, packaging, security scanning, and release
  artifact workflows.
- Typed package marker and public result/data types.
- Optional CUDA FP32/BF16/FP16, cache, and optimizer smoke tests.
- Versioned, checksummed, configuration-bound KV-cache persistence.
- Checkpoint streaming, path-safety, collision, shape, and metadata validation.
- Reproducible validation evidence for the complete 46-shard official Flash
  checkpoint and an explicit production-readiness support boundary.
- Adaptive V4 Memory research proposal, related-work matrix, falsification
  criteria, and preregistered experimental protocol.
- Security and contribution policies.

### Changed

- SFT truncation now preserves supervised response tokens and EOS.
- Top-p sampling retains the token that crosses the probability threshold.
- Configuration and model input validation now fail early with actionable
  errors.
- Cached attention masks must cover the complete past-plus-current key sequence;
  invalid floating-point configuration values are rejected.

### Fixed

- Official checkpoint examples now match the loader's return type.
- Shared-storage cache tensors are cloned before safetensors serialization.
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

[Unreleased]: https://github.com/hebo1221/nano-deepseek-v4/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hebo1221/nano-deepseek-v4/tree/v0.1.0
