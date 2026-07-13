# Production readiness

This repository is maintained as a production-quality **reference library**, not
as a frontier-scale training or serving platform. Its supported production
boundary is deterministic local model experimentation, package consumption,
checkpoint inspection/conversion evidence, and inference-cache persistence.

## Supported boundary

| Area | Status | Evidence or contract |
| --- | --- | --- |
| Tiny reference model on CPU | Supported | Forward, loss, generation, cache, optimizer, and evaluation tests |
| CUDA mixed precision | Supported when CUDA is available | BF16/FP16 cache equivalence, native BF16 training, and cache round-trip tests |
| Official Flash snapshot inspection | Supported | All indexed shards, keys, dtypes, shapes, and payloads scanned |
| Official checkpoint conversion | Supported | Streaming conversion report; materialized loading requires sufficient host/device memory |
| Inference-cache persistence | Supported | Atomic write, versioned manifest, config binding, checksum, schema and shape validation |
| Python package distribution | Supported | Python 3.10–3.14 CI, wheel/sdist inspection, isolated-install smoke test |
| Multi-node training or low-latency frontier serving | Not supported | Requires distributed runtime and optimized kernels outside this nano reference's scope |

The checked-in
[`DeepSeek-V4-Flash-validation.summary.json`](references/DeepSeek-V4-Flash-validation.summary.json)
records the content digests and counts from a full official Flash checkpoint
preflight and streaming payload scan. It contains no machine-local paths.

## Required release gates

Every release candidate must pass:

1. Ruff, mypy, the complete pytest suite, and the configured coverage floor.
2. Wheel and source-distribution build plus `twine check`.
3. Installation and a forward pass from the built wheel in an isolated environment.
4. CodeQL and dependency review in GitHub Actions.
5. Tag-to-package-version validation and artifact provenance attestation.

CUDA tests skip cleanly on CPU-only runners; a release that changes kernels,
cache state, dtype conversion, or checkpoint loading must also be exercised on a
CUDA runner before publication.

## Operational expectations

- Treat checkpoint and cache files as untrusted inputs until their path,
  manifest, checksum, tensor metadata, and configuration checks pass.
- Prefer the streaming checkpoint report for official snapshots when full model
  materialization would exceed available memory.
- Pin the exact model configuration alongside persisted caches. Cache loading
  intentionally rejects even subtle configuration drift.
- Do not interpret the reference architecture's correctness checks as latency,
  throughput, benchmark-reproduction, or distributed-fault-tolerance claims.
- Report security issues through the private process in `SECURITY.md`.
