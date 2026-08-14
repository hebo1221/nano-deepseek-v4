# Production readiness

This repository is a release-gated **reference library**, not a frontier-scale
training or serving platform. Its maintained boundary is deterministic local
model experimentation, package consumption, checkpoint inspection/conversion
evidence, and inference-cache persistence.

## Supported boundary

| Area | Status | Evidence or contract |
| --- | --- | --- |
| Tiny reference model on CPU | Supported | Forward, loss, generation, cache, optimizer, and evaluation tests |
| Aggregate conformance profiles | Supported | Fixed `core`, `dspark`, `offline`, and `live` check sets; `core`→`offline`→`live` is cumulative while `dspark` is an offline fork; deterministic machine receipts with per-check digests and claim boundaries; blocked-attempt accounting for local Python socket calls and Linux network-namespace enforcement for CI/release `core` and `offline` receipts; no official weight loading |
| DSpark semantic vectors | Supported for the fixed tiny fixture | Packaged FP32 inputs, weights, 39 intermediate/output tensors, and exact draft IDs are reproduced by separately written native and dense-oracle paths; mutations cover target concat order, non-causal draft attention, Markov recurrence, and confidence semantics; no official weights, acceptance, scheduler, kernels, speed, or quality claim |
| DSpark prefix-scheduler vectors | Supported for the fixed CPU fixture | Algorithm 1 strict causal traces and Section 5.2 caller-supplied lagged capacity/allocation agree with an exhaustive oracle on calibrated probabilities and synthetic SPS tables; temporal provenance and request-slot alignment are not verified, and there is no STS-quality, measured-hardware, rejection-sampling, losslessness, speed, serving, quality, or official-weight runtime claim |
| Native attention-path reachability | Supported for the deterministic fixture | Source-token perturbations and an embedding-output VJP across the three-layer window-32 boundary with hybrid, near-parameter-matched local-only, and full-window controls; [digest-bound receipt](references/attention-reach-conformance.json) and CI rerun |
| Official eager-model differential parity | Supported for the pinned fixture | Full forward, irregular chunked cache decode, and backward gradients against Transformers 5.15.0; official 64-dimensional main/YaRN inverse frequencies and boundary-position rotary vectors; independently composed official MTP fusion/stream collapse; all three attention families and both routing modes |
| Fixed tiny CPU reproduction | Supported | Wheel-bundled 20-step corpus and recipe contract; portable loss, architecture, optimizer, and independently verified native-v2-bundle acceptance; implementation-bound exact historical losses, sample bytes, and digests reported separately as informational observations |
| Byte-text training experiments | Supported | Deterministic fixed-seed loss comparison, all three attention families, MoE, MTP, Muon/AdamW, optional bundle round-trip, a [CPU smoke receipt](references/tiny-text-training-baseline.json), and one [2.56M-token Tiny Shakespeare/GB10 matched control](references/tiny-shakespeare-attention-control.json) with a bounded null-quality claim |
| Official architecture accounting | Supported | Allocation-free Flash/Pro schedules and total/activated parameter counts; the 0731 preset separately reports three DSpark stages and speculative counts without treating `num_nextn_predict_layers=1` as their depth |
| CUDA mixed precision | Manual execution with a source-bound release gate | GPU-marked tests cover BF16/FP16 cache equivalence, native BF16 training, and cache round-trip across sliding, CSA, and HCA layers on a compatible CUDA host; a machine-readable receipt records exact node IDs, outcomes, source bytes, commit, and device/runtime provenance, while the CPU-only tag workflow verifies current-source and commit-ancestry binding |
| Hub metadata-only namespace inspection | Supported | Requested revision resolves to a commit SHA; config/index documents plus bounded header ranges for selected shards are read; low-precision scale sidecars are one-to-one and dtype/shape checked; the original Flash digest matches its independent local audit and the pinned 0731 DSpark namespace has its own receipt |
| Official Flash snapshot inspection | Supported | All indexed shards, keys, dtypes, shapes, and payloads scanned; focused MTP namespace inventory is digest-bound and header-verifiable without loading payloads |
| Official checkpoint conversion | Supported | Streaming conversion report; materialized loading requires sufficient host/device memory |
| DeepSeek-V4-Flash-0731 official-checkpoint DSpark execution | Not supported | Config, index, shapes, dtypes, and selected headers are recognized; the separate tiny semantic and CPU scheduler vectors do not load official weights; model construction fails closed and reports `runtime_load_supported=false` |
| Native model persistence | Supported | Staged, checksummed bundle; allocation-free file/index/shard verification; exact config/weight round-trip; progressive meta-device load; sequential paired-bundle evaluation and contract-checked multi-run aggregation with source-bound output |
| Native byte-text generation | Supported | Bundle v2 binds byte-v1 tokenizer metadata to config and weights; local/Hub generation verifies every manifest-declared model file before allocation and records exact continuation IDs and immutable Hub revision |
| Inference-cache persistence | Supported | Atomic write, versioned manifest, config and caller-supplied model-revision binding, checksum, schema and shape validation |
| Python package distribution | Supported | Python 3.10, 3.12, and 3.14 CI; wheel payload and entry-point inspection; sdist path-safety, source-inventory, and packaged-link checks; sdist-to-wheel payload parity; direct wheel and sdist CPU installs; accelerator-dependency audit; unified and legacy CLI smoke tests |
| Multi-node training or low-latency frontier serving | Not supported | Requires distributed runtime and optimized kernels outside this reference library's scope |

The checked-in
[`DeepSeek-V4-Flash-validation.summary.json`](references/DeepSeek-V4-Flash-validation.summary.json)
records the content digests and counts from a full official Flash checkpoint
preflight and streaming payload scan. It contains no machine-local paths.
The separate
[`DeepSeek-V4-Flash-0731-metadata.json`](nano_deepseek_v4/_receipts/DeepSeek-V4-Flash-0731-metadata.json)
binds a metadata-only DSpark audit to an immutable Hub revision and explicitly
records that full payload integrity and native execution were not verified.

## Required release gates

Every release candidate must pass:

1. Ruff, mypy, the complete pytest suite, and the configured coverage floor.
2. Wheel and source-distribution build plus `twine check`.
3. Installation with the `official` extra, a forward pass, allocation-free
   Flash and 0731 inspections, offline Hub-path fixture checks, fixed 20-step
   reproduction, saved-bundle inspection, byte-text generation and
   comparison, and checked-in multi-run aggregation from the built wheel in an
   isolated environment. The built wheel must also pass
   `nano-deepseek-v4 reproduce --json` and
   `nano-deepseek-v4 dspark --json`,
   `nano-deepseek-v4 dspark-scheduler --json`,
   `nano-deepseek-v4 conformance --profile core --json`, and
   `nano-deepseek-v4 conformance --profile dspark --json` without optional
   dependencies. Its core release evidence must be generated in a verified Linux
   network namespace with no non-loopback interface. Both `nano-deepseek-v4` and
   `python -m nano_deepseek_v4` must expose the same installed command surface,
   while legacy entry points remain callable.
4. The `offline` conformance profile from the built wheel, generated under the
   same verified no-network namespace and including pinned
   differential parity for official main/YaRN rotary vectors and full, cached,
   backward, and MTP composition on the deterministic hybrid fixture.
5. Attention-path reachability across the d94/d95 local receptive-field boundary
   from the built wheel.
6. CodeQL and dependency review in GitHub Actions.
7. Tag-to-package-version validation and provenance attestation for the wheel,
   source distribution, checksums, release manifest, and clean-wheel
   `core`/`dspark`/`offline`/`live` conformance receipts.
8. The `live` conformance profile from the isolated release environment. Its
   Flash-0731 replay must hash the four allowlisted Hub files, match all
   selected-header fields, record the pinned metadata-only network policy, and
   cache zero `.safetensors` files.
9. A verified `references/cuda-release-gate.json`. GPU execution is manual, but
   the receipt must record a clean tested commit, every dynamically collected
   accelerator node ID, an all-pass zero-skip result, CUDA/device provenance,
   and the exact current inventory of all CUDA-bound source files. Any change to
   that inventory requires a fresh compatible-host run.

Hosted CI and the tag workflow pin receipt-producing jobs to Ubuntu 24.04,
drop capabilities inside a fresh network namespace, and require the receipt to
recognize that isolation. CUDA execution remains manual because those runners
are CPU-only. The tag workflow nevertheless fails closed unless the checked-in
CUDA receipt matches the current source inventory and its tested commit remains
an ancestor of the tag. It hashes the receipt into the release manifest and
attests the published JSON bytes. That attestation proves artifact identity, not
that GitHub witnessed the GPU hardware or reran the CUDA tests. See
`RELEASING.md` for the clean-commit generation sequence.

## Operational expectations

- Treat checkpoint and cache files as untrusted inputs until their path,
  manifest, checksum, tensor metadata, and configuration checks pass.
- Save native model bundles into a new or empty directory. Loading verifies the
  complete manifest-declared model inventory before progressively materializing
  model weights; unrelated non-model files are outside that integrity claim.
- Use `nano-deepseek-v4-inspect --bundle PATH` to verify bundle integrity and
  shard inventory without allocating the model or reading tensor payloads.
- Use `nano-deepseek-v4-generate --bundle PATH --prompt TEXT` only with a v2
  bundle whose byte-v1 sidecar is checksum-bound. A v1 model bundle remains
  loadable but is not evidence of any text-to-token mapping.
- Hub generation resolves the requested revision to one immutable commit,
  validates a bounded manifest before downloading its exact file inventory, and
  never executes repository code. The manifest checksum is still integrity
  metadata supplied by that repository, not publisher authentication.
- Bundle checksums provide integrity, not publisher authentication. Quiesce
  training before saving, and bind distributed artifacts to a trusted external
  revision or digest.
- Use `nano-deepseek-v4 reproduce` for the fixed first-run learning and bundle
  proof. Its hard acceptance is portable; exact historical losses, sample bytes,
  and artifact digests are informational diagnostics. Use the trainer directly
  for a custom experiment rather than treating a changed recipe as a replay of
  the fixed contract.
- Prefer the streaming checkpoint report for official snapshots when full model
  materialization would exceed available memory.
- Use `nano-deepseek-v4-inspect --checkpoint PATH --namespace PREFIX --json`
  for a header-only, digest-bound audit of one official key namespace before a
  payload scan or conversion.
- Before downloading a snapshot, use `nano-deepseek-v4-inspect --hf-repo
  OWNER/NAME --revision REVISION --namespace PREFIX --json`. The tool resolves
  the revision to a commit and reads the config, index, and bounded header ranges
  for selected shards. The 0731 receipt verifier additionally requires zero
  cached `.safetensors` files at completion. A successful namespace result does
  not imply full snapshot or payload
  integrity; check `snapshot_preflight_complete` and run the local verifier when
  that stronger claim is required.
- For 0731, also require `checkpoint_family=deepseek_v4_dspark`,
  `schema_compatible=true`, `scale_metadata_verified=true`, and
  `runtime_load_supported=false`. A successful metadata report is not
  permission to construct or serve the native model.
- Use `nano-deepseek-v4 dspark --json` when investigating the bounded draft
  equations. A pass binds the installed vector, native, oracle, config, and
  pinned-source digests; it is not permission to load the official 0731 weights
  or a claim about speculative acceptance, scheduling, latency, or quality.
- Use `nano-deepseek-v4 dspark-scheduler --json` for the separate Algorithm 1
  and Section 5.2 CPU arithmetic receipt. It consumes calibrated probabilities
  and an SPS table; a pass does not validate temporal provenance or request-slot
  alignment, STS, a measured hardware profile, rejection sampling, end-to-end
  losslessness, speed, serving, model quality, or official-weight runtime.
- Pin the exact model configuration and a trusted immutable checkpoint revision
  or digest alongside persisted caches. Cache loading checks the supplied
  identity for exact equality but does not hash model weights itself.
- Do not interpret the reference architecture's correctness checks as latency,
  throughput, benchmark-reproduction, or distributed-fault-tolerance claims.
- The attention-reach fixture uses randomly initialized nano models. Responses
  above the pinned `1e-6` L∞ tolerance and the local control's lack of material
  response beyond lag 93 test combined attention-path connectivity only. Its
  fixed embedding-output VJP is an orthogonal local graph witness that holds
  discrete routing and top-k choices fixed. Delta and gradient magnitudes are
  diagnostics, not effect-size or quality estimates. This is not evidence of
  learned recall, language quality, pretrained-checkpoint behavior, efficiency,
  optimized kernels, or general long-context performance, and it does not
  isolate CSA and HCA individually.
- The checked-in attention control is one matched training seed. Its descriptive
  loss interval crosses zero, so it does not establish quality separation or
  training-seed stability. The measured throughput is limited to eager
  128-token runs on one GB10 and is not a long-context or optimized-kernel
  claim.
- The differential parity fixture validates eager floating-point equations on a
  tiny model, plus official-dimension main/YaRN rotary vectors through position
  1,048,575. Transformers supplies the rotary/full/cache/backward oracle; the
  MTP wrapper is independently composed from the pinned official inference
  source around a Transformers decoder block. This does not execute a
  million-token sequence or validate official FP4/FP8 kernels, expert
  parallelism, long-context throughput, or pretrained-model quality.
- Report security issues through the private process in `SECURITY.md`.
