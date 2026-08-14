# Conformance profiles

`nano-deepseek-v4 conformance` runs a fixed set of checks and can emit one
machine-readable receipt. It is the shortest way to answer two different
questions without mixing them together:

- whether the native tiny implementation satisfies its local execution
  contracts;
- whether the selected independent references still agree with those
  contracts.

The command is CPU-runnable. It does not load official model weights.

## Quick run

The default `core` profile is local and requires no optional dependencies or
network access. Install the current source with the
[PyTorch-first installation guide](../guides/installation.md), then run:

```bash
nano-deepseek-v4 conformance --profile core
```

`python -m nano_deepseek_v4 conformance --profile core` is equivalent.

Write the same result as JSON for CI or an issue report:

```bash
nano-deepseek-v4 conformance \
  --profile core \
  --json \
  --output conformance-core.json
```

`--json` prints the receipt. `--output` writes it atomically, whether or not
`--json` is present. When both are used, stdout and the file contain the same
JSON bytes.

## Fixed profiles

`core` → `offline` → `live` is cumulative. `dspark` is a deliberate offline
fork from `core`, so checking the draft equations does not require installing
the independent Transformers package. A selected check is always required;
there is no flag that silently turns a failed or unavailable required check
into a pass.

| Profile | Required checks | Network |
| --- | --- | --- |
| `core` | Native tiny full/cache execution, packaged Flash-0731 architecture consistency, attention-path reachability | Forbidden |
| `dspark` | Everything in `core`, plus separate fixed DSpark draft-equation and prefix-scheduler vectors checked against independent oracles | Forbidden |
| `offline` | Everything in `core`, plus differential parity against exactly Transformers 5.15.0 | Forbidden |
| `live` | Everything in `offline`, plus a pinned Flash-0731 Hub metadata and selected-header replay | Explicitly allowed for the pinned replay |

For `core`/`offline`/`live`, the canonical order remains
`tiny_model_cache`, `flash_0731_architecture`, `attention_reach`,
`transformers_parity`, then `flash_0731_receipt`; those reports always list the
same five entries. The `dspark` order inserts `dspark_vectors` and
`dspark_scheduler` after `attention_reach`, then lists the two unselected
reference/live checks, for seven entries total. Checks outside a selected
profile appear as optional `skip` entries with
`reason_code=not_selected_by_profile`; they are not hidden.

### Core

`core` is the no-download check suitable for a clean CPU installation. It:

1. runs a deterministic random-tiny model through full and cached execution;
2. compares the packaged Flash-0731 preset with the bundled pinned config and
   inspects the DSpark stage layout without allocating a model;
3. runs the fixed CPU attention-reachability fixture.

These checks cover native execution and structural contracts. Random tiny
weights do not measure language quality or official-checkpoint behavior.

### DSpark

`dspark` adds two CPU-only, no-download checks to `core`:

```bash
nano-deepseek-v4 conformance --profile dspark --json \
  --output conformance-dspark.json
```

The `dspark_vectors` receipt compares fixed packaged vectors with two separately
written implementations. It checks ordered target-state projection, non-causal
attention across all five draft positions in each of three stages, sequential
previous-token Markov bias, exact greedy proposals, and sigmoid confidence.
The focused `nano-deepseek-v4 dspark` command runs only that check. See
[DSpark semantic vectors](dspark.md) for the field-level contract, immutable
source revision, deliberate mutation tests, and excluded claims.

The separate `dspark_scheduler` receipt checks Algorithm 1's strict causal
greedy trace and Section 5.2's caller-supplied lagged global-capacity rule
against an exhaustive oracle and fixed synthetic SPS tables. Its focused
command is:

```bash
nano-deepseek-v4 dspark-scheduler \
  --json \
  --output dspark-scheduler.json
```

This is CPU arithmetic on supplied calibrated probabilities, not evidence for
temporal provenance or request-slot alignment, STS quality, an actual hardware
profile, rejection sampling, end-to-end losslessness, speed, serving, model
quality, or official-weight runtime. See
[DSpark prefix scheduler](dspark-scheduler.md) for the input schemas, tie rule,
receipt fields, primary paper sections, and exact claim boundary.

### Offline

Install the pinned reference before selecting `offline`:

```bash
python -m pip install -e ".[parity]"
nano-deepseek-v4 conformance --profile offline
```

The parity dependency is part of the oracle, so the required version is exactly
Transformers 5.15.0. A missing or different version makes the suite
`INCOMPLETE`; it is not treated as a passing skip. The profile performs no
network requests.

### Live

`live` is the only profile permitted to access the network. Selecting it is
explicit consent to replay the pinned Hub receipt:

```bash
python -m pip install -e ".[parity,official]"
nano-deepseek-v4 conformance --profile live --output conformance-live.json
```

The replay pins the Hub revision and hashes the allowlisted config, index, and
source metadata. It requests bounded header ranges from three selected shard
URLs and leaves zero cached `.safetensors` files in its dedicated temporary
cache. It does not materialize tensors or verify payload integrity, and it does
not guarantee that zero payload bytes traverse the network. It also does not
execute DSpark or import downloaded inference source.

## Network enforcement

Every `core`, `dspark`, and `offline` check runs with the Hugging Face offline
variables set and a fail-closed Python standard-library socket guard active. A
blocked IP socket or DNS attempt changes a required check to `ERROR`, even if
the code that made the attempt catches the exception. The receipt records the
total as `blocked_attempt_count`.

That in-process guard is deliberately described as
`best_effort_python_stdlib_calls`. It is not a Python sandbox: a bound socket
method captured before the guard, a native extension, or a child process can
bypass Python method interception. Consequently a local receipt with
`os_isolation=none` proves that the instrumented path observed no forbidden
standard-library attempt; it does not prove that the whole process tree lacked
network access.

CI and the tag workflow create their authoritative `core` and `offline`
receipts inside a fresh Linux network namespace with only loopback present.
They drop supplementary groups, effective and bounding capabilities, and set
`no_new_privs` before executing the installed command. The command also passes
`--require-os-network-isolation`, so a missing or malformed namespace fails
before any check runs. Those receipts report
`enforcement_scope=linux_network_namespace_process_tree` and
`os_isolation=linux_network_namespace`.

The `live` profile cannot use that namespace because its pinned replay is
explicitly allowed to contact the Hub. It keeps the Python guard around the
four offline checks, removes it before the live verifier, and records the
verifier's network-attempt callback separately in the aggregate `attempted`
field. Hugging Face caches offline and telemetry flags when its modules are
imported, and Hub 0.x can also cache a session configured with offline-only
adapters. The guard synchronizes already-imported flags and discards an
already-created Hub HTTP session both when it enters and when it exits. This
preserves inherited user settings while ensuring that a cumulative live run
does not remain accidentally offline after its parity check.

The parity boundary also snapshots and restores Python, NumPy, and Torch CPU
RNG state. It supplies a dedicated temporary `TORCHINDUCTOR_CACHE_DIR` while
optional dependencies are imported, removes that directory afterward, and
restores any caller-provided cache setting before the live check begins.

## Receipt contract

JSON receipts use sorted keys and contain no timestamps or durations. A `core`
run is deterministic for the same package, implementation, dependency versions,
platform, and machine architecture. The top level records:

- `profile`, `status`, `passed`, and `complete`;
- package, Python, PyTorch, platform, and optional-dependency versions;
- the conformance implementation SHA-256;
- the network policy, whether a network attempt occurred, blocked-attempt
  count, enforcement mechanism and scope, and OS-isolation state;
- every check's mode, requirement status, reason code, claim boundary, nested
  receipt, and nested-receipt SHA-256;
- pass, fail, skip, error, and required-check counts.

`passed` is true only for a `PASS` suite. `complete` means every required check
reached a pass/fail decision; a required skip or harness error leaves it false.

The per-check and suite-level `claim_boundary` fields travel with the evidence.
Keep them when archiving or comparing receipts: a digest without its stated
scope is easy to overread.

## Release receipts

A tagged release publishes `core.json`, `dspark.json`, `offline.json`,
`live.json`, and the separate `cuda-release-gate.json` beside
`release-manifest.json` on GitHub. The manifest's `conformance_receipts` and
`cuda_receipts` maps bind every filename to its SHA-256 digest. These files are
release evidence, not Python distributions, so PyPI does not carry them.

To consume a receipt, download the evidence from the same tag, verify each
digest against its manifest map, and retain each conformance receipt's
`claim_boundary`. Then verify the GitHub attestation for the receipt and
manifest:

```bash
gh release download vX.Y.Z \
  --repo hebo1221/nano-deepseek-v4 \
  --pattern core.json \
  --pattern dspark.json \
  --pattern offline.json \
  --pattern live.json \
  --pattern cuda-release-gate.json \
  --pattern release-manifest.json
gh attestation verify release-manifest.json --repo hebo1221/nano-deepseek-v4
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("release-manifest.json").read_text())
for field in ("conformance_receipts", "cuda_receipts"):
    for name, expected in manifest[field].items():
        observed = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"receipt digest mismatch: {name}")
print("receipt digests: PASS")
PY
gh attestation verify core.json --repo hebo1221/nano-deepseek-v4
gh attestation verify cuda-release-gate.json --repo hebo1221/nano-deepseek-v4
```

Repeat attestation verification for the profile receipt you rely on. A valid
attestation identifies the published artifact; the manifest digest binds the
receipt bytes; the receipt's own boundary defines what those bytes establish.
For the CUDA receipt, GitHub attests the published JSON, while its source
inventory and tested-commit fields bind the manual run. This is not remote
attestation of the GPU host.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every required check in the selected profile passed. |
| `1` | A required check ran and failed its contract, including pinned receipt drift. |
| `2` | Command-line usage was invalid, such as an unknown profile or option. |
| `3` | The result is incomplete because a required dependency, pinned source, or inspection was unavailable. |
| `4` | The harness hit an unexpected internal error, could not serialize the report, or could not write `--output`. |

For report statuses `FAIL`, `INCOMPLETE`, and `ERROR`, JSON is still emitted or
written when `--json` or `--output` was requested and serialization or the
destination remains available. Automation should use the process exit code and
retain the receipt for the specific check status and reason code.

## Scope boundary

The suite covers random-tiny native execution and cache consistency, packaged
Flash-0731 structure, fixed-seed attention reachability, optional fixed DSpark
draft-equation and prefix-scheduler vectors, tiny eager floating-point parity
against pinned sources, and an optional pinned metadata/header replay.

It does not establish official-weight runtime parity, checkpoint payload
integrity, confidence calibration, a real SPS profile, DSpark acceptance or
end-to-end losslessness, optimized-kernel correctness, distributed or
long-context behavior, measured latency or throughput, training quality, model
quality, or production serving readiness. Use dedicated hardware and
pretrained-model evaluations for those claims.
