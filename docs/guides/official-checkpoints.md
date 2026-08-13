# Official checkpoint inspection

Official DeepSeek-V4 Flash, Pro, and Flash-0731 snapshots use an upstream tensor
layout that is different from this project's native bundle format. Flash-0731
also replaces the conventional auxiliary MTP layout with a three-stage DSpark
checkpoint dialect. This guide separates allocation-free architecture
inspection, remote namespace inspection, complete local verification, and
materialized loading.

## Inspect a preset without weights

```bash
nano-deepseek-v4-inspect --preset flash
nano-deepseek-v4-inspect --preset flash-0731 --json
nano-deepseek-v4-inspect --preset pro --json
```

The Flash report gives 290,942,289,362 logical parameters and 14,118,225,362
activated parameters. Pro gives 1,598,837,347,742 logical and 50,648,438,174
activated parameters. The report separates backbone and MTP attention
schedules, reports MTP depth and context length, and emits a canonical
configuration digest.

The Flash-0731 report preserves the official raw
`num_nextn_predict_layers=1` value and separately reports three DSpark stages,
19,845,850,215 speculative parameters, and 971,482,215 speculative activated
parameters. The 768 correction-bias values are reported separately as
non-parameter routing state. `runtime_load_supported=false` is part of the
report: inspecting this layout does not imply native speculative execution.

Native and official JSON files can be inspected with `--config` and
`--official-config`; a native bundle uses `--bundle`. Parameter counting follows
the official tensor schema and never constructs the model or reads checkpoint
payloads. Routing lookup tables are not counted as model parameters.

## Inspect one remote namespace

Follow the [installation guide](installation.md), install the Hub client extra,
and pin a revision:

```bash
python -m pip install -e ".[official]"
nano-deepseek-v4-inspect \
  --hf-repo deepseek-ai/DeepSeek-V4-Flash \
  --revision fd53f944496234770ba80e15004f9b6d269a71f5 \
  --namespace mtp \
  --json
```

For the newer DSpark checkpoint, use its immutable revision:

```bash
nano-deepseek-v4-inspect \
  --hf-repo deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --namespace mtp \
  --json
```

The command resolves a branch, tag, or commit to an immutable commit SHA before
reading checkpoint metadata. It downloads `config.json` and
`model.safetensors.index.json` as files, identifies only the shards assigned to
the requested namespace, and uses bounded range requests to parse their
safetensors headers. It does not intentionally fetch complete weight payloads
or materialize tensors, and the reproducible receipt verifier requires zero
cached `.safetensors` files at completion.

The Hub client's initial range response can include bytes immediately after a
small header, so this is not a claim of zero payload bytes on the wire.
`metadata_document_bytes` counts the downloaded config and index documents, not
all HTTP traffic.

On the pinned Flash revision, the MTP namespace uses shard 46 and contains 1,575
tensors. Its dtype/shape inventory digest is:

```text
331296a906dd0fdf85384f0f6237602bc49e2e19c285ca19b1946b62cb330561
```

That digest matches the independent local 159.6 GB snapshot audit. A clean-cache
dogfood run left no `.safetensors` weight files in the Hub cache.

On the pinned Flash-0731 revision, the same command recognizes `mtp` as a
DSpark namespace spanning shards 46–48. It checks 4,705 tensors with zero
missing, unexpected, unrecognized, unchecked, or shape-mismatched keys. The
stored namespace size is 10,862,838,300 bytes and its canonical inventory
digest is:

```text
e404f70fb47572253d4d3b3f2f1977611c4a65701bf16c81422303cbdbb4cf0b
```

The check also verifies a one-to-one scale sidecar for every low-precision
weight. FP8 scales must be `F8_E8M0` with the configured 128×128 block grid;
packed FP4 expert scales must be `F8_E8M0` with one value per 32 logical input
elements. `scale_metadata_verified=true` records that the dtype, shape, and
association checks all passed.

The checked-in
[`DeepSeek-V4-Flash-0731-metadata.json`](../../nano_deepseek_v4/_receipts/DeepSeek-V4-Flash-0731-metadata.json)
binds those counts to the pinned config and index digests. The probe downloaded
5,604,759 bytes of config/index documents and cached no `.safetensors` files.
Reproduce that receipt through the installed package:

```bash
nano-deepseek-v4-verify-flash-0731 --json
```

A wheel built from the current Unreleased source contains the command, verifier,
and canonical receipt/config resources. The verifier uses a dedicated empty
temporary cache, pins and hashes
`config.json`, the checkpoint index, and the two official inference source
files, then compares a fresh selected-header report with the receipt. The
downloaded Python is hashed but never imported or executed. A pass also
requires zero cached `.safetensors` files when the verifier finishes; it remains
a metadata-only result and does not claim zero payload bytes on the wire.
`python scripts/verify_flash_0731_receipt.py --json` remains available as a
source-checkout compatibility wrapper.

Automation should distinguish the verifier's four outcomes. Exit `0` is a
pass, exit `1` is semantic drift or pinned-source absence, exit `3` is a stable
`INCOMPLETE` result for a missing dependency or transient source/header
unavailability, and exit `4` is a local execution or harness error. JSON error
receipts expose stable reason codes or exception classes without echoing cache
paths, signed URLs, or raw transport messages. A timeout is therefore not
reported as receipt drift, and a local permission failure is not reported as a
missing remote file.

### Flash-0731 DSpark boundary

The 0731 config contains 43 backbone compression entries followed by three
zero-valued DSpark entries. Those entries are not ordinary MTP modules and do
not change the raw `num_nextn_predict_layers=1` field. The namespace has three
full attention/MoE stages, a stage-0 projection over target layers 40–42, and
final-stage Markov and confidence heads.

This project validates that schema and separately checks the tiny outer draft
equations with [fixed CPU semantic vectors](../verification/dspark.md). A
[separate CPU scheduler checker](../verification/dspark-scheduler.md) covers
Algorithm 1 and Section 5.2 arithmetic on supplied calibrated probabilities and
synthetic SPS tables. Neither check implements official-checkpoint DSpark
proposal execution, speculative verification, token acceptance, production
scheduler integration, or serving kernels. Constructing either
`DeepSeekV4Model` or `DeepSeekV4ForCausalLM` with the 0731 config raises
`NotImplementedError` before parameter allocation instead of silently building
the older MTP runtime.

### Remote result boundary

For a remote namespace, `is_complete=true` means that the selected namespace
agrees with the pinned config, index, and fetched shard headers.
`snapshot_preflight_complete=false` states that the other shard headers and all
payload contents were not checked. It is a scoped metadata result, not a full
checkpoint-integrity claim. `scale_metadata_verified`,
`payload_integrity_verified`, and `runtime_load_supported` must be checked
independently. For the Flash-0731 probe they are true, false, and false
respectively.

## Verify a downloaded snapshot

Download the official snapshot using the Hugging Face CLI or another trusted
transfer path:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ./checkpoints/flash
```

Inspect one local namespace without materializing tensors:

```bash
nano-deepseek-v4-inspect \
  --checkpoint ./checkpoints/flash \
  --namespace mtp \
  --json
```

The local namespace report binds each key to its safetensors dtype and shape in
one canonical inventory digest. On the pinned Flash snapshot, MTP contains 1,575
tensors in shard 46: 797 model/state tensors and 778 scale sidecars, with zero
missing, unrecognized, unchecked, or shape-mismatched keys.

Verify the entire local snapshot from Python:

```python
from nano_deepseek_v4 import verify_deepseek_checkpoint_snapshot

report = verify_deepseek_checkpoint_snapshot("./checkpoints/flash")
assert report.is_complete
print(report.total_shards, report.total_tensor_bytes)
```

The checked-in
[`DeepSeek-V4-Flash-validation.summary.json`](../../references/DeepSeek-V4-Flash-validation.summary.json)
records a complete 46-shard preflight and streaming payload scan, including
content digests, dtype/shape coverage, namespace evidence, and logical parameter
counts.

## Construct and load the native model

Full loading is intentionally explicit and memory-heavy:

```python
from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_official_checkpoint,
    verify_deepseek_checkpoint_snapshot,
)

snapshot = "./checkpoints/flash"

preflight = verify_deepseek_checkpoint_snapshot(snapshot)
assert preflight.is_complete

config = DeepSeekV4Config.from_official_json(f"{snapshot}/config.json")
model = DeepSeekV4ForCausalLM(config)
load_report = load_deepseek_official_checkpoint(model, snapshot)
print(len(load_report.conversion.converted_keys), "tensors loaded")
```

`load_deepseek_official_checkpoint` materializes official tensors and the
converted state dictionary; it is not a lazy or sharded serving loader. For a
snapshot too large to materialize, use
`build_deepseek_official_checkpoint_streaming_load_report` to scan tensor
payloads and emit conversion evidence without constructing a runnable model.

The loading example applies to checkpoint families whose native runtime is
implemented. Flash-0731's official-checkpoint runtime remains inspection-only
and fails before model allocation. Passing the tiny DSpark semantic or scheduler
vectors does not change that boundary; do not use the example as a 0731 serving
recipe.

## Support boundary

Header and payload verification establish integrity and conversion coverage for
the inspected files. They do not demonstrate official benchmark quality,
distributed execution, production latency, or parity with upstream FP4/FP8
kernels. Metadata compatibility also does not establish DSpark runtime parity.
The complete operational boundary is documented in
[`PRODUCTION_READINESS.md`](../../PRODUCTION_READINESS.md).
