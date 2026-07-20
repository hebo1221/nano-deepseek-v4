# P2 direct-controller v1.3.4 runtime-module provenance amendment

Date: 2026-07-20

Status: prospective, quality-blind amendment after the signed v1.3.3
post-activation ready-only-preflight launch failure. No v1.3.3 held-out quality
input, prediction, outcome, aggregate, shard, or matrix result was observed.

Required reporting label:
`quality-blind-v1.3.4-runtime-module-provenance-amendment`.

## Immutable v1.3.3 boundary

The v1.3.3 implementation was frozen at commit
`af26a021dc99c54d48fe0c0e0ea7cfdea615e44a`, tree
`a141844f19c8ca796796db6c7caf766f4045f914`, and implementation digest
`5c61e93bd259aec342ae26dac94447fc5ef40985df7b4af86c00b10e9d4548a0`.
The manifest-only result commit was
`6a4f936a85ad837c7fa0907a23747bb8a494fa96`, tree
`d0a040e02661ec0b1f51920df3353a539c250040`. The 81,675-byte manifest has
SHA256 `b7024052085aee9ab5741575d7968d9d4a758cfeab1d2e240161e5087a68e495`
and Git blob `e28e567c854d46399bc2c2f7ed891a9c93bf52f2`.

The experiment-owned durable namespace contains:

- an exact-two admission root: admission SHA256
  `32eb171fd5131d27a54683c7b917dd4d3f9cec635d65d11d331cb4950caea186`
  (59,688 bytes) and genesis SHA256
  `b276ebb6eb811935a7a1b677518f8e889ea82795cd189fbda051b247dc39dee0`
  (59,252 bytes);
- an exact-two activation root: the persistent empty `matrix.lock` and signed
  activation SHA256
  `adf07b131ae7dcdcafe106a31bea6e202ebd81c89b2cbd78b5a0761318e8fa4c`
  (295,752 bytes), with signed status `activated`;
- one persistent-session ledger pair for nonce
  `67520e50e2778b4650966938efc2fa6f3b8440819bbd6f4578e799c082ed9d56`:
  launch SHA256
  `339e7c7c3f533de09ff8d2a0758dd48b9e041160ca11d0b9b4f7619449c13ed6`
  (22,020 bytes) and terminal SHA256
  `2513be78bd9ec8a01970a6fdc1f6ebe32594c4ea2f894de520d26c78c9f8c95d`
  (19,744 bytes), plus its persistent empty sibling ledger lock.

All five JSON artifacts above are canonical pretty JSON and their HMACs
validate under key ID
`67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf`.
Shared scheduler, activation-bootstrap, and physical-device lock contents are
mutable cross-run coordination state and are not part of the normalized
historical lineage.

The signed activation has `completed_shards=0`, an empty record list, false
quality-start and evaluation-RNG flags, and zero materialized inputs,
predictions, outcomes, aggregates, active claims, worker ledgers, and top-p
quality inputs. The session plan is role `ready_only_preflight`, contains only
the first registered coordinate, has completed-prefix count zero and
`max_new_cells_stop_limit=1`, and permits at most one checkpoint-model load
attempt. The signed terminal has status `launch_failure`, child return code 1,
null ready and final receipts, empty completed-work and completed-result lists,
and zero bundle reingestions. Consequently there was no successful observed
model load and no quality-session launch.

The v1.3.3 quality output root, matrix summary, worker ledger root, integrity
artifact, summary artifact, cell claim, shard envelope, admission/activation
staging directory, and launch-only orphan are absent. The activation boundary
was crossed; the quality boundary was not.

## Source-bound diagnosis

The signed terminal authenticates `launch_failure`, return code 1, and the
absence of ready/final/work/result evidence. It does not contain an exception,
stderr, or traceback field. The following diagnosis is therefore bound to the
frozen source plus captured stderr, not presented as a signed terminal reason.

During `_assert_repository_import_origins`, the evaluator first encountered:

`FileNotFoundError: [Errno 2] No such file or directory: '/home/hebo1221/nano-deepseek-v4/_ops.py'`

It then raised:

`ValueError: Imported module origin is not exact: torch.ops -> /home/hebo1221/nano-deepseek-v4/_ops.py`

The parent subsequently raised
`ValueError: Persistent evaluator exited before its ready receipt.` through
`_start_persistent_evaluator` and `_ensure_ready_only_preflight`.

In PyTorch 2.13.0+cu130, `torch.ops` and `torch.classes` are `ModuleType`
virtual namespaces. Their instance dictionaries contain no `__file__`, their
own `__spec__` is `None`, and their classes expose compatibility sentinels
`_ops.py` and `_classes.py`. The v1.3.3 audit used dynamic attribute lookup,
mistook the inherited class sentinel for an instance filesystem-provenance
claim, and attempted a current-working-directory-relative strict resolution.

## Prospective v1.3.4 change

V1.3.4 determines filesystem provenance structurally, without a module-name
allowlist:

1. An instance-owned `__file__` is a filesystem provenance claim.
2. An instance-owned `__spec__.origin` is a filesystem provenance claim only
   when the own `ModuleSpec` has `has_location=True`.
3. If both claims exist, their lexical and resolved identities must agree.
4. Every filesystem claim retains the existing strict resolution,
   site-packages escape, and repository frozen-inventory checks.
5. A module with neither claim is classified as a virtual or namespace module
   without a source-backed import claim. Class-level compatibility attributes
   do not create an instance provenance claim.
6. Malformed instance-owned `__file__` or `__spec__` metadata is rejected
   rather than treated as an originless virtual module.
7. A non-module `sys.modules` value is accepted only as a structurally named
   class namespace whose metaclass is anchored in a separately audited owner
   module and whose object identity is statically resolved from that owner's
   namespace; arbitrary or merely look-alike registry objects are rejected.
   `None` remains a non-executing negative import-cache entry.

This amendment does not weaken validation for any module that claims a source
file. It changes only how source-backed claims are identified before the
existing fail-closed checks run.

Historical-lineage revalidation is scoped explicitly. The full v1.3.3 lineage
is reloaded before admission or activation staging recovery/publication,
activation or static-capability handoff, and authenticated quality-session
entry before its first write. Within one authenticated session, the signed
lineage bindings and retained activation lease carry authority; the historical
files are not rehashed before every individual shard or ledger write. This is
the implemented boundary and no stronger per-write claim is made.

The canonical v1.3.3/C6 entrypoint is retired. Out-of-band execution or
reattestation by a holder of the shared historical HMAC key is outside this
protocol's threat model; v1.3.4 detects historical byte or lineage drift but
does not claim to defend against a malicious trust-root holder.

V1.3.4 uses new admission, activation, output, worker, persistent-session,
integrity, summary, experiment-ID, attestation-purpose, and launcher
namespaces. The v1.3.3 state is read-only lineage and is never relabeled,
overwritten, resumed as v1.3.4, or used as a v1.3.4 quality artifact.

The registered coordinates, evaluation seeds, 17-arm order and semantics,
estimands, statistical analysis, success gate, one-cell outcome-independent
operational prefix, and mandatory outcome-independent full resume are byte-
canonical continuations of v1.3.3. No evaluation seed initialized a quality
RNG before this amendment.
