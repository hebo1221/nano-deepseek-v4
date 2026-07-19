# Adaptive V4 Memory reproduction guide

This runbook reproduces the paper-grade evidence in dependency order. Run every
command from the repository root on a clean commit. The runners are resume-safe:
they validate existing shard digests before reuse and write progress through an
atomic temporary-file replacement. Never delete a terminal failure artifact or
repair a held-out result post hoc.

## Environment and frozen inputs

Create the project environment and install the separately frozen natural-suite
dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -r research/adaptive_v4_memory/requirements-p3-ruler.txt
.venv/bin/pip install -r research/adaptive_v4_memory/requirements-p3-natural.txt
```

All seeds, scales, workload counts, benchmark revisions, model snapshots, data
digests, and system cells come from `manifests/paper-grade-study-v1.json` and the
phase manifests. Do not substitute a newer dataset or checkpoint under the same
artifact path.

## P2 synthetic core and causal factorial

Run both scales, audit the complete 4,500-shard core, freeze causal prerequisites,
then execute and summarize the 9,000-shard factorial:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_core_parallel.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_core_parallel.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.strict.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_causal_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_causal_parallel.py --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_causal_factorial.py \
  --matrix artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json \
  --output artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal_prerequisites.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py --scale s55 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py --scale s151 --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_seed_extension_causal.py
```

The separately preregistered four-seed extension starts only after the immutable
4,500-shard primary cohort passes its strict audit. It produces an independently
auditable 3,600-shard matrix, then permits the 8,100-shard nine-seed confirmatory
core summary only when frozen contracts match and seed namespaces are disjoint.
After the independently audited 9,000-shard primary causal matrix, it repeats
physical matching and all-arm equivalence for every extension checkpoint, runs
7,200 causal shards, and permits the 16,200-shard combined causal summary under
the same strict pooling rule. The
parallel runners first require exact serial/parallel probes. P2 causal also
requires calibration-only physical-memory matching and exact chunked-quality versus
sequential-tier equivalence for every seed, scale, budget, family, context, and arm.

## P2 prospective direct-controller cohort

The version-2.5 cohort is separate from the legacy factorial above. It uses the
checked-in `p2-post-rank-direct-controller-v1-2.json` amendment manifest, fresh 6071406--
6071410/7071406--7071410/10071406--10071410 seed namespaces, 19 arms, 9,000
budget shards, and exactly 154,280,000 decode-token rows when no technical
failure occurs. Run only from the clean implementation commit bound by that
manifest.

Revision 1.2 must reuse the already-provisioned cohort trust root outside the
repository and every artifact root. Never regenerate, overwrite, move, or copy
its bytes into a command, log, manifest, or artifact; changing it would make the
terminal revision 1.1 training ledger and quarantined evidence unverifiable.
Verify and export the existing file only:

```bash
test -s "$HOME/.adaptive-v4/direct-controller-attestation.key"
test "$(stat -c '%a' "$HOME/.adaptive-v4/direct-controller-attestation.key")" = 600
export ADAPTIVE_V4_ATTESTATION_KEY_PATH="$HOME/.adaptive-v4/direct-controller-attestation.key"
test -z "$(git status --porcelain)"
```

Execute the dependency chain in order. The amended training ledger contains one
exact-byte admission plus nine new cells; calibration still contains ten
seed-scale cells and top-p physical matching contains 40 cells. The original
v1 manifest, empty-prefix ledger, claim, summary, and checkpoint remain at their
original paths and must not be edited, moved, deleted, or retried. On the first
v1.1 invocation the training runner performs the HMAC-attested admission with
zero scientific or experimental subprocesses defined in
[`2026-07-19-p2-direct-training-validator-amendment.md`](reports/2026-07-19-p2-direct-training-validator-amendment.md),
using only frozen read-only `git ls-tree` and `git merge-base --is-ancestor`
provenance probes inside the admission-creation subroutine. Current
manifest/source preflight outside that subroutine may use other frozen
read-only Git probes. The runner atomically commits the admission, reloads and
validates it together with the amended ledger, then launches only the remaining
nine training coordinates.
The failed revision 1.1 `calibration/` root is likewise immutable, but its
observed `GO` artifact is not admitted. The first revision 1.2 calibration
invocation creates a signed zero-scientific-subprocess retry admission in the
new `calibration-v1-2/` root, validates an empty amended ledger, and then runs
the complete ten-cell grid. Only its first coordinate is the registered retry.
Exit code 2 from a terminal calibration, physical-match, or controller artifact
means a recorded NO-GO, not permission to delete and rerun it.
Controller exit code 3 reports a fully validated `draining_infrastructure` or
`paused_infrastructure` state. Non-owner workers and the coordinator may observe
and return that state but may not mutate it; only its bound owner can rerun the
headroom check after capacity changes.
Controller exit code 3 is a retryable, HMAC-attested infrastructure pause or
in-flight drain; it is nonterminal evidence and never a quality-gate pass.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_direct_calibration_matrix.py \
  --gpu-lock-path /tmp/adaptive-v4-direct-gpu0.lock
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_direct_top_p_physical_matrix.py \
  --gpu-lock-path /tmp/adaptive-v4-direct-gpu0.lock
.venv/bin/python research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix.py \
  --gpu-lock-path /tmp/adaptive-v4-direct-gpu0.lock
.venv/bin/python research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p2_direct_controller.py
```

Revision 1.2 does not rerun training. The calibration preflight authenticates the
already-terminal revision 1.1 ten-cell training ledger, summaries, and checkpoints
under their historical manifest/commit-tree context while separately requiring the
clean current v1.2 result context. All three single-GPU runners use the scheduler
path already authenticated by the failed attempt,
`/tmp/adaptive-v4-direct-gpu0.lock`; the one-shot calibration retry rejects every
other path before creating its admission.

Training and calibration freeze one exact CUDA execution environment before
their first cell and require byte-for-byte semantic equality for every child
artifact and exact resume. The binding includes the Python executable and
version, PyTorch/CUDA runtime and driver, platform, `CUDA_VISIBLE_DEVICES`, the
complete visible-device inventory, current device index, UUID/PCI routing,
compute capability, and memory size; elapsed time and peak-memory counters are
validated runtime measurements but are deliberately outside the stable binding.
Their sealed script bootstraps use Python isolated mode (`-I`), transport the
attestation key only through a sealed descriptor, and pass both already-held
GPU descriptors into each child: the user-selected scheduler lease and a
canonical physical-device guard derived from the selected UUID (PCI fallback).
A child therefore keeps both kernel leases if its runner dies; no child acquires
a nested lease and a different scheduler path cannot bypass physical exclusion.

The prospective training/calibration summaries, artifacts, and matrix ledgers
use exact schemas. Their output roots are closed-world inventories on preflight,
terminal validation, and downstream consumption: an extra, missing, renamed,
symlinked, or orphaned file fails closed even when a supplied JSON payload has a
valid MAC. Preserve each canonical matrix ledger with its registered cell files
when archiving. Controller-compatible multi-GPU comparison is narrower: it
compares the GPU class selected by `current_device_index` and may ignore routing
topology. The selected logical index is nevertheless frozen into the local
execution binding, forwarded as an explicit `cuda:N` child argument, and
checked against the child's UUID/PCI evidence; execution never silently falls
back to visible device zero.

Calibration and top-p physical matching create a persistent exact-coordinate
claim before each child launch. They remove it only after the newly published
terminal artifact, exit code, provenance, and frozen execution environment have
all validated, immediately before authoritative ledger promotion. An exception,
unexpected exit, or missing/invalid artifact preserves the claim as fail-closed
evidence; a crash after claim release but before ledger commit instead leaves an
orphan artifact. Both states block resume pending an explicit recorded operator
quarantine, because no whole-process retry policy was frozen.

Revision 1.2 records one narrow exception after the first direct calibration
child exposed a parent-only checkpoint-path spelling bug. The complete failed
`calibration/` root, including its 0/10 ledger, preserved claim, and observed
terminal `GO` artifact, remains byte-exact quarantine at its original path and
is never adopted into the amended cohort. A signed admission under the matrix,
scheduler, and physical-device leases authorizes exactly one result-disclosed,
cause-based retry of S55/6071406 in the disjoint `calibration-v1-2/` root. This is
not presented as counterfactual outcome-independent authorization. Admission creation
starts zero scientific subprocesses and consumes the exception at commit: only
the same creating process may launch the first child, so a restart before first
promotion is terminal fail-closed rather than permission to retry. The corrected parent compares the exact
authenticated relative path spelling while checking its resolved regular-file
referent, digest, and size separately. No other failed calibration, top-p, or
quality coordinate gains a retry from this amendment. See the checked
[`2026-07-19-p2-direct-calibration-path-binding-amendment.md`](reports/2026-07-19-p2-direct-calibration-path-binding-amendment.md).

All read-only checks of the frozen GPU scheduler path, exact CUDA execution
environment, manifest, training prerequisite, quarantine inventory, and incident
report complete before the final admission link can commit and consume the
exception. A staging-only interrupted creation may be completed by the process
that commits and validates the final admission; a crash during or after that
commit, including before the empty-prefix ledger or first promotion is durable,
is an availability-terminal fail-closed event. Owner-controlled deletion,
filesystem snapshot rollback, or restoration of an earlier amended root is
outside the process-level threat model and invalidates the evidence; it is never
a registered way to replay the retry.

The controller runner computes an observed component-wise high-water planning
estimate from fixed sidecar bytes, actual envelope bytes, and bytes per
decode-token row against the exact remaining token weights. Before each launch
it requires at least the frozen one-shard headroom and, once usable observations
exist, the larger observed next-shard estimate. This is neither a future upper
bound nor a space reservation: unseen compression behavior and concurrent
writers can still exhaust the filesystem. Insufficient headroom raises a typed,
retryable infrastructure pause before a cell claim or evaluator launch. The
runner publishes that condition directly as `paused_infrastructure` only when
no distributed claim is live. Otherwise it first publishes an HMAC-attested
`draining_infrastructure` state: no new claim may start, but already-running
claims may commit without becoming orphans. Each such commit re-attests the
latest sparse ledger, storage evidence, live-claim inventory, and `statvfs`;
the last commit converts an insufficient drain to `paused_infrastructure`.
Only the originally bound worker may rerun the frozen check and publish the
`in_progress` transition before claiming the exact next coordinate. Neither
blocked state is a terminal pass or summarizable evidence. Raw outcome, token,
and failure sidecars are streamed once into the summary; the integrity audit
and terminal summary retain HMAC bindings to every raw bundle.

An evaluator-caught arm error is already a terminal technical-failure bundle
and remains in the intent-to-treat cohort. By contrast, an unexpected child
exit with no terminal envelope is not silently retried: its coordinate claim
is preserved as fail-closed orphan evidence and blocks resume until an operator
performs and records a manual quarantine decision. This manifest did not freeze
a bounded whole-process retry policy, so deleting that claim and rerunning the
coordinate would create an unregistered selection path.

The training, calibration, and top-p physical-match matrices use the same
no-silent-retry boundary. Each coordinate acquires an exclusive persistent
claim before its child starts; only an expected exit with a fully validated
terminal artifact may complete the claim/ledger transaction. An unexpected
exit or missing artifact preserves the claim, and subsequent preflight rejects
the coordinate until manual quarantine is recorded. A signed terminal NO-GO is
evidence to commit, not a reason to delete or rerun the cell.

Distributed mode currently supports one host with a shared local filesystem.
Each worker receives its modulo partition and a distinct scheduler-lease path;
the runner additionally acquires the canonical guard for the GPU actually
selected by that process. A coordinator-only invocation publishes the canonical
terminal merge only after every worker ledger is terminal. The same GPU model,
compute capability, memory size, Python/PyTorch/CUDA runtime and driver are
required; UUID, PCI address, logical index, and `CUDA_VISIBLE_DEVICES` may
differ. On a four-GPU single host, launch worker index `N` with
`CUDA_VISIBLE_DEVICES=N`, `--worker-count 4 --worker-index N`, and
`--gpu-lock-path /tmp/adaptive-v4-gpu-N.lock`, then run one coordinator process
with `--worker-count 4 --coordinator-only`. Merely changing the scheduler path
without selecting a different physical GPU fails on the canonical guard.
Multi-host PID/claim coordination is not implemented and must not be inferred
from this command.
The public matrix also binds the closed-world sibling directory
`.<output-root-name>.p2-direct-controller-workers`, including its canonical
worker filenames and each file's size, SHA-256, payload digest, attestation,
terminal status, assigned coordinate count, and absolute root path. Preserve
that directory beside the output root at its original absolute path for
authoritative replay. A relocated copy is archival backup only and does not
become valid evidence merely by retaining the sibling layout; a moved, missing,
extra, renamed, or mutated ledger makes replay fail closed. A single-worker run
requires this sibling directory to be absent.

## P3 compatible-model natural and safety evidence

Acquire only the revisions pinned in the P3 manifests, prepare their source/data
inventories, and verify the snapshot before inference. The natural runner commands
share the same two frozen roots:

Before the P2 sequence gate, the preregistered exception permits only immutable
model and public-code prefetch plus cryptographic verification. The following
command clones and verifies the pinned SCBench, LongBench v2, and LongMemEval code,
writes `p3-natural-source-prefetch-v1`, and explicitly records that it acquired no
benchmark dataset, selected no baseline, generated no dataset, and ran no inference:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_sources.py \
  --prefetch-only
```

Do not run the default source-inventory mode, any dataset preparation, baseline
selection, or prediction command until the primary P2 matrix, its five-seed strict
core audit, the nine-seed confirmatory core audit, and both five- and nine-seed
causal audits all pass the frozen sequence gate. The common executable gate reopens
and hashes all five dependencies, including both core audits' raw-matrix bindings,
so direct runner invocation cannot bypass this order. After that gate, the default source
command re-verifies every prefetched checkout and binds the prefetch inventory
digest into the final source inventory.

```bash
export KVPRESS_ROOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources/kvpress
export RULER_ROOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources/RULER
export MODEL_SNAPSHOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/models/cdbee75f17c01a7cc42f958dc650907174af0554
mkdir -p artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources
git init "$RULER_ROOT"
git -C "$RULER_ROOT" remote add origin https://github.com/hsiehjackson/RULER.git
git -C "$RULER_ROOT" fetch --depth 1 origin 38da79d79519ef87aa46ae804f838e1eab7f86d7
git -C "$RULER_ROOT" checkout --detach FETCH_HEAD
git init "$KVPRESS_ROOT"
git -C "$KVPRESS_ROOT" remote add origin https://github.com/NVIDIA/kvpress.git
git -C "$KVPRESS_ROOT" fetch --depth 1 origin 6d965557a5b9f0201a2301b23c454473dd681d0d
git -C "$KVPRESS_ROOT" checkout --detach FETCH_HEAD
.venv/bin/hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_sources.py
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_datasets.py
.venv/bin/python research/adaptive_v4_memory/scripts/verify_p3_natural_model.py
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_ruler_dataset.py \
  --ruler-root "$RULER_ROOT" --tokenizer-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_scbench.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longmemeval.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --judge-mode blocked
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_mrcr.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
```

After the terminal nine-seed P2 causal audit, fixed scorer selection, and all five
Qwen3-4B RULER datasets exist, run the separately preregistered same-global-budget
compatibility cohort. It compares `fixed+pins` with `natural-adaptive-quota+pins`;
it is not an unchanged port of the synthetic controller.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_ruler.py
```

After both the adaptive RULER audit and baseline SCBench audit are terminal, run the
separately frozen shared-context replication. Adaptive allocation occurs only during
the initial shared-context prefill; later turn tokens are appended identically.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_scbench_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_scbench.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_scbench.py
```

After the adaptive RULER audit and baseline LongBench-v2 audit are terminal, run the
separately frozen one-pass reasoning replication. Adaptive allocation applies only to
the document-context prefill; the question is appended identically under both arms.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_longbench_v2_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_longbench_v2.py
```

After the adaptive RULER audit and baseline LongMemEval audit are terminal, run the
separately frozen judge-blocked generation cohort. It retains all raw responses and
physical quota audits, but emits no quality, non-inferiority, or negative-result claim.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_longmemeval_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longmemeval.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT" --judge-mode blocked
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_longmemeval.py
```

After the adaptive RULER audit and baseline MRCR audit are terminal, run the
separately frozen multi-needle replication. Adaptive allocation applies only to
the long context prefill; the final query is appended identically under both arms.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_mrcr_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_mrcr.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_mrcr.py
```

After all four adaptive Qwen summaries are terminal, run the separately frozen
cross-benchmark audit. It preserves each benchmark's interval and multiplicity
correction; it never pools heterogeneous scores or p-values. LongMemEval remains
outside this adaptive suite while its paid official judge is unavailable.

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_natural_adaptive_quota_suite_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_adaptive_quota_suite.py
```

Run the separately reported Phi-4-mini cross-family transfer only after the
nine-seed P2 causal audit and Qwen fixed-baseline selection are terminal:

```bash
export PHI_SNAPSHOT=artifacts/adaptive_v4_memory/paper_grade/p3/assets/models/cfbefacb99257ffa30c83adab238a50856ac3083
.venv/bin/hf download microsoft/Phi-4-mini-instruct \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --local-dir "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/verify_p3_natural_model.py \
  --manifest research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json \
  --snapshot "$PHI_SNAPSHOT" \
  --output artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-model-verification.json \
  --experiment-id p3-cross-family-model-snapshot-verification-v1
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_cross_family_ruler_dataset.py \
  --ruler-root "$RULER_ROOT" --tokenizer-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_ruler.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_cross_family_adaptive_quota_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_cross_family_ruler.py --cohort adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_adaptive_quota_ruler.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p3_cross_family_adaptive_quota_longbench_v2_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py --cohort cross-family-adaptive-quota \
  --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$PHI_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_cross_family_adaptive_quota_longbench_v2.py
```

Summarize each benchmark with `summarize_p3_natural_benchmark.py`, binding the
`native-dense` and `strongest-memory-matched-fixed` cell artifacts, then run:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/prepare_p3_natural_safety_assets.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p3_safety_stress.py --kvpress-root "$KVPRESS_ROOT" --model-snapshot "$MODEL_SNAPSHOT"
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_safety_stress.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p3_natural_suite.py
```

Operational generation failures score zero. LongSafety generation may be complete
while its paid official judge remains blocked; that state must remain `unverified`
for comparative LongSafety safety claims.

## P4 systems evidence

Run the feasibility probe, serial-interleaved reference matrix, and actual-concurrency
adapter matrix separately:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_500k_context_preflight.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_500k_context_preflight.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_systems_matrix.py \
  --output artifacts/adaptive_v4_memory/paper_grade/p4/reference-systems.summary.json
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_adaptive_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_adaptive_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/validate_p4_adaptive_production_manifest.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_adaptive_production_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_adaptive_production_systems_matrix.py
.venv/bin/python research/adaptive_v4_memory/scripts/run_p4_production_systems_matrix.py \
  --adapter-executable research/adaptive_v4_memory/scripts/p4_continuous_batch_adapter.py
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p4_production_systems_matrix.py
```

Reference interleaving is never relabeled as concurrency. Partial and failed cells
remain terminal observations in the tables and tail-failure counts.

## Online learned lookahead and final package

After P2-P4 are terminal, run the distinct online learned controller and generate
the digest-bound package:

```bash
.venv/bin/python research/adaptive_v4_memory/scripts/run_p1_online_lookahead_parallel.py --workers 3
.venv/bin/python research/adaptive_v4_memory/scripts/summarize_p1_online_learned_lookahead.py
.venv/bin/python research/adaptive_v4_memory/scripts/build_p5_paper_package.py
.venv/bin/ruff check nano_deepseek_v4 research/adaptive_v4_memory/scripts tests
.venv/bin/mypy nano_deepseek_v4 research/adaptive_v4_memory/scripts
.venv/bin/pytest -q
.venv/bin/python -m build
.venv/bin/twine check dist/*
.venv/bin/python research/adaptive_v4_memory/scripts/run_p5_release_gate.py
```

The final package is valid only from a clean source tree. Check
`artifacts/adaptive_v4_memory/paper_grade/p5/artifact-index.json` for every input
and generated SHA-256. During generation, every declared `path`/`sha256` pair in the
P3 evidence graph is reopened and rehashed, including bindings reached through linked
JSON cell artifacts; a missing half-binding, missing file, or nested digest drift fails
closed. GitHub Actions remains disabled by user request and must not be
reported as passed. It is outside the completion gate; the five digest-bound local release
checks above are the final source-verification contract. The release-gate artifact also
requires `HEAD` to have an `origin/*` upstream with zero locally tracked ahead/behind
commits. This is a push-synchronization check against the local tracking ref; it does not
fetch, establish network freshness, open a PR, or imply CI success.

## Official DeepSeek-V4 boundary

Compatible Qwen evidence is not official DeepSeek-V4 evidence. Until the exact
serving checkpoint, conversion contract, accelerator topology, and storage/HBM
requirements in `manifests/p3-flashmemory-deepseek-v4-v1.json` are satisfied, keep
that result blocked and use the frozen reproduction command and cost/resource
contract in that manifest. Never substitute a compatible model or the static local
adapter for official FlashMemory or an external fused multi-GPU production runtime.
The static adapter also cannot verify non-contiguous residency-layout costs,
position-aware cache-miss recomputation, or a fused attention-kernel cost model;
those boundaries are machine-readable in the P4 blocker and final P5 audit.
