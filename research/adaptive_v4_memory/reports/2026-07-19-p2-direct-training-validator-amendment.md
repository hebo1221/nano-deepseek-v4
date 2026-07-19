# P2 direct-training validator incident and pre-outcome amendment boundary

Date: 2026-07-19
Incident status: **v1 transaction incomplete; preserved for exact-byte review**
Execution decision: **no retry, deletion, move, or downstream launch is authorized by v1**

## Decision

The first frozen prospective prerequisite child, S55 training seed `6071406`,
completed all 1,000 global optimization steps and published an authenticated
summary and checkpoint. The parent runner then rejected the checkpoint because
its validator incorrectly required every parameter-local AdamW `step` counter
to equal the global step count. That predicate is invalid for sparsely routed
MoE expert parameters: PyTorch initializes and increments Adam state only when a
parameter has a gradient, and this model executes an expert only when the router
selects it.

This is classified as a post-freeze, pre-primary-outcome validation-specification
incident. It is not a low-quality result, a trainer crash, a signed prerequisite
NO-GO, or permission to rerun the coordinate. The current v1 transaction remains
incomplete with its original claim intact. A fresh statistical cohort is not
required if, and only if, a separately frozen manifest revision performs a
one-shot admission of the exact existing bytes under the requirements below.
The old v1 manifest and artifacts must never be relabeled as if the amendment
had existed when they were produced.

## Frozen v1 identity and exact evidence

The parent scientific contract is
[`p2-post-rank-direct-controller-v1.json`](../manifests/p2-post-rank-direct-controller-v1.json).
It was committed as `f48e69e3cf4c52621095365cf60cf2d4a4f8476b` after the
implementation commit `d2bc60fecb573170e1bf175713ad9f4320f881a5` and binds:

- manifest SHA-256:
  `d8d969b480d692e7ffa17b94504a7b25b75be602e1ce5c8d1821f9f2937384a7`;
- implementation tree SHA-256:
  `96dc5cd28bb113d0c659f8b0deab17f612170584c4187aa1ef211bd4bb139734`;
- attestation key ID:
  `67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf`;
- canonical trainer SHA-256:
  `c8240ba490a78931d86d516ff6d9a69e6cd7fe9c8dc41fb69fed009e6f4d68b4`.

The preserved v1 transaction consists of these exact files:

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| `training/training-matrix.summary.json` | 5,647 | `dc469a9c22ef295ed61022042fd6f1c4ddbd8544adb144986d590fc0f7b2ef7f` |
| `training/s55/seed-6071406/.p2-direct-training-cell.claim` | 65 | `60724fc6226380bc8b457cc3e4518c5060beda037fb4878f21b14614bb204309` |
| `training/s55/seed-6071406/s55-training.summary.json` | 481,864 | `9e3225442e621e6a957a2f3e53138686be7a37f2495552804be9d133d92dd8cf` |
| `training/s55/seed-6071406/s55-step-1000.pt` | 626,727,758 | `dfaa5da812e4a4301d8744ccacb5871dffcc19491a9fd2af4815cc24d0ca2f7f` |

Paths in the table are relative to
`artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/`. The claim
contains launch nonce
`964408511889289bd3306ab5b4fa1c9e2717327eb460b8ec3f6b7b953059572d`,
which exactly matches the signed training contract and checkpoint provenance.

The matrix ledger is HMAC-authenticated for purpose
`p2-direct-training-matrix-v1`, has payload SHA-256
`70be01b544782783a0f888b65ddc51c72a378a1a8ba8d84ded48d3f68d5a9c59`,
envelope payload SHA-256
`2858e06e66eab970eff26aa46d26892a7f7a47a5c9996226d1969c9bc5b8b130`,
and MAC
`4d3190b54419af6e4d39c9f67bfa85e1794d897b39050bd3f1f0ac4bdd9dab44`.
It records `status=in_progress`, `completed_runs=0`, and `expected_runs=10`.

The child summary is HMAC-authenticated for purpose
`p2-direct-training-summary-v1`, has semantic payload SHA-256
`3af80d42ce6115dbb149450c9042e49e015876a974059143840d0e176d3e8188`,
envelope payload SHA-256
`dfa6e3874b1bb2593868654939c1d5ba677b0cad76171dba44c734152644464f`,
and MAC
`0cd30857d610fc41a5552112c83831d49e2b97f089449889880be1709304e456`.
It records `steps_completed=1000`, `stopped_early=false`, and a
1,000-entry `sha256-ordered-training-step-chain-v1` transcript rooted at
`a3804b7a17f8206cf1a289d0ce7692cba0fc2ed7621920453fb0fccc16f6df88`.
The checkpoint byte count and SHA-256 in that summary exactly match the file.

The frozen execution binding was CPython 3.12.3, PyTorch 2.13.0+cu130, CUDA
runtime 13.0, driver 580.126.09, and one NVIDIA GB10 with compute capability
12.1 and 130,596,753,408 bytes. The summary, checkpoint, ledger, claim, source,
manifest, trainer, environment, and attestation bindings were independently
replayed read-only. With only the disputed optimizer-step predicate disabled in
memory, every other validation passed. That diagnostic bypass did not change or
admit an artifact and is not itself recovery authority.

## Observed boundary

At the incident boundary exactly one direct-cohort child had run. Its output
contains training losses and training-evaluation diagnostics, so this amendment
must not be described as blinded to all prerequisite diagnostics. However:

- the authoritative training ledger still contains zero admitted runs;
- no direct calibration artifact exists;
- no top-p physical-match artifact exists;
- no `10071406`--`10071410` held-out controller input, prediction, outcome,
  token stream, failure stream, integrity artifact, or summary exists;
- no arm, comparator, budget, family, context, replicate, threshold, statistical
  gate, or evaluation seed was selected using this training result.

The recovery decision is based only on optimizer and routing semantics plus
authenticated structural evidence. It must apply identically to all registered
training cells regardless of their loss, accuracy, memory gap, or later quality.

## Root cause

The trainer executes `optimizer.step()` once after backward and gradient
clipping in each of the 1,000 loop iterations, then appends the same global step
to the authenticated transcript. The runner nevertheless required every
parameter-local optimizer state to have `step == 1000`.

The preserved checkpoint has 375 optimizer-group parameters and 320 optimizer
state entries:

- 55 deliberately inactive HCA/MTP parameters have no optimizer state;
- all 192 always-gradient parameters are present at step 1000;
- all 128 routed-expert tensors are present;
- 74 routed-expert tensors are at step 1000;
- the remaining 54 routed-expert tensors are at integer steps 728--999;
- every sub-1000 entry belongs to a learned-routed MoE expert;
- each expert's `gate_up_proj` and `down_proj` counters agree;
- every state has exactly `step`, `exp_avg`, and `exp_avg_sq`, matching tensor
  shapes, and all state tensors are finite;
- the three hash-routed MoE layers have every expert at 1000, while only the
  five learned-routed layers contain lower counters.

PyTorch AdamW builds each update group only from parameters whose gradients are
not `None`; its counter is therefore a parameter activation/update count, not a
global optimizer-call counter. The model's MoE forward path invokes an expert
only when at least one token is routed to it. The observed pattern is the exact
architectural consequence of those two frozen behaviors. Requiring every routed
expert counter to equal 1000 would require dense expert execution that the
frozen trainer neither specified nor performed.

## Non-negotiable preservation rule

Until a separately committed and reviewed amendment manifest authorizes exact-byte
admission:

- do not rerun S55 seed `6071406`;
- do not launch another direct training, calibration, physical-match, or quality
  cell;
- do not delete, truncate, edit, replace, rename, or move the claim, ledger,
  summary, or checkpoint;
- do not regenerate or re-sign the v1 summary or checkpoint;
- do not copy checkpoint values into a new artifact and present the copy as a
  newly executed cell;
- do not manually append the cell to the v1 ledger;
- do not reuse the coordinate under v1 after merely removing its claim.

A bare claim unlink followed by a retry would create the unregistered selection
path explicitly forbidden by the frozen no-silent-retry contract. A checkpoint
or summary rewrite would destroy the evidence that recovery admitted the exact
pre-amendment computation.

## Required amendment predicate

Any amended validator must derive its rule from parameter roles, not from the
observed 728--999 range:

1. the optimizer group must retain the exact full parameter ordering, frozen
   hyperparameters, and exact schema;
2. every always-gradient parameter must have state and exact integer step 1000;
3. every registered routed-expert parameter must retain complete paired state,
   preserving the v1 exact active-state inventory requirement;
4. each routed-expert pair must have equal exact integer steps in `1..1000`;
5. one-sided or missing pair state, fractional/negative/zero/past-terminal
   counters, unexpected state, missing always-gradient state, schema/shape
   drift, and any non-finite tensor must fail closed;
6. the authenticated transcript and frozen trainer control flow remain the
   evidence for exactly 1,000 global optimizer calls.

Tests must reject dense step 999, routed steps 0, 1001, or fractional, pair
mismatch, partial pair state, unexpected inactive state, missing dense state,
shape/schema drift, and non-finite state. Tests must also accept synthetic
routed experts selected on fewer than 1,000 global steps without using this
artifact's observed counter values as fixtures or thresholds.

## One-shot exact-byte admission requirements

Admission is permissible only through a new immutable manifest revision with a
new manifest hash and explicit parent binding. The old v1 manifest must remain
byte-identical. The revision must record that it was frozen after one training
prerequisite artifact was observable but before calibration, physical matching,
or held-out quality. It must bind this report, the parent manifest SHA-256, all
four preserved evidence hashes above, the corrected general predicate, and an
unchanged canonical trainer hash.

The canonical invocation validates the current manifest and source before it
enters the one-shot admission-creation subroutine. Those pre-entry checks may
use additional frozen read-only Git provenance commands, including
`git cat-file`, `git ls-files`, `git rev-parse`, and `git status`; none executes
experiment code. The admission-creation subroutine itself must:

1. be entered only while the same scheduler lease, physical-device guard, and
   exclusive matrix lock are held, and re-assert all three immediately before
   commit;
2. refuse to start any scientific or experimental trainer, calibrator,
   physical matcher, or evaluator child process; the only permitted child
   processes within this creation subroutine are the frozen read-only Git
   provenance probes `git ls-tree` and `git merge-base --is-ancestor`, which do
   not execute experiment code;
3. require the preserved v1 ledger to be authenticated, `in_progress`, and at
   exactly zero completed runs;
4. require the exact canonical coordinate S55/6071406, claim path, claim bytes,
   nonce, summary bytes, checkpoint bytes, sizes, and hashes recorded here;
5. validate the old v1 manifest, implementation, source, trainer, environment,
   command, seed rules, transcript chain, payload digests, HMAC envelopes,
   checkpoint internal provenance, model/probe state, optimizer inventory, and
   corrected optimizer predicate;
6. fail if any calibration, physical-match, controller-quality, integrity, or
   summary artifact exists, or if any unregistered file appears in the direct
   output tree; repeat the closed-world check immediately before commit and as
   a postflight before any ledger promotion;
7. emit an HMAC-attested admission record that binds the old and amended
   contracts, incident report, exact original hashes, nonce, corrected predicate
   version, an explicit `scientific_child_processes_started_at_creation = 0`
   assertion, and the exact allowlist of read-only Git provenance commands
   within the creation subroutine;
8. atomically commit the complete admission from a fsynced staging inode using a
   no-replace hard link, derive its public binding from the same verified open
   inode, and recover only an exact staged prefix after interruption;
9. preserve the claim at its original path and exact bytes forever; deletion,
   movement, consumption, or an unrecorded unlink is forbidden;
10. be idempotent only for the identical already-admitted transaction and reject
    every alternate artifact, nonce, coordinate, or second admission attempt.

After that creation subroutine, the canonical runner renews its current-source
and environment checks, promotes only the exact existing cell into an amended
ledger while retaining its original v1 provenance, then independently reloads
and validates the signed admission and on-disk amended ledger. Only after that
validation may the remaining nine training cells run. They must use the
unchanged trainer bytes and scientific configuration. Each such child is
bracketed by open exact-byte snapshots of both the admission and the amended
ledger, with path identity rechecked before and after the child. Calibration
remains blocked until a terminal ten-cell training ledger, and held-out quality
remains blocked until every registered prerequisite is terminal.

If exact-byte admission cannot satisfy every condition, if any preserved hash or
authentication fails, if trainer/model/data/hyperparameters must change, or if a
downstream direct result has already been opened, this artifact cannot be
rescued. The fallback is a new experiment ID with a fully disjoint, prospectively
frozen training/calibration/evaluation seed namespace for the complete cohort;
rerunning only this coordinate is not an acceptable fallback.

## Unchanged scientific design

This incident does not authorize a scientific-design change. The amended logical
cohort must retain:

- scales S55 and S151;
- training seeds `6071406`--`6071410`;
- calibration seeds `7071406`--`7071410`;
- evaluation seeds `10071406`--`10071410`;
- budgets 2x and 4x, nine families, five contexts, and ten 20-example
  replicates;
- all 19 registered arms in their original roles and order:
  `hierarchical-soft-lag+pins`, `fixed+pins`,
  `hierarchical-balanced-fixed+pins`, `fixed-top-p-0.5+pins`,
  `fixed-top-p-0.8+pins`, `fixed`, `calibrated-no-pins`,
  `calibrated+pins`, `shuffled-quota`, `shuffled-quota+pins`,
  `local-no-pins`, `local+pins`, `hierarchical-soft-lag-no-pins`,
  `hierarchical-soft-lag+pins-no-score`,
  `hierarchical-soft-lag+pins-no-temporal`,
  `hierarchical-soft-lag+pins-no-cross-layer`,
  `hierarchical-soft-lag+pins-no-refresh`,
  `hierarchical-soft-lag+pins-permuted-quota`, and
  `hierarchical-soft-lag+pins+fallback`;
- the three independent confirmatory contrasts, physical-parity requirements,
  intent-to-treat failure treatment, intervals, multiplicity correction, and
  all GO/NO-GO clauses.

Training remains exactly 1,000 steps with early stopping disabled, BF16
autocast, batch size 16, four queries, training sequence lengths 64 and 80,
evaluation lengths 48, 64, and 80, training top-k 64, learning rate 0.001,
weight decay 0.01, gradient clipping at 1.0, evaluation every 50 steps with two
batches of 32, and unit answer/ranking/read-ranking/value loss weights. AdamW
remains fused with betas 0.9/0.999, epsilon 1e-8, no AMSGrad, and no maximization.
No observed training diagnostic may alter these values, replace a seed, drop a
checkpoint, reorder an arm, or relax a later gate.

## Paper and claim disclosure

The paper, artifact card, and final traceability table must disclose this event
even if exact-byte admission succeeds. Suitable wording is:

> After the first prospective training prerequisite completed, we found that
> the frozen runner incorrectly equated parameter-local Adam counters with the
> 1,000-step global optimizer count for sparsely routed MoE experts. Before any
> calibration, physical-match, or held-out controller result, we froze a
> validator-only amendment and admitted the exact HMAC-bound checkpoint without
> retraining or changing its bytes. The cohort, seeds, trainer, hyperparameters,
> arms, estimands, and decision rules were unchanged, and the corrected
> architecture-derived predicate was applied to every training cell.

The manuscript must also state that training diagnostics from the first cell
were observable at amendment time. It may describe the held-out controller
evaluation as prospective after the amendment, but it must not claim that the
final validator implementation was frozen before every prerequisite cell or
that v1 completed unchanged. Until a signed admission exists, the only supported
status is **incomplete prerequisite transaction under preserved review**; no
quality, causal, memory, latency, throughput, or production conclusion follows.
