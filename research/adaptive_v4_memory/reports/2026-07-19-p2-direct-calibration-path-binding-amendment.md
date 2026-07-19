# P2 direct calibration path-binding amendment

Date: 2026-07-19

Status: prospective recovery amendment after one calibration result was observed;
no top-p physical-match or held-out controller-quality result had been started.

## Incident

The frozen revision 1.1 calibration runner launched the first registered cell,
S55 training seed `6071406` / calibration seed `7071406`.  The canonical
calibrator completed normally, published a MAC-attested terminal artifact, and
reported `GO` for both frozen budgets and for the aggregate cell.  The parent
runner then stopped before ledger promotion with:

```text
ValueError: Calibration checkpoint binding drifted.
```

This was a verifier false negative, not a model, workload, checkpoint, or
calibration failure.  The terminal training ledger stores the checkpoint path
using its authenticated repository-relative spelling.  The runner passed the
resolved absolute referent to the child.  The child verified that referent and
correctly preserved the authenticated relative binding in its artifact.  Only
the parent expectation helper changed the spelling with `Path.resolve()` before
requiring full-dictionary equality.  SHA-256 and byte size never differed.

The failure occurred after the child artifact was published and before the
claim was released or any cell was appended to the matrix ledger.  Revision
1.1 therefore remains incomplete at zero promoted cells.  Its preserve-on-
failure claim correctly blocks silent resume.

## Frozen observed evidence

The complete failed revision 1.1 calibration root remains at
`artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration/`.
Revision 1.2 must never delete, edit, replace, rename, move, truncate, or use
that tree as its writable output root.

| Evidence | Bytes | SHA-256 |
|---|---:|---|
| `calibration/calibration-matrix.summary.json` | 8,290 | `f0dccaa9861e095b297c22a17735b3a379d4f9ed8db628e0bcf222b12da5e426` |
| `calibration/s55/seed-6071406/.p2-direct-calibration-cell.claim` | 360 | `462793153ad22a19223398ef30c2e4624ae247d179a8c29cb250dca1b3bca2cb` |
| `calibration/s55/seed-6071406/s55-calibration.json` | 16,484,030 | `f805d70cb1379cc71c6d6abbe34d579880bd8cec35b72ea816ce9cbc7775d25b` |
| terminal revision 1.1 training ledger | 18,032 | `786669b8feb74eef5a4aa1e57dccc3ffada10596a8ae995d78e931daeef06cb5` |
| S55/6071406 checkpoint | 626,727,758 | `dfaa5da812e4a4301d8744ccacb5871dffcc19491a9fd2af4815cc24d0ca2f7f` |

The failed artifact has payload SHA-256
`b177648d267b44960e9c0dc2bf8953c5149e76a3854cc0dca3155c0d21130e45`,
calibration attestation MAC
`b0271103f2f8d3d8b53f85cced550cbf62f53d084280a35827f3efb9db27aae1`,
and launch nonce
`0f16510c0b662fe099ecaafd7671fa0db9de608e9e4a31fd3d56f7a1a01c7e37`.
The revision 1.1 manifest SHA-256 is
`1d059f83ca73945b9df5dbee20752fbf3f99c0a24794c533be9c52a4230b4c0b`.

## Non-negotiable quarantine rule

The revision 1.1 root is the quarantine.  Keeping it at its original path
preserves the exact claim and artifact while avoiding a rename transaction.
It is never an input artifact for revision 1.2 scientific decisions.  Before
revision 1.2 creates its admission and before and after every new calibration
child, the runner must verify the old ledger, claim, artifact, terminal training
ledger, and checkpoint against the frozen bytes above.  Exact open-inode
snapshots remain held across each new child.

A bare claim unlink, artifact rewrite, manual append to the old ledger, or
automatic adoption of the observed `GO` artifact is forbidden.

## Corrected verifier predicate

The parent must compare the artifact checkpoint binding with the already
authenticated upstream checkpoint binding exactly, including its original path
spelling.  Resolved referent equality, regular-file/no-symlink status, SHA-256,
and byte size remain separate mandatory checks.  No calibration threshold,
budget, seed, model parameter, workload, or estimator changes.

## Registered one-shot retry

Revision 1.2 writes only to
`artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration-v1-2/`.
Under the project GPU lease, physical-device guard, and matrix lock, it first
creates one exact-byte, MAC-attested retry admission.  Admission creation starts
zero trainer, calibrator, evaluator, or other scientific subprocesses.  The
admission binds the immutable revision 1.1 evidence, current manifest and source,
terminal training matrix, frozen environment, old observed decision, and this
report.

The scheduler lease is frozen to the exact path authenticated by the failed
attempt, `/tmp/adaptive-v4-direct-gpu0.lock`. Before the final admission link can
consume the exception, the runner completes every read-only check of that lease,
the exact failed-attempt CUDA execution environment, manifests, implementation,
terminal training prerequisite, quarantine inventory, and this report. A call
from a different route, driver/runtime, Python environment, or scheduler path
therefore fails without creating the amended root or final admission.

Because the terminal `GO` was already visible when this amendment was registered,
the authorization is described as result-disclosed and cause-based, not as a
counterfactual outcome-independent decision.  The old result is never admitted to
the amended cohort, and the complete ten-cell grid is recomputed unchanged.

Committing the admission consumes the exception.  Only the same still-running
process that exclusively created that admission may launch the first child.  A
crash or restart after admission commit but before the first ledger promotion is
terminal fail-closed evidence and may not launch the coordinate again.  This rule
also makes an admission-only or empty-prefix partial rollback non-replayable.
A staging-only interrupted creation may be recovered because no final admission
was committed; the recovering process consumes the exception only when it
exclusively links and validates the final admission. Failure during or after that
commit, including before the empty-prefix ledger is durable, is an availability-
terminal fail-closed event. Owner-controlled deletion, filesystem snapshot
restoration, or rollback of the complete amended root is outside the process-level
threat model, invalidates the evidence, and is not an authorized replay mechanism.

After the admission and empty-prefix amended ledger independently validate,
the runner executes the complete original ten-cell coordinate sequence.  The
first child is necessarily S55/6071406 and is the sole registered retry.  It is
run exactly once regardless of the old `GO` result.  All later coordinates are
first attempts.  Any new unexpected exit or invalid artifact preserves its new
claim and again fails closed; this amendment does not create a general retry
policy.

The closed-world quarantine inventory is rescanned immediately before and after
every child-bound evidence snapshot, while exact regular-file descriptors for all
registered evidence remain open across the child and ledger promotion.

Revision 1.2 may proceed to top-p physical matching only if all ten new
calibration cells are terminal `GO`.  The old artifact is disclosure evidence,
not one of those ten cells.  Held-out controller quality remains unobserved at
the freeze point, and no result from this incident may change arms, budgets,
seeds, thresholds, estimands, or early stopping.
