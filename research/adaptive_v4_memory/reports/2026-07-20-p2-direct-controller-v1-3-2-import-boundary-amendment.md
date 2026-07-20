# P2 direct-controller v1.3.2 trusted-environment import-boundary amendment

Date: 2026-07-20 (Asia/Seoul)

Status: prospective infrastructure amendment after a v1.3.1 launch failure, before any held-out quality input, prediction, outcome, aggregate, or matrix record existed

## Amendment trigger

Revision v1.3.1 atomically published its static admission, zero-prefix genesis, and quality-start activation, then started the frozen one-cell operational launch. The persistent evaluator exited before its ready receipt. The sealed bootstrap correctly admitted the exact, prevalidated `.venv/lib/python3.12/site-packages` directory as the trusted environment, but the evaluator's later repository-origin guard rejected `typing_extensions.py` because that same exact directory is lexically inside the repository and is not part of the frozen research implementation inventory. The two guards therefore imposed contradictory policies on one already sealed dependency origin.

The failing origin check executes before evaluator input establishment and before the persistent ready receipt. The authenticated terminal ledger records `launch_failure`, child return code 1, no ready or final receipt, and zero work orders, results, or reingestions. The signed ledgers do not contain the exception text. The diagnosis is therefore disclosed by this source-bound incident report and by a deterministic regression reproducer; it is not misrepresented as a signed historical field.

## Immutable v1.3.1 failure lineage

The v1.3.1 roots and files remain in place. They may not be deleted, moved, repaired, resumed, relabeled, or reused as v1.3.2 output. A v1.3.2 admission must authenticate these exact bindings and their HMAC chains:

| Artifact | SHA-256 | Bytes | Authenticated state |
|---|---|---:|---|
| frozen manifest | `980144f5ae00ae8381f4ced1a93d614a1bdd2a361060fa344bb55db8273ae50d` | 56,967 | clean source commit `828e8e0c57042b71a1117cd067b113f20d61c04c` |
| reuse admission | `c2c786d437a3d42d3329947d8d7941751d7b9561317eaf00c92e10cfeb27c5ea` | 11,092 | terminal static admission |
| pre-heldout genesis | `3d8a4f0f006f916586a223ab5f0cc92e72c17e66267eb85bc8ebbaa9ef27948e` | 10,427 | zero prefix |
| activation matrix lock | `101d46a05395e701d04b42b471ca67dc778117e01c87405196df227f4216c642` | 509 | exact activation member |
| quality-start activation | `e14b9e672fe0beb93203b51faa252ec17885b60ec52eac4890cdfe89e0e151d6` | 177,754 | activated zero prefix |
| matrix ledger | `79d726212ca6bf9af1f9f8c380d9dbd8b0e98fcfc36fd6cec64870bdbe17eebb` | 194,432 | in progress, zero records and zero completed shards |
| persistent launch ledger | `db938290daa24802a8cb084525e535b1068417f33bf8608a7a87fef635a46e9a` | 21,891 | one planned first-coordinate launch |
| persistent terminal ledger | `4a638fc41ab27fafcdd04c0bd47eea31fb7122b05442f79a4a0cdb68f3a91454` | 19,659 | failure before ready |
| residual cell claim | `4deb50235d9f6999e09a2b4e3bf3ae53f689c1a75e3b6afd54a2ce53949b0c20` | 532 | unsigned, dead-owner coordination residue |

The manifest source commit is distinct from the implementation commit `b473b237689ed7466a3a27db7054d2770afe60b8`, whose frozen implementation digest is `6b861b5c1869c42442cc432bf940c0403d030a31c4b0438c1bd99166a30db7e0`. The inherited v1.3 signed-empty-lineage digest remains `9ec45d1ededa576d76f0cec3614ce0606e195aee2a1475aa5ff1f1d7ff4f39cc`. The canonical repository-relative v1.3.1 failure-lineage projection is `2e2b7c1df5eaaf316ad53abc4efd8c4d86651da1d18ba1129e2561cdb6b349b0`; v1.3.2 admission must reproduce this digest from the authenticated live artifacts exactly.

The output closed world contains exactly the matrix ledger, the residual claim, and the claim's six parent directories below the output root. No shard envelope, examples, outcomes, token evidence, failure sidecar, worker ledger, integrity result, or final summary exists. The activation receipt's signed `active_claim_count=0` describes the prelaunch instant. After the failed process exited there is one claim file, zero live active claims, and one orphan claim; its dead owner PID must remain dead when the v1.3.2 lineage is admitted. The claim coordinate equals the authenticated session plan's sole first coordinate, but the claim itself is unsigned and is treated only as byte-exact coordination residue.

## Prospective v1.3.2 change

Revision v1.3.2 changes only the launch/import boundary and adds a post-activation, pre-claim evaluator-ready preflight:

1. the sealed bootstrap passes the already validated exact site-packages path into the evaluator as a versioned sealed global;
2. the runtime origin guard accepts a module only when both its lexical absolute origin and resolved origin are within that exact sealed site-packages root;
3. every other repository-local module must still be an exact `.py` member of the frozen implementation inventory;
4. symlink escape, repository shadowing, bytecode-only modules, extension shadowing, and unresolved frozen imports remain fail-closed;
5. immediately after the zero-prefix activation is published, a disposable persistent evaluator must emit a valid ready receipt and then a zero-work stopped receipt; only then may the initial zero-record matrix bind that evidence and become eligible to precede a cell claim.

The preflight loads the admitted model and validates the activation, static inputs, transport, import policy, runtime, checkpoint, and selected-device bindings. It may not construct an evaluation workload, initialize an evaluation seed, materialize a held-out input, issue a work order, create a cell claim or shard bundle, or produce a prediction or outcome. Its launch, ready, stopped, and terminal receipts are HMAC-authenticated in the v1.3.2 persistent-session namespace and are incorporated into the signed zero-record matrix-ledger projection before the first cell can be claimed. Activation preceding this check is an administrative zero-prefix boundary, not a quality observation.

Every v1.3.2 experiment ID, manifest, admission, genesis, activation, output, worker/session ledger, artifact-owned lock, launcher ID, file-descriptor environment name, typed authority/seal, and HMAC purpose uses a distinct `v1.3.2` or `v1-3-2` namespace. The shared host GPU scheduler and physical-device mutexes remain external coordination primitives.

## Unchanged scientific protocol

The amendment does not change any of the 9,000 coordinates, five training seeds, two scales, two budgets, 17 exact-fill arms, workload families, contexts, replicates, generation seeds, estimands, multiplicity rules, success gate, quality-blind one-cell invariant gate, or unconditional full-resume obligation. No outcome direction was available or used to make this amendment. Final reporting must label the result as a quality-blind v1.3.2 import-boundary amendment descended from the immutable v1.3.1 zero-quality launch failure, not as an unchanged v1.3 or v1.3.1 preregistration.
