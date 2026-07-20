# P2 direct-controller v1.3.1 quality-start activation amendment

Date: 2026-07-20 (Asia/Seoul)

Status: prospective protocol amendment, before any held-out quality input, prediction, outcome, or aggregate existed

## Reason for the amendment

Revision v1.3 froze and atomically published a signed historical-reuse admission and a signed zero-prefix genesis. Its canonical Git-object launcher then proved unable to transport the externally stored attestation-key path into the sanitized matrix process. No v1.3 quality output, matrix lock, worker ledger, persistent-session ledger, integrity result, or summary existed when this defect was found. The signed v1.3 admission and genesis remain immutable evidence; they are neither overwritten nor relabeled.

Revision v1.3.1 changes only the quality-start authority and transport boundary. It preserves the 9,000 coordinates, five seeds, two scales, two budgets, 17 exact-fill arms, estimands, multiplicity handling, and success gate. Every v1.3.1 experiment identifier, manifest path, output root, admission root, activation root, artifact-owned lock/ledger namespace, and HMAC purpose is distinct with a `v1-3-1` or `v1.3.1` suffix. The host-wide GPU scheduler and physical-device mutexes are deliberately shared across experiments; they are external coordination locks rather than result artifacts.

The canonical v1.3.1 execution topology is one worker on one selected physical GPU. The sanitized Git-object transport does not carry a sealed per-worker CUDA device route, so distributed execution is explicitly unsupported in this revision rather than being inferred from distinct scheduler-lock paths. A later revision may add authenticated device routing without changing these results.

## Signed superseded-empty lineage

The v1.3.1 admission must authenticate one byte-exact v1.3 lineage only:

- result-source commit `62ba095614f9614cb85a06f1a043a89a21f7f1dd` and tree `5779ff8261029c1c53bc834a6dc7d16c84963fcf`;
- implementation commit `a729723bc3e513dd10ed06ac65f7128b4b726f98`, tree `a7b1cc62f22d3993fc180c33444c96d0ed0eea3d`, and digest `5c96c102ffa7cabc974d0a86ba60f536b513829c59ded31d615111c1312a7ebb`;
- manifest SHA-256 `7fb1bc578b112031ce1ad914c7ea1cdc09fdf21a2c6fcde8362e93df60dd994e`;
- admission SHA-256 `0ec0efc962b77389c1a2db3cd1cf96a19ae989afb131c0d8d05a242f7a38a574`;
- genesis SHA-256 `dff6d05c69c04b1b83d7edb3d5da650daef691d7e508defcc70decd9328bbdd9`;
- attestation key ID `67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf`.

Validation checks the exact bytes, HMAC domains and MACs, manifest/source bindings, admission-to-genesis binding, empty records and counters, and live absence of every superseded v1.3 prospective quality path. Key bytes remain outside the repository, artifacts, command lines, and logs.

## Typed start authority

There is no `skip_preheldout`, `already_started`, or equivalent raw Boolean escape hatch.

1. A `PrestartQualityAuthorityV1_3_1` is issued only after the new static admission/genesis, the signed v1.3 empty lineage, the clean v1.3.1 source/manifest, and live absence of all v1.3.1 quality and activation paths agree.
2. `publish_quality_start_activation` accepts that exact type plus the complete base-prerequisites binding, sealed Git-object source provenance, and sealed matrix routing.
3. Under a fixed bootstrap flock it creates a private staging directory, creates and exclusively flocks `matrix.lock`, binds the root and lock device/inode/ownership/mode/link identities into an HMAC receipt, writes exactly `matrix.lock` and `quality-start-activation.json`, fsyncs both files and the directory, and publishes with `renameat2(RENAME_NOREPLACE)`.
4. The returned `QualityStartActivationLeaseV1_3_1` retains the same pre-rename lock file description. There is no unlocked interval between activation and the initial zero-ledger publication.
5. A restarted runner first obtains a read-only `ValidatedQualityStartActivationV1_3_1`, then opens the existing lock without `O_CREAT`, acquires its flock, and revalidates the complete activation. Evaluators and auditors use only the non-owning validated capability, so they cannot deadlock behind a parent-held flock.

The activation root is an immutable exact-two-member sibling of the quality output root. Recovery may remove only validated staging directories. A partial or existing final root is never repaired, replaced, or deleted automatically.

## Evidence replay, consumer scope, and durability

Each launch authority performs exactly one full replay of the ten admitted scale/training-seed checkpoint bindings. Fresh execution obtains that replay through `PrestartQualityAuthorityV1_3_1`; resume obtains it through the initial full activation load. Publication, lease acquisition, and later owner-side checks reuse the sealed typed capability while rechecking the activation exact-two root, HMACs, live receipt files, source/routing, base-prerequisites binding, and lock identity. They do not silently rescan the full checkpoint grid.

Evaluator and audit consumers receive a distinct sealed `ActivatedConsumerAuthorityV1_3_1`. It authenticates the activation chain and exactly one requested scale/training-seed calibration/checkpoint pair, exposes only that singleton pair, and cannot acquire the activation lease or enter any full-owner/static-bundle API. A run-local audit cache contains ten such authorities and twenty budget-specific arm constructions; callback and no-callback audit paths share the same cache and revalidate each unique authority at finalization. No raw Boolean or nominal dataclass can substitute for either capability.

Directory publication and recovery are durable at every ownership boundary: files, child directories, and parent directory entries are fsynced; canonical modes are normalized and rechecked; interrupted temporary roots are adopted only under their persistent locks; and resume fsyncs the activation root and parent before handing off its lease. Failed post-wrap assertions, post-rename validation, device-guard acquisition, and same-process concurrent lease attempts close or reject their descriptors without an unlocked capability handoff or self-deadlock.

## Frozen execution order

The first execution is operational, not an interim scientific analysis:

1. run the canonical read-only prerequisites-only phase;
2. publish fresh activation while retaining its matrix flock;
3. publish the zero matrix prefix, then run the canonical matrix with `max_new_cells=1`;
4. inspect only integrity, schema, selected-device binding, and closed-world storage invariants;
5. resume the full frozen matrix regardless of the first cell's outcome direction.

The signed ledger state prevents a restart or competing process from bypassing this order. A resume that observes an authenticated zero-shard ledger may execute exactly the one operational cell and must stop. Uncapped full resume is enabled only after the exact first canonical cell passes automated integrity, schema, device-binding, and closed-world validation; no quality value or outcome direction is consulted by that gate.

During step 4, quality values may not be inspected, summarized, compared, or used to select a seed, scale, budget, arm, family, context, replicate, configuration, analysis, continuation decision, or success rule. If the operational invariants are valid, full resume is mandatory. If an infrastructure invariant fails, repair and replay must follow the authenticated recovery protocol and still may not use outcome direction.

This sequence is scientifically safe within the stated trusted-pipeline threat model because the continuation rule is fixed before materialization and is independent of the first outcome. It would cease to be safe if a human or program inspected the value and allowed it to influence stopping, configuration, or reporting. The activation receipt therefore freezes both the permitted checks and the unconditional-resume obligation.

## Claim boundary

This amendment does not restore global outcome independence, change the already disclosed v1.2 feasibility observation, or support any natural-language, large-model, production-serving, total-HBM, transfer, or population-significance claim. Final reporting must identify the experiment as a quality-blind v1.3.1 activation amendment descended from a signed-empty v1.3 prefix, not as an unchanged v1.3 preregistration.
