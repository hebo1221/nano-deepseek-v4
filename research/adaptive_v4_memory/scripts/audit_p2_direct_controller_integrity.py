from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_controller_matrix as matrix

EXPERIMENT_ID = "p2-post-rank-direct-controller-integrity-v1"
ARTIFACT_TYPE = "direct-controller-closed-world-integrity-audit"
SCHEMA_VERSION = 1
ATTESTATION_PURPOSE = "p2-direct-controller-integrity-v1"
INTEGRITY_OUTPUT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-integrity.json"
)
AUDIT_BOUNDARY = (
    "closed-world structural/provenance/HMAC/physical-evidence audit only; raw correctness is "
    "neither aggregated nor used to select, stop, include, exclude, or order any arm or shard"
)
INTEGRITY_CHECK_FIELDS = (
    "terminal_matrix_hmac_verified",
    "closed_world_output_tree_verified",
    "all_coordinates_exact_and_ordered",
    "all_launch_nonces_verified",
    "all_external_inputs_revalidated",
    "all_envelope_hmacs_verified",
    "all_sidecar_digests_and_row_schemas_streamed",
    "all_examples_paired_across_arms",
    "all_20_by_19_outcome_rows_verified",
    "all_token_physical_evidence_verified",
    "no_outcome_dependent_selection_or_stopping",
)
INTEGRITY_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "integrity_status",
        "source",
        "manifest",
        "matrix",
        "audit_boundary",
        "outcome_selection_performed",
        "quality_outcomes_aggregated",
        "coordinate_count",
        "coordinate_digest",
        "examples_per_shard",
        "arm_names",
        "expected_outcome_rows",
        "validated_shards",
        "validated_examples",
        "validated_outcome_rows",
        "validated_integrity_pass_shards",
        "validated_integrity_fail_shards",
        "storage",
        "checks",
        "bundle_inventory_digest",
        "bundle_inventory",
    }
)
INTEGRITY_ARTIFACT_FIELDS = INTEGRITY_PAYLOAD_FIELDS | {
    "payload_sha256",
    "attestation",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json_nofollow(path: Path, *, label: str) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        try:
            payload = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid {label}: {path}") from error
        _require(isinstance(payload, dict), f"{label} must be a JSON object.")
        opened.assert_unchanged()
        return payload
    finally:
        opened.close()


def _file_binding(path: Path, *, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        result: dict[str, Any] = {
            "path": str(opened.path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
        }
        if payload is not None:
            envelope = payload.get("attestation")
            _require(isinstance(envelope, Mapping), f"Attestation is missing: {path}")
            result["payload_sha256"] = payload.get("payload_sha256")
            result["attestation_mac"] = cast(Mapping[str, Any], envelope).get("mac")
        opened.assert_unchanged()
        return result
    finally:
        opened.close()


def _digest_bound_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("attestation", None)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = contract.json_digest(result)
    return result


def _attested_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    result = _digest_bound_payload(payload)
    result["attestation"] = attestation.attest_payload(
        result,
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    return result


def _verify_attestation(payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot) -> None:
    digest = payload.get("payload_sha256")
    source = dict(payload)
    envelope = source.pop("attestation", None)
    source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(source),
        "Integrity-audit payload digest drifted.",
    )
    _require(isinstance(envelope, Mapping), "Integrity-audit attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )


def _exclusive_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"Refusing to overwrite integrity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            identity = (metadata.st_dev, metadata.st_ino)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite integrity artifact: {path}") from error
        opened = attestation.open_regular_nofollow(path)
        try:
            _require(
                (opened.device, opened.inode) == identity and opened.read_bytes() == encoded,
                "Published integrity artifact bytes drifted.",
            )
            opened.assert_unchanged()
        finally:
            opened.close()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _evaluator_module() -> Any:
    return importlib.import_module("evaluate_p2_direct_controller_shard")


def _trust_root_for_manifest(
    manifest_binding: Mapping[str, Any],
    *,
    output_roots: tuple[Path, ...],
    attestation_key_path: Path | None,
) -> attestation.TrustRoot:
    public = manifest_binding.get("attestation")
    _require(isinstance(public, Mapping), "Manifest attestation binding is missing.")
    key_id = cast(Mapping[str, Any], public).get("key_id")
    _require(contract.is_sha256(key_id), "Manifest attestation key ID is invalid.")
    if attestation_key_path is None:
        return attestation.trust_root_from_environment(
            repository_root=matrix.REPOSITORY_ROOT,
            artifact_roots=output_roots,
            expected_key_id=cast(str, key_id),
        )
    return attestation.load_trust_root(
        attestation_key_path,
        repository_root=matrix.REPOSITORY_ROOT,
        artifact_roots=output_roots,
        expected_key_id=cast(str, key_id),
    )


def _load_validated_terminal_matrix(
    *,
    manifest_path: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    top_p_output_root: Path,
    output_root: Path,
    matrix_summary: Path,
    evaluator_script: Path,
    attestation_key_path: Path | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    matrix.FrozenPrerequisites,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    layout = matrix._validate_matrix_layout(
        output_root=output_root,
        matrix_summary=matrix_summary,
        training_output_root=training_output_root,
        calibration_output_root=calibration_output_root,
        top_p_output_root=top_p_output_root,
        attestation_key_path=attestation_key_path,
    )
    prerequisites = matrix.load_and_validate_prerequisites(
        manifest_path=manifest_path,
        training_output_root=layout.training_output_root,
        calibration_output_root=layout.calibration_output_root,
        top_p_output_root=layout.top_p_output_root,
        output_root=layout.output_root,
        attestation_key_path=attestation_key_path,
    )
    canonical = matrix._canonical_evaluator(evaluator_script)
    descriptor, snapshot = matrix._open_evaluator(canonical)
    os.close(descriptor)
    evaluator_binding = snapshot.public_binding
    lock_binding = matrix._matrix_lock_binding(layout.lock_path)
    payload = _load_json_nofollow(layout.matrix_summary, label="terminal controller matrix")
    records = matrix.validate_matrix_summary(
        payload,
        output_root=layout.output_root,
        prerequisites=prerequisites,
        evaluator_script=canonical,
        evaluator_binding=evaluator_binding,
        matrix_lock_binding=lock_binding,
        verify_bundles=True,
    )
    _require(
        payload.get("status") == "terminal"
        and len(records) == matrix.EXPECTED_SHARDS
        and payload.get("completed_shards") == matrix.EXPECTED_SHARDS,
        "Integrity audit requires the terminal 9,000-shard matrix.",
    )
    matrix._preflight_output_tree(
        output_root=layout.output_root,
        matrix_summary=layout.matrix_summary,
        completed_shards=len(records),
    )
    return payload, records, prerequisites, evaluator_binding, lock_binding


def _inventory_record(record: Mapping[str, Any]) -> dict[str, Any]:
    coordinate = {field: record[field] for field in matrix.ShardCoordinate.__dataclass_fields__}
    artifact = record.get("artifact_bundle")
    inputs = record.get("inputs")
    _require(isinstance(artifact, Mapping), "Matrix artifact bundle is missing.")
    _require(isinstance(inputs, Mapping), "Matrix input binding is missing.")
    envelope = cast(Mapping[str, Any], artifact).get("envelope")
    sidecars = cast(Mapping[str, Any], artifact).get("sidecars")
    _require(isinstance(envelope, Mapping), "Matrix envelope binding is missing.")
    _require(isinstance(sidecars, Mapping), "Matrix sidecar bindings are missing.")
    envelope_map = cast(Mapping[str, Any], envelope)
    sidecars_map = cast(Mapping[str, Any], sidecars)
    _require(set(sidecars_map) == set(matrix.BUNDLE_KINDS), "Matrix sidecar inventory drifted.")
    row_counts = {
        kind: cast(Mapping[str, Any], binding).get("row_count")
        for kind, binding in sidecars_map.items()
    }
    _require(
        row_counts["examples"] == matrix.EXAMPLES_PER_SHARD
        and row_counts["outcomes"] == matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
        "Matrix bundle does not contain exactly 20 paired examples by 19 arms.",
    )
    return {
        "coordinate": coordinate,
        "coordinate_key": record.get("coordinate_key"),
        "launch_nonce": record.get("launch_nonce"),
        "integrity_decision": record.get("integrity_decision"),
        "input_binding_digest": cast(Mapping[str, Any], inputs).get("input_binding_digest"),
        "envelope": dict(envelope_map),
        "sidecars": {
            kind: dict(cast(Mapping[str, Any], binding)) for kind, binding in sidecars_map.items()
        },
        "row_counts": row_counts,
        "bundle_digest": contract.json_digest(
            {
                "envelope": dict(envelope_map),
                "sidecars": {
                    kind: dict(cast(Mapping[str, Any], binding))
                    for kind, binding in sidecars_map.items()
                },
            }
        ),
        "matrix_record": dict(record),
    }


def _integrity_payload(
    *,
    matrix_payload: Mapping[str, Any],
    matrix_summary: Path,
    records: Sequence[Mapping[str, Any]],
    prerequisites: matrix.FrozenPrerequisites,
) -> dict[str, Any]:
    inventory = [_inventory_record(record) for record in records]
    pass_count = sum(item["integrity_decision"] == "INTEGRITY-PASS" for item in inventory)
    fail_count = sum(item["integrity_decision"] == "INTEGRITY-FAIL" for item in inventory)
    expected_status = "INTEGRITY-PASS" if fail_count == 0 else "INTEGRITY-FAIL"
    _require(
        matrix_payload.get("integrity_status") == expected_status,
        "Matrix/audit integrity status drifted.",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal",
        "integrity_status": expected_status,
        "source": prerequisites.context.source,
        "manifest": prerequisites.context.manifest_binding,
        "matrix": _file_binding(matrix_summary, payload=matrix_payload),
        "audit_boundary": AUDIT_BOUNDARY,
        "outcome_selection_performed": False,
        "quality_outcomes_aggregated": False,
        "coordinate_count": matrix.EXPECTED_SHARDS,
        "coordinate_digest": matrix.coordinate_digest(),
        "examples_per_shard": matrix.EXAMPLES_PER_SHARD,
        "arm_names": list(matrix.ARM_NAMES),
        "expected_outcome_rows": matrix.EXPECTED_OUTCOME_ROWS,
        "validated_shards": len(inventory),
        "validated_examples": len(inventory) * matrix.EXAMPLES_PER_SHARD,
        "validated_outcome_rows": len(inventory)
        * matrix.EXAMPLES_PER_SHARD
        * len(matrix.ARM_NAMES),
        "validated_integrity_pass_shards": pass_count,
        "validated_integrity_fail_shards": fail_count,
        "storage": matrix_payload.get("storage"),
        "checks": {field: True for field in INTEGRITY_CHECK_FIELDS},
        "bundle_inventory_digest": contract.json_digest(inventory),
        "bundle_inventory": inventory,
    }


def audit_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = matrix.TRAINING_OUTPUT_ROOT,
    calibration_output_root: Path = matrix.CALIBRATION_OUTPUT_ROOT,
    top_p_output_root: Path = matrix.TOP_P_OUTPUT_ROOT,
    output_root: Path = matrix.OUTPUT_ROOT,
    matrix_summary: Path = matrix.MATRIX_SUMMARY,
    evaluator_script: Path = matrix.EVALUATOR_SCRIPT,
    output: Path = INTEGRITY_OUTPUT,
    attestation_key_path: Path | None = None,
) -> dict[str, Any]:
    canonical_output = matrix._exact_resolved_path(output, label="Integrity audit output")
    _require(
        not matrix._paths_overlap(
            canonical_output,
            matrix._exact_resolved_path(output_root, label="Controller output root"),
        ),
        "Integrity artifact must be outside the raw controller output tree.",
    )
    _require(
        not canonical_output.exists() and not canonical_output.is_symlink(),
        f"Refusing to overwrite integrity artifact: {canonical_output}",
    )
    lock_path = matrix._matrix_lock_path(output_root)
    with matrix._exclusive_matrix_lock(lock_path, matrix_summary=matrix_summary):
        payload, records, prerequisites, _evaluator, _lock = _load_validated_terminal_matrix(
            manifest_path=manifest_path,
            training_output_root=training_output_root,
            calibration_output_root=calibration_output_root,
            top_p_output_root=top_p_output_root,
            output_root=output_root,
            matrix_summary=matrix_summary,
            evaluator_script=evaluator_script,
            attestation_key_path=attestation_key_path,
        )
        unsigned = _integrity_payload(
            matrix_payload=payload,
            matrix_summary=matrix_summary,
            records=records,
            prerequisites=prerequisites,
        )
        result = _attested_payload(unsigned, trust_root=prerequisites.trust_root)
        validate_integrity_artifact(
            result,
            matrix_summary=matrix_summary,
            output_root=output_root,
            trust_root=prerequisites.trust_root,
            verify_bindings=False,
        )
        _exclusive_atomic_write_json(canonical_output, result)
        return result


def validate_integrity_artifact(
    payload: Mapping[str, Any],
    *,
    matrix_summary: Path = matrix.MATRIX_SUMMARY,
    output_root: Path = matrix.OUTPUT_ROOT,
    trust_root: attestation.TrustRoot | None = None,
    verify_bindings: bool = True,
    restream_raw: bool = True,
) -> dict[str, Any]:
    """Validate an integrity artifact and, by default, re-stream every raw bundle.

    ``verify_bindings=False`` skips only the terminal-matrix and closed-world tree
    replay.  Raw re-streaming remains independently controlled by ``restream_raw``.
    ``restream_raw=False`` retains the authenticated matrix-record binding and the
    closed-world filesystem check, but defers raw-envelope and sidecar re-streaming
    to :func:`iter_validated_raw_shards`.  This is intended for consumers that
    immediately use that iterator and therefore already perform the same fail-closed
    evaluator validation once before consuming each bundle.
    """

    _require(
        verify_bindings or restream_raw,
        "restream_raw=False requires verify_bindings=True.",
    )
    manifest = payload.get("manifest")
    _require(isinstance(manifest, Mapping), "Integrity manifest binding is missing.")
    if trust_root is None:
        trust_root = _trust_root_for_manifest(
            cast(Mapping[str, Any], manifest),
            output_roots=(output_root,),
            attestation_key_path=None,
        )
    active_trust_root = trust_root
    _verify_attestation(payload, trust_root=active_trust_root)
    _require(
        set(payload) == INTEGRITY_ARTIFACT_FIELDS,
        "Integrity artifact top-level schema drifted.",
    )
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("artifact_type") == ARTIFACT_TYPE
        and payload.get("status") == "terminal"
        and payload.get("integrity_status") in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
        "Integrity artifact identity/status drifted.",
    )
    _require(
        payload.get("audit_boundary") == AUDIT_BOUNDARY
        and payload.get("outcome_selection_performed") is False
        and payload.get("quality_outcomes_aggregated") is False,
        "Integrity outcome-embargo boundary drifted.",
    )
    _require(
        payload.get("coordinate_count") == matrix.EXPECTED_SHARDS
        and payload.get("coordinate_digest") == matrix.coordinate_digest()
        and payload.get("examples_per_shard") == matrix.EXAMPLES_PER_SHARD
        and tuple(payload.get("arm_names", ())) == matrix.ARM_NAMES
        and payload.get("expected_outcome_rows") == matrix.EXPECTED_OUTCOME_ROWS,
        "Integrity frozen design drifted.",
    )
    inventory = payload.get("bundle_inventory")
    _require(isinstance(inventory, list), "Integrity bundle inventory is invalid.")
    inventory = cast(list[Mapping[str, Any]], inventory)
    _require(
        len(inventory) == payload.get("validated_shards") == matrix.EXPECTED_SHARDS,
        "Integrity bundle inventory is incomplete.",
    )
    _require(
        payload.get("validated_examples") == matrix.EXPECTED_SHARDS * matrix.EXAMPLES_PER_SHARD
        and payload.get("validated_outcome_rows") == matrix.EXPECTED_OUTCOME_ROWS,
        "Integrity example/outcome cardinality drifted.",
    )
    _require(
        payload.get("bundle_inventory_digest") == contract.json_digest(inventory),
        "Integrity bundle inventory digest drifted.",
    )
    expected_coordinates = matrix.coordinates()
    pass_count = 0
    fail_count = 0
    for item, coordinate in zip(inventory, expected_coordinates, strict=True):
        _require(isinstance(item, Mapping), "Integrity inventory row is invalid.")
        _require(
            item.get("coordinate") == coordinate.payload, "Integrity coordinate order drifted."
        )
        _require(item.get("coordinate_key") == coordinate.key, "Integrity coordinate key drifted.")
        _require(contract.is_sha256(item.get("launch_nonce")), "Integrity launch nonce is invalid.")
        decision = item.get("integrity_decision")
        _require(
            decision in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
            "Integrity inventory decision is invalid.",
        )
        pass_count += decision == "INTEGRITY-PASS"
        fail_count += decision == "INTEGRITY-FAIL"
        row_counts = item.get("row_counts")
        _require(isinstance(row_counts, Mapping), "Integrity row-count binding is missing.")
        _require(
            cast(Mapping[str, Any], row_counts).get("examples") == matrix.EXAMPLES_PER_SHARD
            and cast(Mapping[str, Any], row_counts).get("outcomes")
            == matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
            "Integrity 20-by-19 evidence binding drifted.",
        )
        envelope = item.get("envelope")
        sidecars = item.get("sidecars")
        _require(isinstance(envelope, Mapping), "Integrity envelope binding is missing.")
        _require(isinstance(sidecars, Mapping), "Integrity sidecar binding is missing.")
        envelope_map = cast(Mapping[str, Any], envelope)
        sidecars_map = cast(Mapping[str, Any], sidecars)
        _require(
            item.get("bundle_digest")
            == contract.json_digest(
                {"envelope": dict(envelope_map), "sidecars": dict(sidecars_map)}
            ),
            "Integrity bundle digest drifted.",
        )
        matrix_record = item.get("matrix_record")
        _require(isinstance(matrix_record, Mapping), "Integrity matrix-record binding is missing.")
        _require(
            dict(item) == _inventory_record(cast(Mapping[str, Any], matrix_record)),
            "Integrity inventory row differs from its bound matrix record.",
        )
    _require(
        payload.get("validated_integrity_pass_shards") == pass_count
        and payload.get("validated_integrity_fail_shards") == fail_count,
        "Integrity decision counts drifted.",
    )
    expected_status = "INTEGRITY-PASS" if fail_count == 0 else "INTEGRITY-FAIL"
    _require(payload.get("integrity_status") == expected_status, "Integrity status drifted.")
    checks = payload.get("checks")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == set(INTEGRITY_CHECK_FIELDS)
        and all(checks[field] is True for field in INTEGRITY_CHECK_FIELDS),
        "Integrity audit checks are incomplete or drifted.",
    )
    if verify_bindings:
        matrix_payload = _load_json_nofollow(matrix_summary, label="bound controller matrix")
        matrix._verify_attested_payload(matrix_payload, trust_root=active_trust_root)
        _require(
            payload.get("matrix") == _file_binding(matrix_summary, payload=matrix_payload),
            "Bound controller matrix changed after audit.",
        )
        _require(
            payload.get("source") == matrix_payload.get("source")
            and payload.get("manifest") == matrix_payload.get("manifest")
            and payload.get("storage") == matrix_payload.get("storage")
            and payload.get("integrity_status") == matrix_payload.get("integrity_status"),
            "Integrity metadata differs from the HMAC-bound matrix.",
        )
        matrix_records = matrix_payload.get("records")
        _require(
            isinstance(matrix_records, list)
            and matrix_payload.get("status") == "terminal"
            and len(matrix_records) == matrix.EXPECTED_SHARDS,
            "Bound controller matrix is not a terminal complete ledger.",
        )
        matrix_record_list = cast(list[Any], matrix_records)
        _require(
            all(
                item.get("matrix_record") == record
                for item, record in zip(inventory, matrix_record_list, strict=True)
            ),
            "Integrity inventory differs from the HMAC-bound matrix records.",
        )
        matrix._preflight_output_tree(
            output_root=output_root,
            matrix_summary=matrix_summary,
            completed_shards=matrix.EXPECTED_SHARDS,
        )
    if restream_raw:
        for _matrix_record, _envelope in iter_validated_raw_shards(
            payload, trust_root=active_trust_root
        ):
            pass
    return dict(payload)


def validate_closed_world_bindings(
    payload: Mapping[str, Any],
    *,
    matrix_summary: Path = matrix.MATRIX_SUMMARY,
    output_root: Path = matrix.OUTPUT_ROOT,
    trust_root: attestation.TrustRoot | None = None,
) -> dict[str, Any]:
    """Validate HMAC, exact matrix binding, and closed-world layout without raw re-streaming."""

    return validate_integrity_artifact(
        payload,
        matrix_summary=matrix_summary,
        output_root=output_root,
        trust_root=trust_root,
        verify_bindings=True,
        restream_raw=False,
    )


def load_validated_integrity_artifact(
    path: Path = INTEGRITY_OUTPUT,
    *,
    matrix_summary: Path = matrix.MATRIX_SUMMARY,
    output_root: Path = matrix.OUTPUT_ROOT,
    trust_root: attestation.TrustRoot | None = None,
    verify_bindings: bool = True,
    restream_raw: bool = True,
) -> dict[str, Any]:
    payload = _load_json_nofollow(path, label="direct-controller integrity artifact")
    return validate_integrity_artifact(
        payload,
        matrix_summary=matrix_summary,
        output_root=output_root,
        trust_root=trust_root,
        verify_bindings=verify_bindings,
        restream_raw=restream_raw,
    )


def iter_validated_raw_shards(
    integrity_payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    outcome_callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    token_callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    failure_callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield envelopes after one-pass validation and optional row consumption.

    When callbacks are supplied, all three are required.  They receive the exact
    HMAC-bound matrix record followed by one validated raw row.  Callers must defer
    publication of callback-derived state until iteration completes successfully.
    """

    inventory = integrity_payload.get("bundle_inventory")
    _require(isinstance(inventory, list), "Integrity bundle inventory is invalid.")
    callbacks = (outcome_callback, token_callback, failure_callback)
    _require(
        all(callback is None for callback in callbacks)
        or all(callback is not None for callback in callbacks),
        "Integrity raw iteration requires all three row callbacks or none.",
    )
    evaluator = _evaluator_module()
    use_callbacks = outcome_callback is not None
    loader = getattr(evaluator, "load_validated_direct_controller_shard", None)
    consumer = getattr(evaluator, "consume_validated_direct_controller_shard", None)
    cache_type = getattr(evaluator, "DirectControllerExternalValidationCache", None)
    validation_cache: Any = None
    if use_callbacks:
        _require(callable(consumer), "Direct evaluator one-pass consumer is unavailable.")
        _require(callable(cache_type), "Direct evaluator external-validation cache is unavailable.")
        validation_cache = cast(Callable[[], Any], cache_type)()
    else:
        _require(callable(loader), "Direct evaluator safe loader is unavailable.")

    def no_op(_record: Mapping[str, Any], _row: Mapping[str, Any]) -> None:
        return None

    on_outcome = no_op if outcome_callback is None else outcome_callback
    on_token = no_op if token_callback is None else token_callback
    on_failure = no_op if failure_callback is None else failure_callback
    for raw in cast(list[Mapping[str, Any]], inventory):
        envelope_binding = raw.get("envelope")
        _require(isinstance(envelope_binding, Mapping), "Integrity envelope binding is missing.")
        envelope_binding_map = cast(Mapping[str, Any], envelope_binding)
        path = Path(cast(str, envelope_binding_map["path"]))
        matrix_record = raw.get("matrix_record")
        _require(isinstance(matrix_record, Mapping), "Integrity matrix-record binding is missing.")
        matrix_record_map = cast(Mapping[str, Any], matrix_record)
        if use_callbacks:
            envelope = cast(Any, consumer)(
                path,
                outcome_callback=lambda row, record=matrix_record_map: on_outcome(record, row),
                token_callback=lambda row, record=matrix_record_map: on_token(record, row),
                failure_callback=lambda row, record=matrix_record_map: on_failure(record, row),
                external_validation_cache=validation_cache,
                trust_root=trust_root,
            )
        else:
            envelope = cast(Any, loader)(path, verify_bindings=True, trust_root=trust_root)
        _require(
            isinstance(envelope, Mapping)
            and _file_binding(path, payload=cast(Mapping[str, Any], envelope))
            == dict(envelope_binding_map),
            "Integrity raw envelope binding drifted.",
        )
        yield dict(matrix_record_map), cast(dict[str, Any], envelope)
    if use_callbacks:
        finalizer = getattr(validation_cache, "assert_unchanged", None)
        _require(callable(finalizer), "External-validation cache finalizer is unavailable.")
        cast(Callable[..., Any], finalizer)(trust_root=trust_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-world integrity audit for the terminal direct-controller matrix."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-output-root", type=Path, default=matrix.TRAINING_OUTPUT_ROOT)
    parser.add_argument(
        "--calibration-output-root", type=Path, default=matrix.CALIBRATION_OUTPUT_ROOT
    )
    parser.add_argument("--top-p-output-root", type=Path, default=matrix.TOP_P_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=matrix.OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path, default=matrix.MATRIX_SUMMARY)
    parser.add_argument("--output", type=Path, default=INTEGRITY_OUTPUT)
    args = parser.parse_args()
    result = audit_matrix(
        manifest_path=args.manifest,
        training_output_root=args.training_output_root,
        calibration_output_root=args.calibration_output_root,
        top_p_output_root=args.top_p_output_root,
        output_root=args.output_root,
        matrix_summary=args.matrix_summary,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "integrity_status": result["integrity_status"],
                "validated_shards": result["validated_shards"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["integrity_status"] == "INTEGRITY-PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
