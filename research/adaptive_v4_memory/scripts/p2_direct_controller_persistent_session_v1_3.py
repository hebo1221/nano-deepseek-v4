from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import p2_direct_attestation as attestation
import p2_direct_controller_contract_v1_3 as contract

SCHEMA_VERSION = 1
PLAN_MESSAGE_TYPE = contract.PERSISTENT_SESSION_PLAN_MESSAGE_TYPE
WORK_MESSAGE_TYPE = contract.PERSISTENT_SESSION_WORK_MESSAGE_TYPE
RESULT_MESSAGE_TYPE = contract.PERSISTENT_SESSION_RESULT_MESSAGE_TYPE
RECEIPT_MESSAGE_TYPE = contract.PERSISTENT_SESSION_RECEIPT_MESSAGE_TYPE
LAUNCH_ARTIFACT_TYPE = contract.PERSISTENT_SESSION_LAUNCH_ARTIFACT_TYPE
TERMINAL_ARTIFACT_TYPE = contract.PERSISTENT_SESSION_TERMINAL_ARTIFACT_TYPE

PLAN_ATTESTATION_PURPOSE = contract.V1_3_3_PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE
WORK_ATTESTATION_PURPOSE = contract.V1_3_3_PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE
RESULT_ATTESTATION_PURPOSE = contract.V1_3_3_PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE
RECEIPT_ATTESTATION_PURPOSE = contract.V1_3_3_PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE
LAUNCH_LEDGER_ATTESTATION_PURPOSE = (
    contract.V1_3_3_PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE
)
TERMINAL_LEDGER_ATTESTATION_PURPOSE = (
    contract.V1_3_3_PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE
)
_CANONICAL_OUTPUT_ROOT = contract.V1_3_3_OUTPUT_ROOT
_CANONICAL_SESSION_LEDGER_ROOT = contract.V1_3_3_PERSISTENT_SESSION_LEDGER_ROOT
_CANONICAL_SESSION_LEDGER_LOCK_PATH = (
    contract.V1_3_3_PERSISTENT_SESSION_LEDGER_LOCK_PATH
)
SESSION_LEDGER_ROOT_SUFFIX = _CANONICAL_SESSION_LEDGER_ROOT.name.removeprefix(
    f".{_CANONICAL_OUTPUT_ROOT.name}."
)
_require_suffix = f".{_CANONICAL_OUTPUT_ROOT.name}.{SESSION_LEDGER_ROOT_SUFFIX}"
if _CANONICAL_SESSION_LEDGER_ROOT.name != _require_suffix:
    raise RuntimeError("Canonical v1.3.3 persistent-session ledger layout drifted.")
del _require_suffix

MAXIMUM_PLAN_BYTES = 4 << 20
MAXIMUM_JSONL_MESSAGE_BYTES = 1 << 20
MAXIMUM_LEDGER_BYTES = 8 << 20
CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT = 0
MODEL_LOADS_PER_SESSION = 1
READY_ONLY_PREFLIGHT_SESSION_ROLE = contract.V1_3_3_READY_ONLY_PREFLIGHT_SESSION_ROLE
QUALITY_SESSION_ROLE = contract.V1_3_3_QUALITY_SESSION_ROLE
if READY_ONLY_PREFLIGHT_SESSION_ROLE == QUALITY_SESSION_ROLE:
    raise RuntimeError("Ready-only and quality persistent session roles must be distinct.")
SESSION_ROLES = frozenset(
    {READY_ONLY_PREFLIGHT_SESSION_ROLE, QUALITY_SESSION_ROLE}
)

# Frozen planning constants. Durable receipts below distinguish launch-time
# upper bounds from model loads that actually reached the ready boundary.
NORMAL_PATH_UNIQUE_MODEL_COHORTS = (
    contract.V1_3_3_QUALITY_SESSION_NORMAL_PATH_MODEL_LOADS
)
READY_ONLY_PREFLIGHT_MODEL_LOAD_UPPER_BOUND = (
    contract.V1_3_3_READY_ONLY_PREFLIGHT_MODEL_LOADS
)
SINGLE_WORKER_TOTAL_MODEL_LOAD_UPPER_BOUND = (
    contract.V1_3_3_TOTAL_NORMAL_PATH_CHECKPOINT_MODEL_LOADS
)
if SINGLE_WORKER_TOTAL_MODEL_LOAD_UPPER_BOUND != (
    READY_ONLY_PREFLIGHT_MODEL_LOAD_UPPER_BOUND + NORMAL_PATH_UNIQUE_MODEL_COHORTS
):
    raise RuntimeError("Ready-only plus quality model-load bounds do not sum exactly.")
NORMAL_PATH_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES = 11_810_258_620
LEGACY_PER_SHARD_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES = 10_629_232_758_000
THEORETICAL_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_REDUCTION_FACTOR = 900

_COORDINATE_FIELDS = frozenset(
    {
        "scale",
        "training_seed",
        "calibration_seed",
        "evaluation_seed",
        "budget",
        "family",
        "context",
        "replicate",
        "generation_seed",
    }
)


def _normal_path_claim_semantics(
    *,
    launch_count: int,
    terminal_count: int,
    graceful_terminal_count: int,
    controlled_stop_count: int,
    cohort_count: int,
    launch_authority_count: int,
    worker_counts: Sequence[int],
) -> tuple[int, bool]:
    additional_attempts = max(
        max(0, launch_count - cohort_count),
        max(0, launch_authority_count - 1),
    )
    applicable = (
        terminal_count == launch_count
        and graceful_terminal_count == launch_count
        and controlled_stop_count == 0
        and additional_attempts == 0
        and (launch_count == 0 or launch_authority_count == 1)
        and all(worker_count == 1 for worker_count in worker_counts)
    )
    return additional_attempts, applicable


_PLAN_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "message_type",
        "session_nonce",
        "launch_authority_nonce",
        "session_role",
        "worker_index",
        "worker_count",
        "scale",
        "training_seed",
        "coordinate_count",
        "coordinate_digest",
        "coordinates",
        "assignment_completed_prefix_count",
        "assignment_completed_prefix_digest",
        "max_new_cells_stop_limit",
        "coordinate_selection_rule",
        "input_binding_digest",
        "canonical_evaluator_digest",
        "gpu_lease_binding_digest",
        "prerequisites_binding_digest",
        "output_root",
        "model_load_limit",
        "child_full_historical_evidence_replay_count",
        "outcome_dependent_selection",
    }
)
_WORK_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "message_type",
        "session_nonce",
        "launch_authority_nonce",
        "plan_payload_sha256",
        "sequence_index",
        "coordinate",
        "launch_nonce",
        "envelope_path",
        "projected_remaining_shards",
        "projected_remaining_token_rows",
    }
)
_RESULT_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "message_type",
        "session_nonce",
        "launch_authority_nonce",
        "plan_payload_sha256",
        "work_payload_sha256",
        "worker_index",
        "worker_count",
        "scale",
        "training_seed",
        "planned_coordinate_count",
        "planned_coordinate_digest",
        "sequence_index",
        "coordinate",
        "launch_nonce",
        "terminal_decision",
        "envelope_binding",
        "completed_in_session",
        "model_load_count",
        "child_full_historical_evidence_replay_count",
        "active_activation_validation_count",
        "active_admission_validation_count",
        "active_genesis_validation_count",
        "active_calibration_validation_count",
        "active_checkpoint_validation_count",
        "model_state_reset_count",
    }
)
_RECEIPT_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "message_type",
        "session_nonce",
        "launch_authority_nonce",
        "plan_payload_sha256",
        "status",
        "planned_coordinates",
        "completed_coordinates",
        "completed_work_payload_sha256",
        "completed_result_payload_sha256",
        "model_load_count",
        "child_full_historical_evidence_replay_count",
        "active_activation_validation_count",
        "active_admission_validation_count",
        "active_genesis_validation_count",
        "active_calibration_validation_count",
        "active_checkpoint_validation_count",
        "model_state_reset_count",
        "outcome_dependent_selection",
    }
)
_ATTESTED_SUFFIX_FIELDS = frozenset({"payload_sha256", "attestation"})
_LAUNCH_LEDGER_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "session_nonce",
        "launch_authority_nonce",
        "plan",
        "actual_session_argv",
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count",
        "session_triggered_full_historical_evidence_replay_count",
        "checkpoint_model_load_attempt_upper_bound",
        "outcome_dependent_selection",
    }
)
_TERMINAL_LEDGER_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "session_nonce",
        "launch_authority_nonce",
        "plan_payload_sha256",
        "status",
        "ready_receipt",
        "final_receipt",
        "completed_work_payload_sha256",
        "completed_result_payload_sha256",
        "published_bundle_reingestion_count",
        "actual_session_argv",
        "child_process_returncode",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _validate_actual_session_argv(value: Any, *, plan: Mapping[str, Any]) -> list[str]:
    _require(
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value),
        "Persistent actual session argv is invalid.",
    )
    result = cast(list[str], value)
    launch_indices = [index for index, item in enumerate(result) if item == "--launch-nonce"]
    scale_indices = [index for index, item in enumerate(result) if item == "--scale"]
    seed_indices = [index for index, item in enumerate(result) if item == "--training-seed"]
    _require(
        result.count("--persistent-session") == 1
        and len(launch_indices) == len(scale_indices) == len(seed_indices) == 1
        and launch_indices[0] + 1 < len(result)
        and scale_indices[0] + 1 < len(result)
        and seed_indices[0] + 1 < len(result)
        and result[launch_indices[0] + 1] == plan.get("session_nonce")
        and result[scale_indices[0] + 1] == plan.get("scale")
        and result[seed_indices[0] + 1] == str(plan.get("training_seed")),
        "Persistent actual session argv differs from its sealed plan.",
    )
    return list(result)


@lru_cache(maxsize=1)
def _canonical_coordinate_index() -> dict[str, tuple[int, dict[str, int | str]]]:
    result: dict[str, tuple[int, dict[str, int | str]]] = {}
    for index, raw in enumerate(contract.quality_coordinates()):
        coordinate = dict(raw)
        digest = contract.json_digest(coordinate)
        _require(digest not in result, "Frozen quality coordinate digest repeated.")
        result[digest] = (index, coordinate)
    _require(
        len(result) == contract.BUDGET_SHARDS_TOTAL == 9_000,
        "Frozen quality coordinate inventory drifted.",
    )
    return result


def validate_coordinate(value: Mapping[str, Any]) -> dict[str, int | str]:
    _require(set(value) == _COORDINATE_FIELDS, "Persistent coordinate schema drifted.")
    candidate = _json_clone(dict(value))
    digest = contract.json_digest(candidate)
    frozen = _canonical_coordinate_index().get(digest)
    _require(frozen is not None and candidate == frozen[1], "Unknown persistent coordinate.")
    checked = cast(tuple[int, dict[str, int | str]], frozen)
    return dict(checked[1])


def coordinate_global_index(value: Mapping[str, Any]) -> int:
    coordinate = validate_coordinate(value)
    return _canonical_coordinate_index()[contract.json_digest(coordinate)][0]


def coordinate_cohort(value: Mapping[str, Any]) -> tuple[str, int]:
    coordinate = validate_coordinate(value)
    return cast(str, coordinate["scale"]), cast(int, coordinate["training_seed"])


def assigned_coordinates(*, worker_index: int, worker_count: int) -> tuple[dict[str, Any], ...]:
    _require(
        type(worker_count) is int and 1 <= worker_count <= contract.BUDGET_SHARDS_TOTAL,
        "Persistent worker count is invalid.",
    )
    _require(
        type(worker_index) is int and 0 <= worker_index < worker_count,
        "Persistent worker index is invalid.",
    )
    return tuple(
        dict(coordinate)
        for index, coordinate in enumerate(contract.quality_coordinates())
        if index % worker_count == worker_index
    )


def unique_worker_scale_seed_assignments(
    coordinates: Sequence[Mapping[str, Any]],
    *,
    worker_index: int,
    worker_count: int,
) -> tuple[tuple[int, str, int], ...]:
    allowed = {
        contract.json_digest(item)
        for item in assigned_coordinates(worker_index=worker_index, worker_count=worker_count)
    }
    cohorts: set[tuple[int, str, int]] = set()
    last_index = -1
    for raw in coordinates:
        coordinate = validate_coordinate(raw)
        digest = contract.json_digest(coordinate)
        _require(digest in allowed, "Coordinate is outside its persistent worker assignment.")
        index = coordinate_global_index(coordinate)
        _require(index > last_index, "Persistent coordinates are not in canonical order.")
        last_index = index
        scale, training_seed = coordinate_cohort(coordinate)
        cohorts.add((worker_index, scale, training_seed))
    return tuple(sorted(cohorts))


def normal_model_load_upper_bound(
    coordinates: Sequence[Mapping[str, Any]],
    *,
    worker_index: int,
    worker_count: int,
) -> int:
    return len(
        unique_worker_scale_seed_assignments(
            coordinates,
            worker_index=worker_index,
            worker_count=worker_count,
        )
    )


def _attested_message(
    source: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    purpose: str,
) -> dict[str, Any]:
    cloned = _json_clone(dict(source))
    digest_bound = {**cloned, "payload_sha256": contract.json_digest(cloned)}
    return {
        **digest_bound,
        "attestation": attestation.attest_payload(
            digest_bound,
            trust_root=trust_root,
            purpose=purpose,
        ),
    }


def _verified_message(
    value: Mapping[str, Any],
    *,
    source_fields: frozenset[str],
    trust_root: attestation.TrustRoot,
    purpose: str,
) -> dict[str, Any]:
    _require(
        set(value) == source_fields | _ATTESTED_SUFFIX_FIELDS,
        "Persistent session message schema drifted.",
    )
    payload = dict(value)
    raw_attestation = payload.pop("attestation")
    _require(isinstance(raw_attestation, Mapping), "Persistent attestation is missing.")
    observed_digest = payload.pop("payload_sha256")
    _require(contract.is_sha256(observed_digest), "Persistent payload digest is invalid.")
    source = _json_clone(payload)
    _require(
        observed_digest == contract.json_digest(source),
        "Persistent payload digest drifted.",
    )
    digest_bound = {**source, "payload_sha256": observed_digest}
    attestation.verify_attestation(
        digest_bound,
        cast(Mapping[str, Any], raw_attestation),
        trust_root=trust_root,
        purpose=purpose,
    )
    return {**digest_bound, "attestation": dict(raw_attestation)}


def build_session_plan(
    coordinates: Sequence[Mapping[str, Any]],
    *,
    session_nonce: str,
    launch_authority_nonce: str,
    worker_index: int,
    worker_count: int,
    assignment_completed_prefix: Sequence[Mapping[str, Any]],
    max_new_cells_stop_limit: int | None,
    input_binding_digest: str,
    canonical_evaluator_digest: str,
    gpu_lease_binding_digest: str,
    prerequisites_binding_digest: str,
    output_root: Path,
    trust_root: attestation.TrustRoot,
    session_role: str = QUALITY_SESSION_ROLE,
) -> dict[str, Any]:
    _require(contract.is_sha256(session_nonce), "Persistent session nonce is invalid.")
    _require(
        contract.is_sha256(launch_authority_nonce),
        "Persistent launch-authority nonce is invalid.",
    )
    _require(session_role in SESSION_ROLES, "Persistent session role is invalid.")
    for digest in (
        input_binding_digest,
        canonical_evaluator_digest,
        gpu_lease_binding_digest,
        prerequisites_binding_digest,
    ):
        _require(contract.is_sha256(digest), "Persistent plan binding digest is invalid.")
    frozen = [validate_coordinate(item) for item in coordinates]
    _require(bool(frozen), "Persistent session plan is empty.")
    completed_prefix = [validate_coordinate(item) for item in assignment_completed_prefix]
    assigned = list(assigned_coordinates(worker_index=worker_index, worker_count=worker_count))
    _require(
        completed_prefix == assigned[: len(completed_prefix)],
        "Persistent completed assignment prefix is not canonical.",
    )
    _require(
        len(completed_prefix) < len(assigned),
        "Persistent completed assignment prefix has no remaining coordinate.",
    )
    _require(
        max_new_cells_stop_limit is None
        or (
            type(max_new_cells_stop_limit) is int
            and 1 <= max_new_cells_stop_limit <= len(assigned) - len(completed_prefix)
        ),
        "Persistent max-new-cells stop limit is invalid.",
    )
    first_remaining = assigned[len(completed_prefix)]
    expected: list[dict[str, Any]] = []
    for candidate in assigned[len(completed_prefix) :]:
        if coordinate_cohort(candidate) != coordinate_cohort(first_remaining):
            break
        if max_new_cells_stop_limit is not None and len(expected) >= max_new_cells_stop_limit:
            break
        expected.append(candidate)
    _require(
        frozen == expected,
        "Persistent plan is not the deterministic maximal remaining cohort prefix.",
    )
    assignments = unique_worker_scale_seed_assignments(
        frozen,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    _require(len(assignments) == 1, "Persistent session plan crosses a model cohort.")
    _worker, scale, training_seed = assignments[0]
    exact_root = Path(os.path.abspath(output_root))
    _require(
        exact_root == output_root and not output_root.is_symlink(),
        "Persistent output root is not an exact absolute path.",
    )
    source = {
        "schema_version": SCHEMA_VERSION,
        "message_type": PLAN_MESSAGE_TYPE,
        "session_nonce": session_nonce,
        "launch_authority_nonce": launch_authority_nonce,
        "session_role": session_role,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "scale": scale,
        "training_seed": training_seed,
        "coordinate_count": len(frozen),
        "coordinate_digest": contract.json_digest(frozen),
        "coordinates": frozen,
        "assignment_completed_prefix_count": len(completed_prefix),
        "assignment_completed_prefix_digest": contract.json_digest(completed_prefix),
        "max_new_cells_stop_limit": max_new_cells_stop_limit,
        "coordinate_selection_rule": (
            "maximal-remaining-canonical-worker-scale-seed-prefix-truncated-only-by-"
            "predeclared-max-new-cells-v1"
        ),
        "input_binding_digest": input_binding_digest,
        "canonical_evaluator_digest": canonical_evaluator_digest,
        "gpu_lease_binding_digest": gpu_lease_binding_digest,
        "prerequisites_binding_digest": prerequisites_binding_digest,
        "output_root": str(exact_root),
        "model_load_limit": MODEL_LOADS_PER_SESSION,
        "child_full_historical_evidence_replay_count": (
            CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
        ),
        "outcome_dependent_selection": False,
    }
    return _attested_message(
        source,
        trust_root=trust_root,
        purpose=PLAN_ATTESTATION_PURPOSE,
    )


def validate_session_plan(
    value: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    plan = _verified_message(
        value,
        source_fields=_PLAN_SOURCE_FIELDS,
        trust_root=trust_root,
        purpose=PLAN_ATTESTATION_PURPOSE,
    )
    raw_coordinates = plan.get("coordinates")
    _require(isinstance(raw_coordinates, list), "Persistent plan coordinates are invalid.")
    rebuilt = build_session_plan(
        cast(list[Mapping[str, Any]], raw_coordinates),
        session_nonce=cast(str, plan.get("session_nonce")),
        launch_authority_nonce=cast(str, plan.get("launch_authority_nonce")),
        worker_index=cast(int, plan.get("worker_index")),
        worker_count=cast(int, plan.get("worker_count")),
        assignment_completed_prefix=assigned_coordinates(
            worker_index=cast(int, plan.get("worker_index")),
            worker_count=cast(int, plan.get("worker_count")),
        )[: cast(int, plan.get("assignment_completed_prefix_count"))],
        max_new_cells_stop_limit=cast(int | None, plan.get("max_new_cells_stop_limit")),
        input_binding_digest=cast(str, plan.get("input_binding_digest")),
        canonical_evaluator_digest=cast(str, plan.get("canonical_evaluator_digest")),
        gpu_lease_binding_digest=cast(str, plan.get("gpu_lease_binding_digest")),
        prerequisites_binding_digest=cast(str, plan.get("prerequisites_binding_digest")),
        output_root=Path(cast(str, plan.get("output_root"))),
        trust_root=trust_root,
        session_role=cast(str, plan.get("session_role")),
    )
    _require(plan == rebuilt, "Persistent session plan semantics drifted.")
    return plan


def build_work_order(
    plan: Mapping[str, Any],
    *,
    sequence_index: int,
    launch_nonce: str,
    envelope_path: Path,
    projected_remaining_shards: int,
    projected_remaining_token_rows: int,
    trust_root: attestation.TrustRoot,
    require_target_absent: bool = True,
) -> dict[str, Any]:
    validated = validate_session_plan(plan, trust_root=trust_root)
    coordinates = cast(list[Mapping[str, Any]], validated["coordinates"])
    _require(
        type(sequence_index) is int and 0 <= sequence_index < len(coordinates),
        "Persistent work sequence index is invalid.",
    )
    _require(contract.is_sha256(launch_nonce), "Persistent work launch nonce is invalid.")
    _require(
        type(projected_remaining_shards) is int
        and 1 <= projected_remaining_shards <= contract.BUDGET_SHARDS_TOTAL,
        "Persistent projected shard count is invalid.",
    )
    _require(
        type(projected_remaining_token_rows) is int
        and 1
        <= projected_remaining_token_rows
        <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
        "Persistent projected token-row count is invalid.",
    )
    root = Path(cast(str, validated["output_root"]))
    exact_envelope = Path(os.path.abspath(envelope_path))
    _require(
        exact_envelope == envelope_path
        and exact_envelope.is_relative_to(root)
        and exact_envelope.suffix == ".json"
        and (not require_target_absent or not os.path.lexists(exact_envelope)),
        "Persistent envelope target is unsafe or unexpectedly pre-existing.",
    )
    source = {
        "schema_version": SCHEMA_VERSION,
        "message_type": WORK_MESSAGE_TYPE,
        "session_nonce": validated["session_nonce"],
        "launch_authority_nonce": validated["launch_authority_nonce"],
        "plan_payload_sha256": validated["payload_sha256"],
        "sequence_index": sequence_index,
        "coordinate": dict(coordinates[sequence_index]),
        "launch_nonce": launch_nonce,
        "envelope_path": str(exact_envelope),
        "projected_remaining_shards": projected_remaining_shards,
        "projected_remaining_token_rows": projected_remaining_token_rows,
    }
    return _attested_message(
        source,
        trust_root=trust_root,
        purpose=WORK_ATTESTATION_PURPOSE,
    )


def validate_work_order(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_sequence_index: int,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    validated_plan = validate_session_plan(plan, trust_root=trust_root)
    work = validate_attested_work_message(value, trust_root=trust_root)
    rebuilt = build_work_order(
        validated_plan,
        sequence_index=expected_sequence_index,
        launch_nonce=cast(str, work.get("launch_nonce")),
        envelope_path=Path(cast(str, work.get("envelope_path"))),
        projected_remaining_shards=cast(int, work.get("projected_remaining_shards")),
        projected_remaining_token_rows=cast(int, work.get("projected_remaining_token_rows")),
        trust_root=trust_root,
        require_target_absent=False,
    )
    _require(work == rebuilt, "Persistent work-order semantics drifted.")
    return work


def validate_attested_work_message(
    value: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    work = _verified_message(
        value,
        source_fields=_WORK_SOURCE_FIELDS,
        trust_root=trust_root,
        purpose=WORK_ATTESTATION_PURPOSE,
    )
    _require(
        work.get("schema_version") == SCHEMA_VERSION
        and work.get("message_type") == WORK_MESSAGE_TYPE
        and contract.is_sha256(work.get("session_nonce"))
        and contract.is_sha256(work.get("launch_authority_nonce"))
        and contract.is_sha256(work.get("plan_payload_sha256"))
        and type(work.get("sequence_index")) is int
        and cast(int, work["sequence_index"]) >= 0
        and contract.is_sha256(work.get("launch_nonce"))
        and type(work.get("projected_remaining_shards")) is int
        and 1 <= cast(int, work["projected_remaining_shards"]) <= contract.BUDGET_SHARDS_TOTAL
        and type(work.get("projected_remaining_token_rows")) is int
        and 1
        <= cast(int, work["projected_remaining_token_rows"])
        <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
        and isinstance(work.get("envelope_path"), str),
        "Persistent attested work-order values are invalid.",
    )
    validate_coordinate(cast(Mapping[str, Any], work.get("coordinate")))
    return work


def build_work_result(
    plan: Mapping[str, Any],
    work_order: Mapping[str, Any],
    *,
    terminal_decision: str,
    envelope_binding: Mapping[str, Any],
    completed_in_session: int,
    model_state_reset_count: int,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    validated_plan = validate_session_plan(plan, trust_root=trust_root)
    sequence_index = work_order.get("sequence_index")
    _require(type(sequence_index) is int, "Persistent work-result sequence is invalid.")
    work = validate_work_order(
        work_order,
        plan=validated_plan,
        expected_sequence_index=cast(int, sequence_index),
        trust_root=trust_root,
    )
    _require(
        terminal_decision in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
        "Persistent work-result decision is invalid.",
    )
    _require(
        type(completed_in_session) is int and completed_in_session == cast(int, sequence_index) + 1,
        "Persistent completed-work count is invalid.",
    )
    _require(
        type(model_state_reset_count) is int and model_state_reset_count == completed_in_session,
        "Persistent model-reset count drifted.",
    )
    binding = _json_clone(dict(envelope_binding))
    _require(
        binding.get("path") == work["envelope_path"]
        and type(binding.get("bytes")) is int
        and cast(int, binding["bytes"]) > 0
        and contract.is_sha256(binding.get("sha256"))
        and contract.is_sha256(binding.get("payload_sha256"))
        and contract.is_sha256(binding.get("attestation_mac")),
        "Persistent envelope result binding is invalid.",
    )
    source = {
        "schema_version": SCHEMA_VERSION,
        "message_type": RESULT_MESSAGE_TYPE,
        "session_nonce": validated_plan["session_nonce"],
        "launch_authority_nonce": validated_plan["launch_authority_nonce"],
        "plan_payload_sha256": validated_plan["payload_sha256"],
        "work_payload_sha256": work["payload_sha256"],
        "worker_index": validated_plan["worker_index"],
        "worker_count": validated_plan["worker_count"],
        "scale": validated_plan["scale"],
        "training_seed": validated_plan["training_seed"],
        "planned_coordinate_count": validated_plan["coordinate_count"],
        "planned_coordinate_digest": validated_plan["coordinate_digest"],
        "sequence_index": sequence_index,
        "coordinate": work["coordinate"],
        "launch_nonce": work["launch_nonce"],
        "terminal_decision": terminal_decision,
        "envelope_binding": binding,
        "completed_in_session": completed_in_session,
        "model_load_count": MODEL_LOADS_PER_SESSION,
        "child_full_historical_evidence_replay_count": (
            CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
        ),
        "active_activation_validation_count": 1,
        "active_admission_validation_count": 1,
        "active_genesis_validation_count": 1,
        "active_calibration_validation_count": 1,
        "active_checkpoint_validation_count": 1,
        "model_state_reset_count": model_state_reset_count,
    }
    return _attested_message(
        source,
        trust_root=trust_root,
        purpose=RESULT_ATTESTATION_PURPOSE,
    )


def validate_work_result(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    work_order: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    result = validate_attested_work_result(value, trust_root=trust_root)
    rebuilt = build_work_result(
        plan,
        work_order,
        terminal_decision=cast(str, result.get("terminal_decision")),
        envelope_binding=cast(Mapping[str, Any], result.get("envelope_binding")),
        completed_in_session=cast(int, result.get("completed_in_session")),
        model_state_reset_count=cast(int, result.get("model_state_reset_count")),
        trust_root=trust_root,
    )
    _require(result == rebuilt, "Persistent work-result semantics drifted.")
    return result


def validate_attested_work_result(
    value: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    result = _verified_message(
        value,
        source_fields=_RESULT_SOURCE_FIELDS,
        trust_root=trust_root,
        purpose=RESULT_ATTESTATION_PURPOSE,
    )
    coordinate = validate_coordinate(cast(Mapping[str, Any], result.get("coordinate")))
    _require(
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("message_type") == RESULT_MESSAGE_TYPE
        and contract.is_sha256(result.get("session_nonce"))
        and contract.is_sha256(result.get("launch_authority_nonce"))
        and contract.is_sha256(result.get("plan_payload_sha256"))
        and contract.is_sha256(result.get("work_payload_sha256"))
        and result.get("scale") == coordinate["scale"]
        and result.get("training_seed") == coordinate["training_seed"]
        and result.get("model_load_count") == MODEL_LOADS_PER_SESSION
        and result.get("child_full_historical_evidence_replay_count")
        == CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
        and all(
            result.get(field) == 1
            for field in (
                "active_admission_validation_count",
                "active_activation_validation_count",
                "active_genesis_validation_count",
                "active_calibration_validation_count",
                "active_checkpoint_validation_count",
            )
        )
        and type(result.get("completed_in_session")) is int
        and result.get("model_state_reset_count") == result.get("completed_in_session"),
        "Persistent attested work-result values are invalid.",
    )
    return result


def build_session_receipt(
    plan: Mapping[str, Any],
    work_orders: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    validated_plan = validate_session_plan(plan, trust_root=trust_root)
    _require(len(work_orders) == len(results), "Persistent receipt work/result count drifted.")
    validated_work: list[dict[str, Any]] = []
    validated_results: list[dict[str, Any]] = []
    for sequence_index, (work, result) in enumerate(zip(work_orders, results, strict=True)):
        checked_work = validate_work_order(
            work,
            plan=validated_plan,
            expected_sequence_index=sequence_index,
            trust_root=trust_root,
        )
        checked_result = validate_work_result(
            result,
            plan=validated_plan,
            work_order=checked_work,
            trust_root=trust_root,
        )
        validated_work.append(checked_work)
        validated_results.append(checked_result)
    completed = len(validated_results)
    source = {
        "schema_version": SCHEMA_VERSION,
        "message_type": RECEIPT_MESSAGE_TYPE,
        "session_nonce": validated_plan["session_nonce"],
        "launch_authority_nonce": validated_plan["launch_authority_nonce"],
        "plan_payload_sha256": validated_plan["payload_sha256"],
        "status": ("complete" if completed == validated_plan["coordinate_count"] else "stopped"),
        "planned_coordinates": validated_plan["coordinate_count"],
        "completed_coordinates": completed,
        "completed_work_payload_sha256": [item["payload_sha256"] for item in validated_work],
        "completed_result_payload_sha256": [item["payload_sha256"] for item in validated_results],
        "model_load_count": MODEL_LOADS_PER_SESSION,
        "child_full_historical_evidence_replay_count": (
            CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
        ),
        "active_activation_validation_count": 1,
        "active_admission_validation_count": 1,
        "active_genesis_validation_count": 1,
        "active_calibration_validation_count": 1,
        "active_checkpoint_validation_count": 1,
        "model_state_reset_count": completed,
        "outcome_dependent_selection": False,
    }
    return _attested_message(
        source,
        trust_root=trust_root,
        purpose=RECEIPT_ATTESTATION_PURPOSE,
    )


def validate_session_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    work_orders: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    receipt = _verified_message(
        value,
        source_fields=_RECEIPT_SOURCE_FIELDS,
        trust_root=trust_root,
        purpose=RECEIPT_ATTESTATION_PURPOSE,
    )
    rebuilt = build_session_receipt(
        plan,
        work_orders,
        results,
        trust_root=trust_root,
    )
    _require(receipt == rebuilt, "Persistent session receipt semantics drifted.")
    return receipt


def validate_attested_session_receipt(
    value: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    receipt = _verified_message(
        value,
        source_fields=_RECEIPT_SOURCE_FIELDS,
        trust_root=trust_root,
        purpose=RECEIPT_ATTESTATION_PURPOSE,
    )
    _require(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("message_type") == RECEIPT_MESSAGE_TYPE
        and contract.is_sha256(receipt.get("session_nonce"))
        and contract.is_sha256(receipt.get("launch_authority_nonce"))
        and contract.is_sha256(receipt.get("plan_payload_sha256"))
        and receipt.get("status") in {"complete", "stopped"}
        and receipt.get("model_load_count") == MODEL_LOADS_PER_SESSION
        and receipt.get("child_full_historical_evidence_replay_count")
        == CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
        and receipt.get("outcome_dependent_selection") is False,
        "Persistent attested session receipt values are invalid.",
    )
    return receipt


def efficiency_counters(
    plans: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require(len(plans) == len(receipts), "Persistent efficiency inventory is incomplete.")
    sessions: set[str] = set()
    assignments: set[tuple[int, str, int]] = set()
    model_loads = 0
    child_replays = 0
    for raw_plan, raw_receipt in zip(plans, receipts, strict=True):
        plan = validate_session_plan(raw_plan, trust_root=trust_root)
        receipt = _verified_message(
            raw_receipt,
            source_fields=_RECEIPT_SOURCE_FIELDS,
            trust_root=trust_root,
            purpose=RECEIPT_ATTESTATION_PURPOSE,
        )
        _require(
            receipt.get("plan_payload_sha256") == plan["payload_sha256"]
            and receipt.get("session_nonce") == plan["session_nonce"]
            and receipt.get("launch_authority_nonce") == plan["launch_authority_nonce"],
            "Persistent efficiency receipt/plan binding drifted.",
        )
        session_nonce = cast(str, plan["session_nonce"])
        _require(session_nonce not in sessions, "Persistent session nonce repeated.")
        sessions.add(session_nonce)
        assignments.add(
            (
                cast(int, plan["worker_index"]),
                cast(str, plan["scale"]),
                cast(int, plan["training_seed"]),
            )
        )
        model_loads += cast(int, receipt["model_load_count"])
        child_replays += cast(int, receipt["child_full_historical_evidence_replay_count"])
    bound = len(assignments)
    return {
        "persistent_session_count": len(sessions),
        "unique_worker_scale_seed_assignments": bound,
        "model_load_count": model_loads,
        "normal_model_load_upper_bound": bound,
        "normal_model_load_bound_satisfied": model_loads <= bound,
        "child_full_historical_evidence_replay_count": child_replays,
    }


def encode_jsonl_message(value: Mapping[str, Any]) -> bytes:
    encoded = attestation.canonical_json(value) + b"\n"
    _require(
        len(encoded) <= MAXIMUM_JSONL_MESSAGE_BYTES,
        "Persistent JSONL message exceeds its size limit.",
    )
    return encoded


def decode_jsonl_message(value: bytes) -> dict[str, Any]:
    _require(
        bool(value)
        and len(value) <= MAXIMUM_JSONL_MESSAGE_BYTES
        and value.endswith(b"\n")
        and b"\n" not in value[:-1],
        "Persistent JSONL message framing is invalid.",
    )
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Persistent JSONL message is invalid JSON.") from error
    _require(isinstance(parsed, dict), "Persistent JSONL message is not an object.")
    _require(
        encode_jsonl_message(parsed) == value,
        "Persistent JSONL message is not canonical.",
    )
    return cast(dict[str, Any], parsed)


def create_sealed_plan_fd(plan: Mapping[str, Any]) -> int:
    encoded = attestation.canonical_json(plan)
    _require(len(encoded) <= MAXIMUM_PLAN_BYTES, "Persistent plan exceeds its size limit.")
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        callable(create) and allow_sealing is not None,
        "Persistent plan transport requires sealed memfd support.",
    )
    descriptor = cast(Any, create)(
        "adaptive-v4-direct-exact-fill-v1-3-3-persistent-plan",
        cast(int, getattr(os, "MFD_CLOEXEC", 0)) | cast(int, allow_sealing),
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            _require(written > 0, "Persistent plan memfd write stalled.")
            offset += written
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        _require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
            "Persistent plan memfd was not sealed.",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_sealed_plan_fd(descriptor: int, *, trust_root: attestation.TrustRoot) -> dict[str, Any]:
    _require(type(descriptor) is int and descriptor >= 0, "Persistent plan FD is invalid.")
    metadata = os.fstat(descriptor)
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    try:
        observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    except OSError as error:
        raise ValueError("Persistent plan FD is not an exact sealed regular file.") from error
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size <= MAXIMUM_PLAN_BYTES
        and observed_seals & seals == seals,
        "Persistent plan FD is not an exact sealed regular file.",
    )
    data = os.pread(descriptor, metadata.st_size + 1, 0)
    _require(len(data) == metadata.st_size, "Persistent plan FD bytes drifted.")
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Persistent plan FD is invalid JSON.") from error
    _require(
        isinstance(parsed, dict) and attestation.canonical_json(parsed) == data,
        "Persistent plan FD is not canonical.",
    )
    return validate_session_plan(cast(Mapping[str, Any], parsed), trust_root=trust_root)


def session_ledger_root(output_root: Path) -> Path:
    root = Path(os.path.abspath(output_root))
    _require(
        root == output_root
        and not output_root.is_symlink()
        and (not os.path.lexists(root) or root.resolve(strict=True) == root),
        "Persistent session output root is not exact.",
    )
    canonical_output_root = Path(os.path.abspath(_CANONICAL_OUTPUT_ROOT))
    if root == canonical_output_root:
        canonical_ledger_root = Path(os.path.abspath(_CANONICAL_SESSION_LEDGER_ROOT))
        _require(
            canonical_ledger_root.parent == root.parent,
            "Canonical v1.3.3 persistent-session ledger root drifted.",
        )
        return canonical_ledger_root
    return root.parent / f".{root.name}.{SESSION_LEDGER_ROOT_SUFFIX}"


def _require_plan_output_root(plan: Mapping[str, Any], *, output_root: Path) -> Path:
    exact = Path(os.path.abspath(output_root))
    _require(
        exact == output_root
        and plan.get("output_root") == str(exact)
        and session_ledger_root(exact)
        == exact.parent / f".{exact.name}.{SESSION_LEDGER_ROOT_SUFFIX}",
        "Persistent session plan/output-root cross-binding drifted.",
    )
    return exact


def session_ledger_lock_path(output_root: Path) -> Path:
    root = session_ledger_root(output_root)
    canonical_output_root = Path(os.path.abspath(_CANONICAL_OUTPUT_ROOT))
    if Path(os.path.abspath(output_root)) == canonical_output_root:
        canonical_lock = Path(os.path.abspath(_CANONICAL_SESSION_LEDGER_LOCK_PATH))
        _require(
            canonical_lock == root.parent / f"{root.name}.lock",
            "Canonical v1.3.3 persistent-session ledger lock drifted.",
        )
        return canonical_lock
    return root.parent / f"{root.name}.lock"


def _session_ledger_path(output_root: Path, session_nonce: str, kind: str) -> Path:
    _require(contract.is_sha256(session_nonce), "Persistent ledger session nonce is invalid.")
    _require(kind in {"launch", "terminal"}, "Persistent ledger kind is invalid.")
    return session_ledger_root(output_root) / f"{session_nonce}.{kind}.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _session_ledger_lock(
    root: Path, *, create: bool, exclusive_create: bool = False
) -> Iterator[None]:
    lock_path = root.parent / f"{root.name}.lock"
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _require(nofollow is not None, "Persistent session ledger locking requires O_NOFOLLOW.")
    flags = (
        os.O_RDWR
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | cast(int, nofollow)
    )
    if create:
        flags |= os.O_CREAT
    if exclusive_create:
        _require(create, "Exclusive persistent lock creation requires create mode.")
        flags |= os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileNotFoundError as error:
        raise ValueError(
            "Persistent session ledger lock is missing for an existing ledger root."
        ) from error
    try:
        if exclusive_create:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(lock_path.parent)
        opened = os.fstat(descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        _require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Persistent session ledger lock is unsafe.",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held = os.fstat(descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        _require(
            (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino) and held.st_nlink == 1,
            "Persistent session ledger lock changed while held.",
        )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _publish_json_exclusive_locked(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root = path.parent
    root_created = False
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        root_created = True
    except FileExistsError:
        pass
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _require(nofollow is not None, "Persistent ledger publication requires O_NOFOLLOW.")
    root_descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | cast(int, nofollow),
    )
    try:
        if root_created:
            os.fchmod(root_descriptor, 0o700)
        opened_root = os.fstat(root_descriptor)
        metadata = os.stat(root, follow_symlinks=False)
        _require(
            stat.S_ISDIR(opened_root.st_mode)
            and (opened_root.st_dev, opened_root.st_ino)
            == (metadata.st_dev, metadata.st_ino)
            and opened_root.st_uid == metadata.st_uid == os.getuid()
            and stat.S_IMODE(opened_root.st_mode)
            == stat.S_IMODE(metadata.st_mode)
            == 0o700,
            "Persistent session ledger root is unsafe.",
        )
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)
    # This is deliberately unconditional: it also durably adopts a safe root
    # left by a crash between mkdir and the original parent-directory fsync.
    _fsync_directory(root.parent)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=root,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(raw_path)
        os.fchmod(descriptor, 0o600)
        try:
            temporary_metadata = os.fstat(descriptor)
            _require(
                stat.S_ISREG(temporary_metadata.st_mode)
                and temporary_metadata.st_uid == os.getuid()
                and temporary_metadata.st_nlink == 1
                and stat.S_IMODE(temporary_metadata.st_mode) == 0o600,
                "Persistent ledger temporary metadata is unsafe.",
            )
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                _require(written > 0, "Persistent ledger write stalled.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Persistent session ledger already exists: {path.name}.") from error
        _fsync_directory(root)
        metadata = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 2
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Published persistent session ledger metadata drifted.",
        )
        return {
            "path": str(path),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "payload_sha256": payload.get("payload_sha256"),
            "attestation_mac": cast(Mapping[str, Any], payload["attestation"]).get("mac"),
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            if root.exists():
                _fsync_directory(root)


def _publish_json_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root = path.parent
    if os.path.lexists(root):
        with _session_ledger_lock(root, create=False):
            return _publish_json_exclusive_locked(path, payload)
    try:
        with _session_ledger_lock(root, create=True, exclusive_create=True):
            return _publish_json_exclusive_locked(path, payload)
    except FileExistsError:
        # Another first writer, or recovery from a crash after the durable lock
        # publication, owns the only allowed initialization transition.
        with _session_ledger_lock(root, create=False):
            return _publish_json_exclusive_locked(path, payload)


def publish_session_launch(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    actual_session_argv: Sequence[str],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    validated_plan = validate_session_plan(plan, trust_root=trust_root)
    exact_output_root = _require_plan_output_root(validated_plan, output_root=output_root)
    checked_argv = _validate_actual_session_argv(list(actual_session_argv), plan=validated_plan)
    source = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LAUNCH_ARTIFACT_TYPE,
        "session_nonce": validated_plan["session_nonce"],
        "launch_authority_nonce": validated_plan["launch_authority_nonce"],
        "plan": validated_plan,
        "actual_session_argv": checked_argv,
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count": 1,
        "session_triggered_full_historical_evidence_replay_count": 0,
        "checkpoint_model_load_attempt_upper_bound": 1,
        "outcome_dependent_selection": False,
    }
    payload = _attested_message(
        source,
        trust_root=trust_root,
        purpose=LAUNCH_LEDGER_ATTESTATION_PURPOSE,
    )
    return _publish_json_exclusive(
        _session_ledger_path(
            exact_output_root,
            cast(str, validated_plan["session_nonce"]),
            "launch",
        ),
        payload,
    )


_LEDGER_NAME = re.compile(r"^(?P<nonce>[0-9a-f]{64})\.(?P<kind>launch|terminal)\.json$")
_LEDGER_TEMPORARY_NAME = re.compile(
    r"^\.(?P<final>[0-9a-f]{64}\.(?:launch|terminal)\.json)\."
    r"(?P<random>[A-Za-z0-9_-]+)\.tmp$"
)


def _recover_interrupted_publications_locked(root: Path) -> None:
    recovered = False
    for temporary in sorted(root.iterdir(), key=lambda item: item.name):
        match = _LEDGER_TEMPORARY_NAME.fullmatch(temporary.name)
        if match is None:
            continue
        temporary_meta = os.stat(temporary, follow_symlinks=False)
        _require(
            stat.S_ISREG(temporary_meta.st_mode)
            and temporary_meta.st_uid == os.getuid()
            and stat.S_IMODE(temporary_meta.st_mode) == 0o600
            and temporary_meta.st_nlink in {1, 2},
            "Interrupted persistent ledger temporary is unsafe.",
        )
        final = root / match.group("final")
        if os.path.lexists(final):
            final_meta = os.stat(final, follow_symlinks=False)
            _require(
                stat.S_ISREG(final_meta.st_mode)
                and final_meta.st_uid == os.getuid()
                and stat.S_IMODE(final_meta.st_mode) == 0o600
                and temporary_meta.st_nlink == final_meta.st_nlink == 2
                and (temporary_meta.st_dev, temporary_meta.st_ino)
                == (final_meta.st_dev, final_meta.st_ino),
                "Interrupted persistent ledger final/temporary identity drifted.",
            )
        else:
            _require(
                temporary_meta.st_nlink == 1,
                "Interrupted unpublished persistent ledger temporary has extra links.",
            )
        temporary.unlink()
        recovered = True
    if recovered:
        _fsync_directory(root)


def _load_ledger_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _require(nofollow is not None, "Persistent ledger loading requires O_NOFOLLOW.")
    descriptor = os.open(path, flags | cast(int, nofollow))
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o600,
            "Persistent session ledger member metadata is unsafe.",
        )
        _require(
            0 < before.st_size <= MAXIMUM_LEDGER_BYTES,
            "Persistent session ledger member byte size is unsafe.",
        )
        chunks: list[bytes] = []
        remaining = MAXIMUM_LEDGER_BYTES + 1
        while True:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            _require(
                remaining > 0,
                "Persistent session ledger member exceeds its byte limit.",
            )
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        _require(
            (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_uid,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_uid,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "Persistent session ledger changed while open.",
        )
    finally:
        os.close(descriptor)
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Persistent session ledger is invalid JSON.") from error
    _require(
        isinstance(parsed, dict)
        and data == (json.dumps(parsed, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        "Persistent session ledger bytes are not canonical pretty JSON.",
    )
    return cast(dict[str, Any], parsed), {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "payload_sha256": parsed.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], parsed.get("attestation", {})).get("mac"),
    }


def _receipt_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project every readiness-relevant field from one verified receipt."""

    attestation_value = value.get("attestation")
    _require(
        isinstance(attestation_value, Mapping),
        "Persistent receipt projection lacks its attestation.",
    )
    attestation_map = cast(Mapping[str, Any], attestation_value)
    return {
        "payload_sha256": value["payload_sha256"],
        "attestation_mac": attestation_map["mac"],
        "status": value["status"],
        "planned_coordinates": value["planned_coordinates"],
        "completed_coordinates": value["completed_coordinates"],
        "completed_work_payload_sha256": list(
            cast(list[Any], value["completed_work_payload_sha256"])
        ),
        "completed_result_payload_sha256": list(
            cast(list[Any], value["completed_result_payload_sha256"])
        ),
        "model_load_count": value["model_load_count"],
        "child_full_historical_evidence_replay_count": value[
            "child_full_historical_evidence_replay_count"
        ],
        "active_activation_validation_count": value[
            "active_activation_validation_count"
        ],
        "active_admission_validation_count": value[
            "active_admission_validation_count"
        ],
        "active_genesis_validation_count": value[
            "active_genesis_validation_count"
        ],
        "active_calibration_validation_count": value[
            "active_calibration_validation_count"
        ],
        "active_checkpoint_validation_count": value[
            "active_checkpoint_validation_count"
        ],
        "model_state_reset_count": value["model_state_reset_count"],
        "outcome_dependent_selection": value["outcome_dependent_selection"],
    }


def _load_session_ledger_projection_locked(
    output_root: Path,
    *,
    trust_root: attestation.TrustRoot,
    require_root_absent: bool = False,
) -> dict[str, Any]:
    root = session_ledger_root(output_root)
    empty_projection = {
        "root": str(root),
        "launch_attempt_count": 0,
        "ready_model_load_count": 0,
        "checkpoint_model_load_attempt_upper_bound": 0,
        "observed_successful_model_loads": 0,
        "launch_only_interrupted_attempt_count": 0,
        "terminal_count": 0,
        "child_eof_count": 0,
        "published_bundle_reingestion_count": 0,
        "launch_authority_count": 0,
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count": 0,
        "durably_evidenced_parent_full_evidence_replays": 0,
        "session_triggered_full_historical_evidence_replay_count": 0,
        "normal_no_restart_unique_worker_scale_seed_assignments": 0,
        "normal_path_model_load_bound": 0,
        "normal_path_model_load_bound_applicable": True,
        "normal_path_model_load_bound_observed_satisfied": True,
        "additional_controlled_or_recovery_launch_attempts": 0,
        "controlled_stop_session_count": 0,
        "ready_only_preflight_launch_attempt_count": 0,
        "ready_only_preflight_terminal_count": 0,
        "ready_only_preflight_success_count": 0,
        "ready_only_preflight_failed_or_interrupted_attempt_count": 0,
        "ready_only_preflight_ready_model_load_count": 0,
        "quality_launch_attempt_count": 0,
        "quality_terminal_count": 0,
        "quality_ready_model_load_count": 0,
        "single_worker_normal_path_ready_only_preflight_model_load_upper_bound": (
            READY_ONLY_PREFLIGHT_MODEL_LOAD_UPPER_BOUND
        ),
        "single_worker_normal_path_quality_model_load_upper_bound": (
            NORMAL_PATH_UNIQUE_MODEL_COHORTS
        ),
        "single_worker_normal_path_total_model_load_upper_bound": (
            SINGLE_WORKER_TOTAL_MODEL_LOAD_UPPER_BOUND
        ),
        "single_worker_full_matrix_unique_scale_seed_cohorts": (NORMAL_PATH_UNIQUE_MODEL_COHORTS),
        "single_worker_theoretical_child_checkpoint_model_deserialization_payload_bytes": (
            NORMAL_PATH_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        ),
        "single_worker_legacy_per_shard_child_checkpoint_model_deserialization_payload_bytes": (
            LEGACY_PER_SHARD_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        ),
        "single_worker_theoretical_child_checkpoint_model_deserialization_reduction_factor": (
            THEORETICAL_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_REDUCTION_FACTOR
        ),
        "parent_full_historical_checkpoint_hash_read_bytes": "not_measured",
        "sessions": [],
        "sessions_digest": contract.json_digest([]),
        "registry": [],
        "registry_digest": contract.json_digest([]),
    }
    if require_root_absent:
        _require(
            not os.path.lexists(root),
            "Persistent session ledger appeared during an empty projection read.",
        )
        return empty_projection
    if not os.path.lexists(root):
        return empty_projection
    root_meta = os.stat(root, follow_symlinks=False)
    _require(
        stat.S_ISDIR(root_meta.st_mode)
        and root_meta.st_uid == os.getuid()
        and stat.S_IMODE(root_meta.st_mode) == 0o700,
        "Persistent session ledger root is unsafe.",
    )
    _recover_interrupted_publications_locked(root)
    root_meta = os.stat(root, follow_symlinks=False)
    launches: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    members = sorted(root.iterdir(), key=lambda item: item.name)
    for path in members:
        _require(not path.is_symlink(), "Persistent session ledger contains a symlink.")
        match = _LEDGER_NAME.fullmatch(path.name)
        _require(match is not None, f"Persistent session ledger contains an extra file: {path}")
        checked_match = cast(re.Match[str], match)
        payload, binding = _load_ledger_json(path)
        nonce = checked_match.group("nonce")
        kind = checked_match.group("kind")
        if kind == "launch":
            checked = _verified_message(
                payload,
                source_fields=_LAUNCH_LEDGER_SOURCE_FIELDS,
                trust_root=trust_root,
                purpose=LAUNCH_LEDGER_ATTESTATION_PURPOSE,
            )
            plan = checked.get("plan")
            _require(isinstance(plan, Mapping), "Persistent launch ledger plan is missing.")
            validated_plan = validate_session_plan(
                cast(Mapping[str, Any], plan), trust_root=trust_root
            )
            _require_plan_output_root(validated_plan, output_root=output_root)
            checked_launch_argv = _validate_actual_session_argv(
                checked.get("actual_session_argv"), plan=validated_plan
            )
            _require(
                checked.get("schema_version") == SCHEMA_VERSION
                and checked.get("artifact_type") == LAUNCH_ARTIFACT_TYPE
                and checked.get("session_nonce") == nonce == validated_plan["session_nonce"]
                and checked.get("launch_authority_nonce")
                == validated_plan["launch_authority_nonce"]
                and checked.get(
                    "durably_evidenced_launch_authority_full_historical_evidence_replay_count"
                )
                == 1
                and checked.get("session_triggered_full_historical_evidence_replay_count") == 0
                and checked.get("checkpoint_model_load_attempt_upper_bound") == 1
                and checked.get("outcome_dependent_selection") is False,
                "Persistent launch ledger semantics drifted.",
            )
            _require(nonce not in launches, "Persistent launch ledger nonce repeated.")
            launches[nonce] = checked
            launches[nonce]["actual_session_argv"] = checked_launch_argv
        else:
            checked = _verified_message(
                payload,
                source_fields=_TERMINAL_LEDGER_SOURCE_FIELDS,
                trust_root=trust_root,
                purpose=TERMINAL_LEDGER_ATTESTATION_PURPOSE,
            )
            _require(
                checked.get("schema_version") == SCHEMA_VERSION
                and checked.get("artifact_type") == TERMINAL_ARTIFACT_TYPE
                and checked.get("session_nonce") == nonce,
                "Persistent terminal ledger identity drifted.",
            )
            _require(nonce not in terminals, "Persistent terminal ledger nonce repeated.")
            terminals[nonce] = checked
        bindings.append({"kind": kind, "session_nonce": nonce, **binding})
    root_after = os.stat(root, follow_symlinks=False)
    members_after = sorted(item.name for item in root.iterdir())
    _require(
        (
            root_meta.st_dev,
            root_meta.st_ino,
            root_meta.st_mode,
            root_meta.st_nlink,
            root_meta.st_uid,
            root_meta.st_mtime_ns,
            root_meta.st_ctime_ns,
        )
        == (
            root_after.st_dev,
            root_after.st_ino,
            root_after.st_mode,
            root_after.st_nlink,
            root_after.st_uid,
            root_after.st_mtime_ns,
            root_after.st_ctime_ns,
        )
        and members_after == [item.name for item in members],
        "Persistent session ledger root or member inventory changed during its closed-world scan.",
    )
    _require(
        set(terminals).issubset(launches),
        "Persistent terminal ledger lacks its launch ledger.",
    )
    authorities: set[str] = set()
    authority_bindings: dict[str, tuple[Any, ...]] = {}
    quality_authorities: set[str] = set()
    quality_cohorts: set[tuple[int, str, int]] = set()
    ready_loads = 0
    quality_ready_loads = 0
    preflight_ready_loads = 0
    eof_count = 0
    reingestions = 0
    quality_controlled_stop_sessions = 0
    uninterrupted_quality_terminal_sessions = 0
    session_rows: list[dict[str, Any]] = []
    for nonce, launch in launches.items():
        plan = cast(Mapping[str, Any], launch["plan"])
        authority = cast(str, launch["launch_authority_nonce"])
        session_role = cast(str, plan["session_role"])
        authorities.add(authority)
        authority_binding = (
            plan["worker_index"],
            plan["worker_count"],
            plan["canonical_evaluator_digest"],
            plan["gpu_lease_binding_digest"],
            plan["prerequisites_binding_digest"],
        )
        prior_authority_binding = authority_bindings.setdefault(authority, authority_binding)
        _require(
            prior_authority_binding == authority_binding,
            "Persistent launch authority crossed its frozen worker or prerequisite binding.",
        )
        if session_role == QUALITY_SESSION_ROLE:
            quality_authorities.add(authority)
            quality_cohorts.add(
                (
                    cast(int, plan["worker_index"]),
                    cast(str, plan["scale"]),
                    cast(int, plan["training_seed"]),
                )
            )
            quality_controlled_stop_sessions += (
                plan.get("max_new_cells_stop_limit") is not None
            )
        terminal = terminals.get(nonce)
        if terminal is None:
            session_rows.append(
                {
                    "session_nonce": nonce,
                    "launch_authority_nonce": authority,
                    "session_role": session_role,
                    "plan_payload_sha256": plan["payload_sha256"],
                    "input_binding_digest": plan["input_binding_digest"],
                    "canonical_evaluator_digest": plan["canonical_evaluator_digest"],
                    "gpu_lease_binding_digest": plan["gpu_lease_binding_digest"],
                    "prerequisites_binding_digest": plan[
                        "prerequisites_binding_digest"
                    ],
                    "output_root": plan["output_root"],
                    "worker_index": plan["worker_index"],
                    "worker_count": plan["worker_count"],
                    "scale": plan["scale"],
                    "training_seed": plan["training_seed"],
                    "coordinate_count": plan["coordinate_count"],
                    "coordinate_digest": plan["coordinate_digest"],
                    "max_new_cells_stop_limit": plan["max_new_cells_stop_limit"],
                    "status": "launch_only",
                    "ready_model_load_observed": False,
                    "ready_receipt_binding": None,
                    "final_receipt_binding": None,
                    "completed_work_payload_sha256": [],
                    "completed_result_payload_sha256": [],
                    "published_bundle_reingestion_count": 0,
                    "actual_session_argv": list(launch["actual_session_argv"]),
                    "child_process_returncode": None,
                }
            )
            continue
        _require(
            terminal.get("launch_authority_nonce") == authority
            and terminal.get("plan_payload_sha256") == plan["payload_sha256"],
            "Persistent terminal/launch cross-binding drifted.",
        )
        ready = terminal.get("ready_receipt")
        ready_binding: dict[str, Any] | None = None
        if ready is not None:
            _require(isinstance(ready, Mapping), "Persistent ready receipt is invalid.")
            checked_ready = validate_attested_session_receipt(
                cast(Mapping[str, Any], ready), trust_root=trust_root
            )
            _require(
                checked_ready.get("session_nonce") == nonce
                and checked_ready.get("launch_authority_nonce") == authority
                and checked_ready.get("plan_payload_sha256") == plan["payload_sha256"]
                and checked_ready.get("planned_coordinates") == plan["coordinate_count"]
                and checked_ready.get("completed_coordinates") == 0,
                "Persistent ready receipt cross-binding drifted.",
            )
            ready_loads += 1
            ready_binding = _receipt_projection(checked_ready)
            if session_role == QUALITY_SESSION_ROLE:
                quality_ready_loads += 1
            else:
                preflight_ready_loads += 1
        raw_work_digests = terminal.get("completed_work_payload_sha256")
        raw_result_digests = terminal.get("completed_result_payload_sha256")
        _require(
            isinstance(raw_work_digests, list)
            and isinstance(raw_result_digests, list)
            and len(raw_work_digests) == len(raw_result_digests)
            and all(contract.is_sha256(item) for item in raw_work_digests)
            and all(contract.is_sha256(item) for item in raw_result_digests),
            "Persistent terminal completed-message digests are invalid.",
        )
        work_digests = cast(list[Any], raw_work_digests)
        result_digests = cast(list[Any], raw_result_digests)
        status = terminal.get("status")
        reingestion_count = terminal.get("published_bundle_reingestion_count")
        actual_session_argv = terminal.get("actual_session_argv")
        returncode = terminal.get("child_process_returncode")
        _require(
            status
            in {
                "complete",
                "stopped",
                "child_eof",
                "launch_failure",
                "parent_commit_failure",
                "parent_crash_recovered",
            }
            and type(reingestion_count) is int
            and 0 <= reingestion_count <= 1
            and isinstance(actual_session_argv, list)
            and bool(actual_session_argv)
            and all(isinstance(item, str) and item for item in actual_session_argv)
            and (returncode is None or type(returncode) is int),
            "Persistent terminal status counters are invalid.",
        )
        checked_reingestion_count = cast(int, reingestion_count)
        checked_actual_session_argv = _validate_actual_session_argv(actual_session_argv, plan=plan)
        _require(
            checked_actual_session_argv == launch["actual_session_argv"],
            "Persistent terminal argv differs from its launch ledger.",
        )
        final = terminal.get("final_receipt")
        final_binding: dict[str, Any] | None = None
        if final is not None:
            _require(isinstance(final, Mapping), "Persistent final receipt is invalid.")
            checked_final = validate_attested_session_receipt(
                cast(Mapping[str, Any], final), trust_root=trust_root
            )
            _require(
                checked_final.get("session_nonce") == nonce
                and checked_final.get("launch_authority_nonce") == authority
                and checked_final.get("plan_payload_sha256") == plan["payload_sha256"],
                "Persistent final receipt cross-binding drifted.",
            )
            _require(
                checked_final.get("planned_coordinates") == plan["coordinate_count"]
                and checked_final.get("completed_work_payload_sha256") == work_digests
                and checked_final.get("completed_result_payload_sha256") == result_digests
                and checked_final.get("completed_coordinates") == len(result_digests)
                and checked_final.get("status") == status,
                "Persistent final receipt completion projection drifted.",
            )
            final_binding = _receipt_projection(checked_final)
        if status in {"complete", "stopped"}:
            final_map = cast(Mapping[str, Any], final)
            _require(
                ready is not None
                and final is not None
                and returncode == 0
                and checked_reingestion_count == 0
                and (final_map.get("completed_coordinates") == plan["coordinate_count"])
                is (status == "complete"),
                "Persistent graceful terminal semantics drifted.",
            )
            uninterrupted_quality_terminal_sessions += (
                status == "complete" and session_role == QUALITY_SESSION_ROLE
            )
        elif status == "child_eof":
            _require(
                ready is not None and final is None and type(returncode) is int,
                "Persistent child-EOF terminal semantics drifted.",
            )
        elif status == "parent_commit_failure":
            _require(
                ready is not None
                and final is None
                and type(returncode) is int
                and checked_reingestion_count == 0,
                "Persistent parent-commit-failure terminal semantics drifted.",
            )
        elif status == "parent_crash_recovered":
            _require(
                ready is None
                and final is None
                and returncode is None
                and checked_reingestion_count in {0, 1},
                "Persistent parent-crash-recovery terminal semantics drifted.",
            )
        else:
            _require(
                ready is None
                and final is None
                and not work_digests
                and not result_digests
                and checked_reingestion_count == 0,
                "Persistent launch-failure terminal semantics drifted.",
            )
        eof_count += status == "child_eof"
        reingestions += checked_reingestion_count
        session_rows.append(
            {
                "session_nonce": nonce,
                "launch_authority_nonce": authority,
                "session_role": session_role,
                "plan_payload_sha256": plan["payload_sha256"],
                "input_binding_digest": plan["input_binding_digest"],
                "canonical_evaluator_digest": plan["canonical_evaluator_digest"],
                "gpu_lease_binding_digest": plan["gpu_lease_binding_digest"],
                "prerequisites_binding_digest": plan[
                    "prerequisites_binding_digest"
                ],
                "output_root": plan["output_root"],
                "worker_index": plan["worker_index"],
                "worker_count": plan["worker_count"],
                "scale": plan["scale"],
                "training_seed": plan["training_seed"],
                "coordinate_count": plan["coordinate_count"],
                "coordinate_digest": plan["coordinate_digest"],
                "max_new_cells_stop_limit": plan["max_new_cells_stop_limit"],
                "status": status,
                "ready_model_load_observed": ready is not None,
                "ready_receipt_binding": ready_binding,
                "final_receipt_binding": final_binding,
                "completed_work_payload_sha256": list(work_digests),
                "completed_result_payload_sha256": list(result_digests),
                "published_bundle_reingestion_count": checked_reingestion_count,
                "actual_session_argv": checked_actual_session_argv,
                "child_process_returncode": returncode,
            }
        )
    authority_replays = len(authorities)
    quality_rows = [
        row for row in session_rows if row["session_role"] == QUALITY_SESSION_ROLE
    ]
    preflight_rows = [
        row
        for row in session_rows
        if row["session_role"] == READY_ONLY_PREFLIGHT_SESSION_ROLE
    ]
    quality_terminal_count = sum(row["status"] != "launch_only" for row in quality_rows)
    preflight_terminal_count = sum(
        row["status"] != "launch_only" for row in preflight_rows
    )
    preflight_success_count = sum(
        row["status"] == "stopped"
        and row["ready_receipt_binding"] == row["final_receipt_binding"]
        for row in preflight_rows
    )
    additional_attempts, normal_path_applicable = _normal_path_claim_semantics(
        launch_count=len(quality_rows),
        terminal_count=quality_terminal_count,
        graceful_terminal_count=uninterrupted_quality_terminal_sessions,
        controlled_stop_count=quality_controlled_stop_sessions,
        cohort_count=len(quality_cohorts),
        launch_authority_count=len(quality_authorities),
        worker_counts=[cast(int, row["worker_count"]) for row in quality_rows],
    )
    projection = {
        "root": str(root),
        "launch_attempt_count": len(launches),
        "ready_model_load_count": ready_loads,
        "checkpoint_model_load_attempt_upper_bound": len(launches),
        "observed_successful_model_loads": ready_loads,
        "launch_only_interrupted_attempt_count": len(set(launches) - set(terminals)),
        "terminal_count": len(terminals),
        "child_eof_count": eof_count,
        "published_bundle_reingestion_count": reingestions,
        "launch_authority_count": len(authorities),
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count": (
            authority_replays
        ),
        "durably_evidenced_parent_full_evidence_replays": authority_replays,
        "session_triggered_full_historical_evidence_replay_count": 0,
        "normal_no_restart_unique_worker_scale_seed_assignments": len(quality_cohorts),
        "normal_path_model_load_bound": len(quality_cohorts),
        "normal_path_model_load_bound_applicable": normal_path_applicable,
        "normal_path_model_load_bound_observed_satisfied": (
            quality_ready_loads <= len(quality_cohorts) if normal_path_applicable else None
        ),
        "additional_controlled_or_recovery_launch_attempts": additional_attempts,
        "controlled_stop_session_count": quality_controlled_stop_sessions,
        "ready_only_preflight_launch_attempt_count": len(preflight_rows),
        "ready_only_preflight_terminal_count": preflight_terminal_count,
        "ready_only_preflight_success_count": preflight_success_count,
        "ready_only_preflight_failed_or_interrupted_attempt_count": (
            len(preflight_rows) - preflight_success_count
        ),
        "ready_only_preflight_ready_model_load_count": preflight_ready_loads,
        "quality_launch_attempt_count": len(quality_rows),
        "quality_terminal_count": quality_terminal_count,
        "quality_ready_model_load_count": quality_ready_loads,
        "single_worker_normal_path_ready_only_preflight_model_load_upper_bound": (
            READY_ONLY_PREFLIGHT_MODEL_LOAD_UPPER_BOUND
        ),
        "single_worker_normal_path_quality_model_load_upper_bound": (
            NORMAL_PATH_UNIQUE_MODEL_COHORTS
        ),
        "single_worker_normal_path_total_model_load_upper_bound": (
            SINGLE_WORKER_TOTAL_MODEL_LOAD_UPPER_BOUND
        ),
        "single_worker_full_matrix_unique_scale_seed_cohorts": (NORMAL_PATH_UNIQUE_MODEL_COHORTS),
        "single_worker_theoretical_child_checkpoint_model_deserialization_payload_bytes": (
            NORMAL_PATH_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        ),
        "single_worker_legacy_per_shard_child_checkpoint_model_deserialization_payload_bytes": (
            LEGACY_PER_SHARD_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        ),
        "single_worker_theoretical_child_checkpoint_model_deserialization_reduction_factor": (
            THEORETICAL_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_REDUCTION_FACTOR
        ),
        "parent_full_historical_checkpoint_hash_read_bytes": "not_measured",
        "sessions": session_rows,
        "sessions_digest": contract.json_digest(session_rows),
        "registry": bindings,
    }
    return validate_session_ledger_projection(
        {**projection, "registry_digest": contract.json_digest(bindings)}
    )


def validate_session_ledger_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "root",
        "launch_attempt_count",
        "ready_model_load_count",
        "checkpoint_model_load_attempt_upper_bound",
        "observed_successful_model_loads",
        "launch_only_interrupted_attempt_count",
        "terminal_count",
        "child_eof_count",
        "published_bundle_reingestion_count",
        "launch_authority_count",
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count",
        "durably_evidenced_parent_full_evidence_replays",
        "session_triggered_full_historical_evidence_replay_count",
        "normal_no_restart_unique_worker_scale_seed_assignments",
        "normal_path_model_load_bound",
        "normal_path_model_load_bound_applicable",
        "normal_path_model_load_bound_observed_satisfied",
        "additional_controlled_or_recovery_launch_attempts",
        "controlled_stop_session_count",
        "ready_only_preflight_launch_attempt_count",
        "ready_only_preflight_terminal_count",
        "ready_only_preflight_success_count",
        "ready_only_preflight_failed_or_interrupted_attempt_count",
        "ready_only_preflight_ready_model_load_count",
        "quality_launch_attempt_count",
        "quality_terminal_count",
        "quality_ready_model_load_count",
        "single_worker_normal_path_ready_only_preflight_model_load_upper_bound",
        "single_worker_normal_path_quality_model_load_upper_bound",
        "single_worker_normal_path_total_model_load_upper_bound",
        "single_worker_full_matrix_unique_scale_seed_cohorts",
        "single_worker_theoretical_child_checkpoint_model_deserialization_payload_bytes",
        "single_worker_legacy_per_shard_child_checkpoint_model_deserialization_payload_bytes",
        "single_worker_theoretical_child_checkpoint_model_deserialization_reduction_factor",
        "parent_full_historical_checkpoint_hash_read_bytes",
        "sessions",
        "sessions_digest",
        "registry",
        "registry_digest",
    }
    _require(set(value) == expected_fields, "Persistent ledger projection schema drifted.")
    root = value.get("root")
    sessions = value.get("sessions")
    registry = value.get("registry")
    _require(
        isinstance(root, str)
        and Path(root) == Path(os.path.abspath(root))
        and isinstance(sessions, list)
        and isinstance(registry, list)
        and value.get("sessions_digest") == contract.json_digest(sessions)
        and value.get("registry_digest") == contract.json_digest(registry),
        "Persistent ledger projection roots or digests are invalid.",
    )
    session_values = cast(list[Any], sessions)
    registry_values = cast(list[Any], registry)
    root_path = Path(cast(str, root))
    expected_output_name = root_path.name.removeprefix(".").removesuffix(
        f".{SESSION_LEDGER_ROOT_SUFFIX}"
    )
    expected_output_root = str(root_path.parent / expected_output_name)
    integer_fields = expected_fields - {
        "root",
        "normal_path_model_load_bound_applicable",
        "normal_path_model_load_bound_observed_satisfied",
        "parent_full_historical_checkpoint_hash_read_bytes",
        "sessions",
        "sessions_digest",
        "registry",
        "registry_digest",
    }
    _require(
        all(
            type(value.get(field)) is int and cast(int, value[field]) >= 0
            for field in integer_fields
        ),
        "Persistent ledger projection counters are invalid.",
    )
    session_fields = {
        "session_nonce",
        "launch_authority_nonce",
        "session_role",
        "plan_payload_sha256",
        "input_binding_digest",
        "canonical_evaluator_digest",
        "gpu_lease_binding_digest",
        "prerequisites_binding_digest",
        "output_root",
        "worker_index",
        "worker_count",
        "scale",
        "training_seed",
        "coordinate_count",
        "coordinate_digest",
        "max_new_cells_stop_limit",
        "status",
        "ready_model_load_observed",
        "ready_receipt_binding",
        "final_receipt_binding",
        "completed_work_payload_sha256",
        "completed_result_payload_sha256",
        "published_bundle_reingestion_count",
        "actual_session_argv",
        "child_process_returncode",
    }
    checked_sessions: list[Mapping[str, Any]] = []
    for raw in session_values:
        _require(isinstance(raw, Mapping), "Persistent session projection row is invalid.")
        row = cast(Mapping[str, Any], raw)
        work_digests = row.get("completed_work_payload_sha256")
        result_digests = row.get("completed_result_payload_sha256")
        _require(
            set(row) == session_fields
            and contract.is_sha256(row.get("session_nonce"))
            and contract.is_sha256(row.get("launch_authority_nonce"))
            and row.get("session_role") in SESSION_ROLES
            and contract.is_sha256(row.get("plan_payload_sha256"))
            and contract.is_sha256(row.get("input_binding_digest"))
            and contract.is_sha256(row.get("canonical_evaluator_digest"))
            and contract.is_sha256(row.get("gpu_lease_binding_digest"))
            and contract.is_sha256(row.get("prerequisites_binding_digest"))
            and row.get("output_root") == expected_output_root
            and contract.is_sha256(row.get("coordinate_digest"))
            and type(row.get("worker_index")) is int
            and type(row.get("worker_count")) is int
            and 0 <= cast(int, row["worker_index"]) < cast(int, row["worker_count"])
            and row.get("scale") in contract.SCALES
            and row.get("training_seed") in contract.TRAINING_SEEDS
            and type(row.get("coordinate_count")) is int
            and cast(int, row["coordinate_count"]) > 0
            and (
                row.get("max_new_cells_stop_limit") is None
                or (
                    type(row.get("max_new_cells_stop_limit")) is int
                    and cast(int, row["max_new_cells_stop_limit"]) > 0
                )
            )
            and row.get("status")
            in {
                "launch_only",
                "complete",
                "stopped",
                "child_eof",
                "launch_failure",
                "parent_commit_failure",
                "parent_crash_recovered",
            }
            and type(row.get("ready_model_load_observed")) is bool
            and isinstance(work_digests, list)
            and isinstance(result_digests, list)
            and len(work_digests) == len(result_digests)
            and all(contract.is_sha256(item) for item in work_digests)
            and all(contract.is_sha256(item) for item in result_digests)
            and type(row.get("published_bundle_reingestion_count")) is int
            and 0 <= cast(int, row["published_bundle_reingestion_count"]) <= 1
            and isinstance(row.get("actual_session_argv"), list)
            and bool(cast(list[Any], row["actual_session_argv"]))
            and all(
                isinstance(item, str) and item
                for item in cast(list[Any], row["actual_session_argv"])
            )
            and (
                row.get("child_process_returncode") is None
                or type(row.get("child_process_returncode")) is int
            ),
            "Persistent session projection row semantics drifted.",
        )
        receipt_fields = {
            "payload_sha256",
            "attestation_mac",
            "status",
            "planned_coordinates",
            "completed_coordinates",
            "completed_work_payload_sha256",
            "completed_result_payload_sha256",
            "model_load_count",
            "child_full_historical_evidence_replay_count",
            "active_activation_validation_count",
            "active_admission_validation_count",
            "active_genesis_validation_count",
            "active_calibration_validation_count",
            "active_checkpoint_validation_count",
            "model_state_reset_count",
            "outcome_dependent_selection",
        }
        for name in ("ready_receipt_binding", "final_receipt_binding"):
            raw_receipt = row.get(name)
            if raw_receipt is None:
                continue
            _require(
                isinstance(raw_receipt, Mapping)
                and set(raw_receipt) == receipt_fields
                and contract.is_sha256(raw_receipt.get("payload_sha256"))
                and contract.is_sha256(raw_receipt.get("attestation_mac"))
                and raw_receipt.get("status") in {"complete", "stopped"}
                and type(raw_receipt.get("planned_coordinates")) is int
                and cast(int, raw_receipt["planned_coordinates"]) > 0
                and type(raw_receipt.get("completed_coordinates")) is int
                and 0
                <= cast(int, raw_receipt["completed_coordinates"])
                <= cast(int, raw_receipt["planned_coordinates"])
                and isinstance(raw_receipt.get("completed_work_payload_sha256"), list)
                and isinstance(raw_receipt.get("completed_result_payload_sha256"), list)
                and len(cast(list[Any], raw_receipt["completed_work_payload_sha256"]))
                == len(cast(list[Any], raw_receipt["completed_result_payload_sha256"]))
                == cast(int, raw_receipt["completed_coordinates"])
                and all(
                    contract.is_sha256(item)
                    for item in cast(
                        list[Any], raw_receipt["completed_work_payload_sha256"]
                    )
                )
                and all(
                    contract.is_sha256(item)
                    for item in cast(
                        list[Any], raw_receipt["completed_result_payload_sha256"]
                    )
                )
                and raw_receipt.get("model_load_count") == MODEL_LOADS_PER_SESSION
                and raw_receipt.get("child_full_historical_evidence_replay_count")
                == CHILD_FULL_HISTORICAL_EVIDENCE_REPLAY_COUNT
                and all(
                    raw_receipt.get(field) == 1
                    for field in (
                        "active_activation_validation_count",
                        "active_admission_validation_count",
                        "active_genesis_validation_count",
                        "active_calibration_validation_count",
                        "active_checkpoint_validation_count",
                    )
                )
                and raw_receipt.get("model_state_reset_count")
                == raw_receipt.get("completed_coordinates")
                and raw_receipt.get("outcome_dependent_selection") is False,
                "Persistent projected receipt semantics drifted.",
            )
        _require(
            (row["ready_model_load_observed"] is True)
            is (row["ready_receipt_binding"] is not None)
            and (row["status"] in {"complete", "stopped"})
            is (row["final_receipt_binding"] is not None),
            "Persistent projected receipt presence drifted.",
        )
        argv = row.get("actual_session_argv")
        if isinstance(argv, list):
            launch_indices = [index for index, item in enumerate(argv) if item == "--launch-nonce"]
            scale_indices = [index for index, item in enumerate(argv) if item == "--scale"]
            seed_indices = [index for index, item in enumerate(argv) if item == "--training-seed"]
            _require(
                argv.count("--persistent-session") == 1
                and len(launch_indices) == len(scale_indices) == len(seed_indices) == 1
                and launch_indices[0] + 1 < len(argv)
                and scale_indices[0] + 1 < len(argv)
                and seed_indices[0] + 1 < len(argv)
                and argv[launch_indices[0] + 1] == row["session_nonce"]
                and argv[scale_indices[0] + 1] == row["scale"]
                and argv[seed_indices[0] + 1] == str(row["training_seed"]),
                "Persistent projected actual session argv drifted.",
            )
        checked_sessions.append(row)
    nonces = [cast(str, row["session_nonce"]) for row in checked_sessions]
    _require(
        nonces == sorted(set(nonces)),
        "Persistent session projection rows are not unique canonical sessions.",
    )
    launch_count = len(checked_sessions)
    terminal_count = sum(row["status"] != "launch_only" for row in checked_sessions)
    launch_only_count = launch_count - terminal_count
    ready_count = sum(row["ready_model_load_observed"] is True for row in checked_sessions)
    eof_count = sum(row["status"] == "child_eof" for row in checked_sessions)
    reingestions = sum(
        cast(int, row["published_bundle_reingestion_count"]) for row in checked_sessions
    )
    authorities = {row["launch_authority_nonce"] for row in checked_sessions}
    quality_rows = [
        row for row in checked_sessions if row["session_role"] == QUALITY_SESSION_ROLE
    ]
    preflight_rows = [
        row
        for row in checked_sessions
        if row["session_role"] == READY_ONLY_PREFLIGHT_SESSION_ROLE
    ]
    quality_authorities = {row["launch_authority_nonce"] for row in quality_rows}
    cohorts = {
        (row["worker_index"], row["scale"], row["training_seed"]) for row in checked_sessions
        if row["session_role"] == QUALITY_SESSION_ROLE
    }
    controlled = sum(row["max_new_cells_stop_limit"] is not None for row in quality_rows)
    graceful_terminal_count = sum(row["status"] == "complete" for row in quality_rows)
    quality_terminal_count = sum(row["status"] != "launch_only" for row in quality_rows)
    quality_ready_count = sum(
        row["ready_model_load_observed"] is True for row in quality_rows
    )
    preflight_terminal_count = sum(
        row["status"] != "launch_only" for row in preflight_rows
    )
    preflight_ready_count = sum(
        row["ready_model_load_observed"] is True for row in preflight_rows
    )
    preflight_success_count = sum(
        row["status"] == "stopped"
        and row["ready_receipt_binding"] == row["final_receipt_binding"]
        for row in preflight_rows
    )
    additional, expected_applicable = _normal_path_claim_semantics(
        launch_count=len(quality_rows),
        terminal_count=quality_terminal_count,
        graceful_terminal_count=graceful_terminal_count,
        controlled_stop_count=controlled,
        cohort_count=len(cohorts),
        launch_authority_count=len(quality_authorities),
        worker_counts=[cast(int, row["worker_count"]) for row in quality_rows],
    )
    _require(
        value.get("launch_attempt_count") == launch_count
        and value.get("checkpoint_model_load_attempt_upper_bound") == launch_count
        and value.get("terminal_count") == terminal_count
        and value.get("launch_only_interrupted_attempt_count") == launch_only_count
        and value.get("ready_model_load_count") == ready_count
        and value.get("observed_successful_model_loads") == ready_count
        and value.get("child_eof_count") == eof_count
        and value.get("published_bundle_reingestion_count") == reingestions
        and value.get("launch_authority_count") == len(authorities)
        and value.get("durably_evidenced_launch_authority_full_historical_evidence_replay_count")
        == len(authorities)
        and value.get("durably_evidenced_parent_full_evidence_replays") == len(authorities)
        and value.get("session_triggered_full_historical_evidence_replay_count") == 0
        and value.get("normal_no_restart_unique_worker_scale_seed_assignments") == len(cohorts)
        and value.get("normal_path_model_load_bound") == len(cohorts)
        and value.get("additional_controlled_or_recovery_launch_attempts") == additional
        and value.get("controlled_stop_session_count") == controlled
        and value.get("ready_only_preflight_launch_attempt_count") == len(preflight_rows)
        and value.get("ready_only_preflight_terminal_count") == preflight_terminal_count
        and value.get("ready_only_preflight_success_count") == preflight_success_count
        and value.get("ready_only_preflight_failed_or_interrupted_attempt_count")
        == len(preflight_rows) - preflight_success_count
        and value.get("ready_only_preflight_ready_model_load_count")
        == preflight_ready_count
        and value.get("quality_launch_attempt_count") == len(quality_rows)
        and value.get("quality_terminal_count") == quality_terminal_count
        and value.get("quality_ready_model_load_count") == quality_ready_count,
        "Persistent ledger projection aggregate counters drifted.",
    )
    applicable = cast(bool, value["normal_path_model_load_bound_applicable"])
    observed_bound = value["normal_path_model_load_bound_observed_satisfied"]
    _require(
        type(applicable) is bool
        and applicable is expected_applicable
        and (
            (applicable and observed_bound is (quality_ready_count <= len(cohorts)))
            or (not applicable and observed_bound is None)
        )
        and value.get("single_worker_full_matrix_unique_scale_seed_cohorts")
        == NORMAL_PATH_UNIQUE_MODEL_COHORTS
        and value.get(
            "single_worker_normal_path_ready_only_preflight_model_load_upper_bound"
        )
        == READY_ONLY_PREFLIGHT_MODEL_LOAD_UPPER_BOUND
        and value.get("single_worker_normal_path_quality_model_load_upper_bound")
        == NORMAL_PATH_UNIQUE_MODEL_COHORTS
        and value.get("single_worker_normal_path_total_model_load_upper_bound")
        == SINGLE_WORKER_TOTAL_MODEL_LOAD_UPPER_BOUND
        and value.get(
            "single_worker_theoretical_child_checkpoint_model_deserialization_payload_bytes"
        )
        == NORMAL_PATH_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        and value.get(
            "single_worker_legacy_per_shard_child_checkpoint_model_deserialization_payload_bytes"
        )
        == LEGACY_PER_SHARD_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_PAYLOAD_BYTES
        and value.get(
            "single_worker_theoretical_child_checkpoint_model_deserialization_reduction_factor"
        )
        == THEORETICAL_CHILD_CHECKPOINT_MODEL_DESERIALIZATION_REDUCTION_FACTOR
        and value.get("parent_full_historical_checkpoint_hash_read_bytes") == "not_measured",
        "Persistent ledger projection efficiency semantics drifted.",
    )
    registry_rows = [
        cast(Mapping[str, Any], row) for row in registry_values if isinstance(row, Mapping)
    ]
    _require(
        len(registry_rows) == len(registry_values) == launch_count + terminal_count
        and [cast(str, row.get("path")) for row in registry_rows]
        == sorted(cast(str, row.get("path")) for row in registry_rows)
        and all(
            set(row)
            == {
                "kind",
                "session_nonce",
                "path",
                "bytes",
                "sha256",
                "payload_sha256",
                "attestation_mac",
            }
            and row.get("kind") in {"launch", "terminal"}
            and contract.is_sha256(row.get("session_nonce"))
            and isinstance(row.get("path"), str)
            and type(row.get("bytes")) is int
            and cast(int, row["bytes"]) > 0
            and contract.is_sha256(row.get("sha256"))
            and contract.is_sha256(row.get("payload_sha256"))
            and contract.is_sha256(row.get("attestation_mac"))
            for row in registry_rows
        ),
        "Persistent ledger projection file registry drifted.",
    )
    return _json_clone(dict(value))


def ready_only_preflight_binding(
    value: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reconstruct the unique zero-work readiness proof from authenticated ledgers."""

    projection = validate_session_ledger_projection(value)
    rows = [
        cast(Mapping[str, Any], row)
        for row in cast(list[Any], projection["sessions"])
        if cast(Mapping[str, Any], row).get("session_role")
        == READY_ONLY_PREFLIGHT_SESSION_ROLE
    ]
    if not rows:
        return None
    first_coordinate = dict(contract.quality_coordinates()[0])
    expected_coordinate_digest = contract.json_digest([first_coordinate])
    successes: list[Mapping[str, Any]] = []
    for row in rows:
        ready = row.get("ready_receipt_binding")
        final = row.get("final_receipt_binding")
        _require(
            row.get("worker_index") == 0
            and row.get("worker_count") == 1
            and row.get("scale") == first_coordinate["scale"]
            and row.get("training_seed") == first_coordinate["training_seed"]
            and row.get("coordinate_count") == 1
            and row.get("coordinate_digest") == expected_coordinate_digest
            and row.get("max_new_cells_stop_limit") == 1
            and row.get("completed_work_payload_sha256") == []
            and row.get("completed_result_payload_sha256") == []
            and row.get("published_bundle_reingestion_count") == 0
            and row.get("status") != "complete",
            "Ready-only preflight attempt performed work or changed its frozen plan.",
        )
        if ready is not None:
            _require(
                isinstance(ready, Mapping)
                and ready.get("status") == "stopped"
                and ready.get("planned_coordinates") == 1
                and ready.get("completed_coordinates") == 0
                and ready.get("completed_work_payload_sha256") == []
                and ready.get("completed_result_payload_sha256") == []
                and ready.get("model_load_count") == 1
                and ready.get("child_full_historical_evidence_replay_count") == 0
                and ready.get("model_state_reset_count") == 0
                and ready.get("outcome_dependent_selection") is False
                and all(
                    ready.get(field) == 1
                    for field in (
                        "active_activation_validation_count",
                        "active_admission_validation_count",
                        "active_genesis_validation_count",
                        "active_calibration_validation_count",
                        "active_checkpoint_validation_count",
                    )
                ),
                "Ready-only preflight ready receipt is not an exact zero-work proof.",
            )
        if row.get("status") == "stopped":
            _require(
                ready is not None
                and final is not None
                and ready == final
                and row.get("child_process_returncode") == 0,
                "Ready-only preflight ready/final receipts are not identical.",
            )
            successes.append(row)
        else:
            _require(
                final is None,
                "Failed ready-only preflight attempt has a final success receipt.",
            )
    _require(
        len(successes) <= 1,
        "Ready-only preflight has more than one successful execution.",
    )
    if not successes:
        return None
    _require(
        all(row.get("status") != "launch_only" for row in rows),
        "Successful ready-only preflight coexists with an unrecovered launch.",
    )
    success = successes[0]
    registry_rows = [
        cast(Mapping[str, Any], row)
        for row in cast(list[Any], projection["registry"])
        if cast(Mapping[str, Any], row).get("session_nonce")
        in {attempt["session_nonce"] for attempt in rows}
    ]
    success_registry = [
        dict(row)
        for row in registry_rows
        if row.get("session_nonce") == success["session_nonce"]
    ]
    _require(
        [row["kind"] for row in success_registry] == ["launch", "terminal"],
        "Ready-only preflight success lacks exact launch/terminal file bindings.",
    )
    source = {
        "schema_version": 1,
        "session_role": READY_ONLY_PREFLIGHT_SESSION_ROLE,
        "status": "stopped",
        "session_nonce": success["session_nonce"],
        "launch_authority_nonce": success["launch_authority_nonce"],
        "plan_payload_sha256": success["plan_payload_sha256"],
        "input_binding_digest": success["input_binding_digest"],
        "canonical_evaluator_digest": success["canonical_evaluator_digest"],
        "gpu_lease_binding_digest": success["gpu_lease_binding_digest"],
        "prerequisites_binding_digest": success["prerequisites_binding_digest"],
        "output_root": success["output_root"],
        "worker_index": 0,
        "worker_count": 1,
        "coordinate_count": 1,
        "coordinate_digest": expected_coordinate_digest,
        "ready_receipt_binding": dict(
            cast(Mapping[str, Any], success["ready_receipt_binding"])
        ),
        "final_receipt_binding": dict(
            cast(Mapping[str, Any], success["final_receipt_binding"])
        ),
        "launch_ledger_binding": success_registry[0],
        "terminal_ledger_binding": success_registry[1],
        "launch_attempt_count": len(rows),
        "failed_attempt_count": len(rows) - 1,
        "attempt_history_digest": contract.json_digest(rows),
        "attempt_registry_digest": contract.json_digest(registry_rows),
        "model_load_count": 1,
        "quality_work_order_count": 0,
        "quality_result_count": 0,
        "model_state_reset_count": 0,
        "published_bundle_reingestion_count": 0,
        "outcome_dependent_selection": False,
        "child_process_returncode": 0,
    }
    return _json_clone(source)


def load_session_ledger_projection(
    output_root: Path, *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    root = session_ledger_root(output_root)
    if not os.path.lexists(root):
        lock_path = session_ledger_lock_path(output_root)
        if not os.path.lexists(lock_path):
            projection = _load_session_ledger_projection_locked(
                output_root,
                trust_root=trust_root,
                require_root_absent=True,
            )
            _require(
                not os.path.lexists(root) and not os.path.lexists(lock_path),
                "Persistent session ledger appeared during an unlocked empty read.",
            )
            return projection
        # A durable lock-only state is the recoverable first-write boundary.  It
        # is adopted by inode without O_CREAT; an active first writer may create
        # the root before we acquire the flock, so recheck under the lease.
        with _session_ledger_lock(root, create=False):
            if os.path.lexists(root):
                return _load_session_ledger_projection_locked(
                    output_root, trust_root=trust_root
                )
            return _load_session_ledger_projection_locked(
                output_root,
                trust_root=trust_root,
                require_root_absent=True,
            )
    with _session_ledger_lock(root, create=False):
        return _load_session_ledger_projection_locked(output_root, trust_root=trust_root)


def publish_session_terminal(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    status: str,
    ready_receipt: Mapping[str, Any] | None,
    final_receipt: Mapping[str, Any] | None,
    work_orders: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    published_bundle_reingestion_count: int,
    actual_session_argv: Sequence[str],
    child_process_returncode: int | None,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    validated_plan = validate_session_plan(plan, trust_root=trust_root)
    exact_output_root = _require_plan_output_root(validated_plan, output_root=output_root)
    _require(
        status
        in {
            "complete",
            "stopped",
            "child_eof",
            "launch_failure",
            "parent_commit_failure",
            "parent_crash_recovered",
        },
        "Persistent terminal-ledger status is invalid.",
    )
    checked_argv = _validate_actual_session_argv(list(actual_session_argv), plan=validated_plan)
    _require(
        type(published_bundle_reingestion_count) is int
        and 0 <= published_bundle_reingestion_count <= 1,
        "Persistent reingestion count is invalid.",
    )
    _require(
        child_process_returncode is None or type(child_process_returncode) is int,
        "Persistent child return code is invalid.",
    )
    _require(
        len(work_orders) == len(results),
        "Persistent terminal work/result inventory is incomplete.",
    )
    checked_work: list[dict[str, Any]] = []
    checked_results: list[dict[str, Any]] = []
    for sequence_index, (raw_work, raw_result) in enumerate(zip(work_orders, results, strict=True)):
        work = validate_work_order(
            raw_work,
            plan=validated_plan,
            expected_sequence_index=sequence_index,
            trust_root=trust_root,
        )
        result = validate_work_result(
            raw_result,
            plan=validated_plan,
            work_order=work,
            trust_root=trust_root,
        )
        checked_work.append(work)
        checked_results.append(result)
    checked_ready = None
    if ready_receipt is not None:
        checked_ready = validate_session_receipt(
            ready_receipt,
            plan=validated_plan,
            work_orders=(),
            results=(),
            trust_root=trust_root,
        )
    checked_final = None
    if final_receipt is not None:
        checked_final = validate_session_receipt(
            final_receipt,
            plan=validated_plan,
            work_orders=checked_work,
            results=checked_results,
            trust_root=trust_root,
        )
    _require(
        (
            status in {"complete", "stopped"}
            and checked_ready is not None
            and checked_final is not None
            and child_process_returncode == 0
            and published_bundle_reingestion_count == 0
            and (checked_final["status"] == status)
        )
        or (
            status == "child_eof"
            and checked_ready is not None
            and checked_final is None
            and type(child_process_returncode) is int
        )
        or (
            status == "parent_commit_failure"
            and checked_ready is not None
            and checked_final is None
            and type(child_process_returncode) is int
            and published_bundle_reingestion_count == 0
        )
        or (
            status == "parent_crash_recovered"
            and checked_ready is None
            and checked_final is None
            and child_process_returncode is None
            and published_bundle_reingestion_count in {0, 1}
        )
        or (
            status == "launch_failure"
            and checked_ready is None
            and checked_final is None
            and not checked_work
            and not checked_results
            and published_bundle_reingestion_count == 0
        ),
        "Persistent terminal status/receipt semantics drifted.",
    )
    source = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TERMINAL_ARTIFACT_TYPE,
        "session_nonce": validated_plan["session_nonce"],
        "launch_authority_nonce": validated_plan["launch_authority_nonce"],
        "plan_payload_sha256": validated_plan["payload_sha256"],
        "status": status,
        "ready_receipt": checked_ready,
        "final_receipt": checked_final,
        "completed_work_payload_sha256": [item["payload_sha256"] for item in checked_work],
        "completed_result_payload_sha256": [item["payload_sha256"] for item in checked_results],
        "published_bundle_reingestion_count": published_bundle_reingestion_count,
        "actual_session_argv": checked_argv,
        "child_process_returncode": child_process_returncode,
    }
    payload = _attested_message(
        source,
        trust_root=trust_root,
        purpose=TERMINAL_LEDGER_ATTESTATION_PURPOSE,
    )
    return _publish_json_exclusive(
        _session_ledger_path(
            exact_output_root,
            cast(str, validated_plan["session_nonce"]),
            "terminal",
        ),
        payload,
    )


def reconcile_committed_launch_only_sessions(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    worker_index: int,
    gpu_lease_binding_digest: str,
    terminal_authority_check: Callable[[Mapping[str, Any], Sequence[str]], None],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    exact_output_root = Path(os.path.abspath(output_root))
    _require(
        exact_output_root == output_root,
        "Persistent recovery output root is not exact.",
    )
    projection = load_session_ledger_projection(exact_output_root, trust_root=trust_root)
    _require(
        type(worker_index) is int
        and worker_index >= 0
        and contract.is_sha256(gpu_lease_binding_digest),
        "Persistent recovery worker or held GPU lease binding is invalid.",
    )
    launch_only = {
        cast(str, row["session_nonce"])
        for row in cast(list[Mapping[str, Any]], projection["sessions"])
        if row["status"] == "launch_only" and row["worker_index"] == worker_index
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        execution = record.get("persistent_session_execution")
        _require(
            isinstance(execution, Mapping),
            "Persistent recovery record lacks execution evidence.",
        )
        plan_projection = cast(Mapping[str, Any], execution).get("plan")
        _require(
            isinstance(plan_projection, Mapping)
            and contract.is_sha256(plan_projection.get("session_nonce")),
            "Persistent recovery record lacks a session identity.",
        )
        checked_plan_projection = cast(Mapping[str, Any], plan_projection)
        nonce = cast(str, checked_plan_projection["session_nonce"])
        if nonce in launch_only:
            grouped.setdefault(nonce, []).append(record)
    for nonce in sorted(launch_only):
        launch_path = _session_ledger_path(exact_output_root, nonce, "launch")
        raw_launch, _binding = _load_ledger_json(launch_path)
        launch = _verified_message(
            raw_launch,
            source_fields=_LAUNCH_LEDGER_SOURCE_FIELDS,
            trust_root=trust_root,
            purpose=LAUNCH_LEDGER_ATTESTATION_PURPOSE,
        )
        raw_plan = launch.get("plan")
        _require(
            isinstance(raw_plan, Mapping),
            "Persistent recovery launch plan is missing.",
        )
        plan = validate_session_plan(cast(Mapping[str, Any], raw_plan), trust_root=trust_root)
        _require_plan_output_root(plan, output_root=exact_output_root)
        _require(
            plan["worker_index"] == worker_index
            and plan["gpu_lease_binding_digest"] == gpu_lease_binding_digest,
            "Persistent recovery session is not owned by the reacquired worker GPU lease.",
        )
        rows = sorted(
            grouped.get(nonce, []),
            key=lambda row: cast(
                int,
                cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], row["persistent_session_execution"])["work_order"],
                )["sequence_index"],
            ),
        )
        work_orders: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        session_argv = _validate_actual_session_argv(launch.get("actual_session_argv"), plan=plan)
        reingested = False
        for sequence_index, row in enumerate(rows):
            execution = cast(Mapping[str, Any], row["persistent_session_execution"])
            plan_projection = cast(Mapping[str, Any], execution["plan"])
            _require(
                plan_projection.get("session_nonce") == nonce
                and plan_projection.get("launch_authority_nonce") == plan["launch_authority_nonce"]
                and plan_projection.get("payload_sha256") == plan["payload_sha256"]
                and plan_projection.get("coordinate_digest") == plan["coordinate_digest"],
                "Persistent recovery record plan projection drifted.",
            )
            raw_argv = execution.get("session_command")
            checked_argv = _validate_actual_session_argv(raw_argv, plan=plan)
            _require(
                session_argv == checked_argv,
                "Persistent recovery session argv changed within one launch.",
            )
            raw_work = execution.get("work_order")
            _require(
                isinstance(raw_work, Mapping),
                "Persistent recovery work order is missing.",
            )
            work = validate_work_order(
                cast(Mapping[str, Any], raw_work),
                plan=plan,
                expected_sequence_index=sequence_index,
                trust_root=trust_root,
            )
            raw_result = execution.get("work_result")
            if raw_result is None:
                _require(
                    not reingested
                    and sequence_index == len(rows) - 1
                    and execution.get("published_bundle_reingested_after_child_eof") is True,
                    "Persistent recovery EOF record is not one exact final reingestion.",
                )
                reingested = True
                continue
            _require(
                not reingested and isinstance(raw_result, Mapping),
                "Persistent recovery result ordering drifted.",
            )
            result = validate_work_result(
                cast(Mapping[str, Any], raw_result),
                plan=plan,
                work_order=work,
                trust_root=trust_root,
            )
            work_orders.append(work)
            results.append(result)
        terminal_authority_check(plan, session_argv)
        publish_session_terminal(
            exact_output_root,
            plan,
            status="parent_crash_recovered",
            ready_receipt=None,
            final_receipt=None,
            work_orders=work_orders,
            results=results,
            published_bundle_reingestion_count=int(reingested),
            actual_session_argv=session_argv,
            child_process_returncode=None,
            trust_root=trust_root,
        )
    if not launch_only:
        return projection
    return load_session_ledger_projection(exact_output_root, trust_root=trust_root)
