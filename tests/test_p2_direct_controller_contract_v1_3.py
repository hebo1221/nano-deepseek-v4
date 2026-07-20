from __future__ import annotations

import copy
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_contract as v1_2  # noqa: E402
import p2_direct_controller_contract_v1_3 as contract  # noqa: E402
import p2_direct_controller_persistent_session_v1_3 as persistent_session  # noqa: E402

TEST_TRUST_ROOT = contract.attestation.TrustRoot(
    key=bytes(range(32)),
    key_id=contract.attestation.derive_key_id(bytes(range(32))),
)


def _calibration() -> dict[str, Any]:
    def cell(budgets: list[list[int]], digest: str) -> dict[str, Any]:
        total = sum(value for _layer, value in budgets)
        return {
            "requested_global_budget": total,
            "csa_layer_count": len(budgets),
            "signal_config": {
                "global_block_budget": total,
                "dense_fallback_block_budget": total,
            },
            "quota": {"layer_budgets": budgets, "calibration_digest": digest},
            "identifiability": {
                "terminal_decision": "GO",
                "all_requested_budgets_exact": True,
                "nonbaseline_gate_passed": True,
                "distinct_quota_gate_passed": True,
                "successful_plan_count": 100,
                "requested_plan_count": 100,
                "failures": [],
            },
            "terminal_decision": "GO",
        }

    payload: dict[str, Any] = {
        "experiment_id": v1_2.DIRECT_CALIBRATION_EXPERIMENT_ID,
        "scale": "s55",
        "training_seed": contract.TRAINING_SEEDS[0],
        "seed": contract.CALIBRATION_SEEDS[0],
        "calibration_seed": contract.CALIBRATION_SEEDS[0],
        "evaluation_seed_reserved": contract.EVALUATION_SEEDS[0],
        "status": "terminal",
        "terminal_decision": "GO",
        "budget_decisions": {"2x": "GO", "4x": "GO"},
        "calibrations": {
            "2x": cell([[2, 2], [4, 4], [6, 6]], "1" * 64),
            "4x": cell([[2, 4], [4, 8], [6, 12]], "2" * 64),
        },
    }
    payload["payload_sha256"] = contract.json_digest(payload)
    return payload


def _validated_admission(calibration: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        payload={
            "quality_evaluation_started": False,
            "evaluation_seed_used_to_initialize_quality_rng": False,
        },
        public_binding={
            "path": "/tmp/v1-3-reuse-admission.json",
            "sha256": "a" * 64,
            "bytes": 1024,
            "experiment_id": "p2-direct-controller-exact-fill-reuse-admission-v1.3",
            "payload_sha256": "b" * 64,
            "attestation_mac": "c" * 64,
            "historical_receipt_sha256": "d" * 64,
            "canonical_nonobservation_sha256": "e" * 64,
        },
        calibrations={("s55", 6071406): SimpleNamespace(path=Path("/tmp/calibration.json"))},
        checkpoints={("s55", 6071406): object()},
    )


def _stub_admission_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, Mapping[str, Any], contract.attestation.TrustRoot, str, int]]:
    calls: list[tuple[object, Mapping[str, Any], contract.attestation.TrustRoot, str, int]] = []

    class FakeAdmissionModule:
        @staticmethod
        def validate_admitted_calibration(
            calibration: Mapping[str, Any],
            *,
            artifact_path: Path,
            admission: object,
            trust_root: contract.attestation.TrustRoot,
            expected_scale: str | None = None,
            expected_training_seed: int | None = None,
        ) -> dict[str, Any]:
            assert expected_scale is not None
            assert expected_training_seed is not None
            calls.append(
                (
                    admission,
                    calibration,
                    trust_root,
                    expected_scale,
                    expected_training_seed,
                )
            )
            assert artifact_path == Path("/tmp/calibration.json")
            return dict(calibration)

    original_import = contract.importlib.import_module

    def importer(name: str) -> object:
        if name == "p2_direct_controller_reuse_admission_v1_3":
            return FakeAdmissionModule
        return original_import(name)

    monkeypatch.setattr(contract.importlib, "import_module", importer)
    return calls


def _build(
    calibration: Mapping[str, Any],
    admission: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return contract.build_direct_controller_arms(
        calibration,
        "2x",
        reuse_admission=admission,
        trust_root=TEST_TRUST_ROOT,
        expected_scale="s55",
        expected_training_seed=contract.TRAINING_SEEDS[0],
        expected_global_block_budget=12,
        expected_csa_layers=(2, 4, 6),
    )


def _contains_threshold_metadata(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            "top_p" in str(key).lower()
            or "top-p" in str(key).lower()
            or _contains_threshold_metadata(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_threshold_metadata(child) for child in value)
    if isinstance(value, str):
        return "top_p" in value.lower() or "top-p" in value.lower()
    return False


def test_exact_fill_arm_projection_and_phase_registry_are_frozen() -> None:
    projected = tuple(
        name for name in v1_2.ALL_ARM_NAMES if name not in v1_2.SENSITIVITY_COMPARATOR_ARMS
    )

    assert contract.ALL_ARM_NAMES == projected
    assert contract.PHASE_A_ARM_NAMES == (
        "hierarchical-soft-lag+pins",
        "fixed+pins",
        "hierarchical-balanced-fixed+pins",
    )
    assert len(contract.PHASE_B_DIAGNOSTIC_ARM_NAMES) == 14
    assert len(contract.ALL_ARM_NAMES) == len(set(contract.ALL_ARM_NAMES)) == 17
    assert contract.EXACT_FILL_ARM_NAMES == contract.ALL_ARM_NAMES
    assert contract.VARIABLE_FILL_SENSITIVITY_ARMS == ()
    assert set(contract.ALL_ARM_NAMES).isdisjoint(contract.REMOVED_VARIABLE_FILL_ARMS)
    assert all(
        contract.EXPECTED_ARM_SEMANTICS[name] == v1_2.EXPECTED_ARM_SEMANTICS[name]
        and contract.EXPECTED_ARM_SEMANTICS[name].fill_mode == "exact-feasible-B"
        and contract.EXPECTED_ARM_SEMANTICS[name].top_p is None
        for name in contract.ALL_ARM_NAMES
    )
    assert contract.json_digest(list(contract.ALL_ARM_NAMES)) == contract.ARM_ORDER_SHA256
    assert (
        contract.json_digest(contract.expected_arm_features())
        == contract.ARM_FEATURES_PROJECTION_SHA256
    )


def test_cardinalities_and_contrast_inventory_are_exact() -> None:
    counts = contract.expected_grid_cardinalities()

    assert counts["budget_shards_total"] == 9_000
    assert counts["phase_a_arm_conversations"] == 540_000
    assert counts["phase_b_additional_arm_conversations"] == 2_520_000
    assert counts["all_arm_conversations"] == counts["quality_outcomes_total"] == 3_060_000
    assert counts["raw_token_rows_without_technical_failures_per_arm"] == 8_120_000
    assert counts["raw_token_rows_without_technical_failures"] == 138_040_000
    assert counts["system_slices_total"] == 15_300
    assert contract.CONFIRMATORY_CONTRAST_COUNT == 3
    assert contract.CAUSAL_DIAGNOSTIC_CONTRAST_COUNT == 15
    assert contract.REGISTERED_QUALITY_CONTRAST_COUNT == 18
    assert contract.TOP_P_QUALITY_INPUT_COUNT == contract.TOP_P_QUALITY_CONTRAST_COUNT == 0
    assert all(
        candidate in contract.ALL_ARM_NAMES and comparator in contract.ALL_ARM_NAMES
        for _name, candidate, comparator in contract.CAUSAL_DIAGNOSTIC_CONTRAST_SPECS
    )
    assert (
        contract.json_digest(
            [
                {"name": name, "candidate": candidate, "comparator": comparator}
                for name, candidate, comparator in contract.CAUSAL_DIAGNOSTIC_CONTRAST_SPECS
            ]
        )
        == contract.CAUSAL_DIAGNOSTIC_CONTRASTS_SHA256
    )


def test_arm_execution_order_is_a_complete_seventeen_way_rotation() -> None:
    orders = [contract.arm_execution_order(index) for index in range(17)]

    assert len(set(orders)) == 17
    assert all(len(order) == 17 and set(order) == set(contract.ALL_ARM_NAMES) for order in orders)
    assert [order[0] for order in orders] == list(contract.ALL_ARM_NAMES)
    assert contract.arm_execution_order(17) == contract.arm_execution_order(0)
    with pytest.raises(ValueError, match="integer"):
        contract.arm_execution_order(True)
    with pytest.raises(ValueError, match="nonnegative"):
        contract.arm_execution_order(-1)


def test_builder_signature_forbids_legacy_comparator_and_strict_inputs() -> None:
    parameters = inspect.signature(contract.build_direct_controller_arms).parameters

    assert "reuse_admission" in parameters
    assert "comparator_matches" not in parameters
    assert "strict_matching" not in parameters


def test_builder_uses_admission_validator_and_returns_only_exact_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_admission_validator(monkeypatch)
    monkeypatch.setattr(
        v1_2,
        "_direct_calibration_validator",
        lambda: pytest.fail("v1.3 must not call the live v1.2 binding validator"),
    )
    calibration = _calibration()
    admission = _validated_admission(calibration)

    arms, metadata = _build(calibration, admission)

    assert calls == [(admission, calibration, TEST_TRUST_ROOT, "s55", contract.TRAINING_SEEDS[0])]
    assert tuple(arms) == contract.ALL_ARM_NAMES
    assert len(arms) == 17
    assert set(arms).isdisjoint(contract.REMOVED_VARIABLE_FILL_ARMS)
    assert all(len(arm.configs) == 1 for arm in arms.values())
    assert all(
        sum(value for _layer, value in arm.configs[0].layer_budgets) == 12 for arm in arms.values()
    )
    assert metadata["exact_fill_arms"] == contract.ALL_ARM_NAMES
    assert metadata["dual_manifest_contexts_validated"] is True
    assert not _contains_threshold_metadata(metadata)
    contract.validate_arm_semantics(arms)


def test_builder_rejects_raw_or_quality_observed_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_admission_validator(monkeypatch)
    calibration = _calibration()

    with pytest.raises(ValueError, match="raw reuse-admission"):
        _build(calibration, {"calibrations": [calibration]})

    observed = _validated_admission(calibration)
    observed.payload["quality_evaluation_started"] = True
    with pytest.raises(ValueError, match="quality remained unstarted"):
        _build(calibration, observed)

    used = _validated_admission(calibration)
    used.payload["evaluation_seed_used_to_initialize_quality_rng"] = True
    with pytest.raises(ValueError, match="quality RNG remained uninitialized"):
        _build(calibration, used)


def test_manifest_builder_is_exact_and_discloses_the_quality_blind_fork() -> None:
    payload = contract.build_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )

    assert contract.validate_manifest_payload(copy.deepcopy(payload)) == payload
    disclosure = payload["lineage_and_adaptation_disclosure"]
    assert disclosure["v1_2_top_p_result_observed_before_freeze"] is True
    assert disclosure["v1_2_held_out_controller_quality_observed"] is False
    assert disclosure["feasibility_outcome_used_to_create_fork"] is True
    assert disclosure["quality_outcome_used_to_create_fork"] is False
    assert disclosure["global_outcome_independence_claimed"] is False
    assert disclosure["evaluation_seed_policy"]["preserved_without_reselection"] is True
    assert disclosure["reused_upstream_evidence"]["training_ledger"]["sha256"] == (
        contract.V1_2_TRAINING_LEDGER_SHA256
    )
    assert disclosure["reused_upstream_evidence"]["calibration_ledger"]["sha256"] == (
        contract.V1_2_CALIBRATION_LEDGER_SHA256
    )
    assert (
        disclosure["reused_upstream_evidence"]["recalibration_after_top_p_result_forbidden"] is True
    )
    assert disclosure["v1_2_quality_roots_at_terminal_report"] == {
        "controller_root": str(contract.V1_2_CONTROLLER_OUTPUT_ROOT),
        "worker_ledger_root": str(contract.V1_2_CONTROLLER_WORKER_ROOT),
        "matrix_lock_path": str(contract.V1_2_CONTROLLER_MATRIX_LOCK_PATH),
        "controller_root_absent": True,
        "worker_ledger_root_absent": True,
        "matrix_lock_path_absent": True,
        "absence_is_pipeline_scoped_not_proof_against_owner_deletion": True,
    }
    assert payload["phases"]["all_arms"] == list(contract.ALL_ARM_NAMES)
    assert payload["grid"]["quality_coordinate_digest"] == contract.quality_coordinate_digest()
    assert payload["phases"]["phase_a_pareto_sensitivities"] == []
    assert payload["execution_contract"]["top_p_schedule_or_match_input_forbidden"] is True
    assert payload["descriptive_feasibility_evidence"]["quality_statistical_input"] is False
    assert payload["artifact_namespaces"] == contract.expected_artifact_namespaces()
    assert contract.MATRIX_LOCK_PATH == contract.OUTPUT_ROOT.parent / (
        f".{contract.OUTPUT_ROOT.name}.{contract.MATRIX_LOCK_SUFFIX}"
    )
    assert contract.WORKER_LEDGER_ROOT == contract.OUTPUT_ROOT.parent / (
        f".{contract.OUTPUT_ROOT.name}.{contract.WORKER_LEDGER_ROOT_SUFFIX}"
    )
    assert contract.REUSE_ADMISSION_PATH == (
        contract.ADMISSION_ROOT / "historical-reuse-admission.json"
    )
    assert contract.ADMISSION_ROOT == contract.OUTPUT_ROOT.parent / (
        f"{contract.OUTPUT_ROOT.name}-admission"
    )
    assert contract.PREHELDOUT_GENESIS_PATH == contract.ADMISSION_ROOT / ("preheldout-genesis.json")


def test_persistent_session_namespaces_and_transport_claims_are_exact() -> None:
    namespaces = contract.expected_artifact_namespaces()
    payload = contract.build_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )

    assert contract.PERSISTENT_SESSION_LEDGER_ROOT == contract.OUTPUT_ROOT.parent / (
        f".{contract.OUTPUT_ROOT.name}.p2-direct-controller-persistent-sessions-v1-3"
    )
    assert contract.PERSISTENT_SESSION_LEDGER_LOCK_PATH == (
        contract.PERSISTENT_SESSION_LEDGER_ROOT.parent
        / f"{contract.PERSISTENT_SESSION_LEDGER_ROOT.name}.lock"
    )
    assert contract.PERSISTENT_SESSION_LEDGER_ROOT.name.startswith(".")
    assert not contract.PERSISTENT_SESSION_LEDGER_ROOT.name.startswith("..")
    absolute_output_root = (
        Path(__file__).resolve().parents[1] / contract.V1_3_1_OUTPUT_ROOT
    ).resolve()
    expected_v1_3_1_suffix = contract.V1_3_1_PERSISTENT_SESSION_LEDGER_ROOT.name.removeprefix(
        f".{contract.V1_3_1_OUTPUT_ROOT.name}."
    )
    assert persistent_session.SESSION_LEDGER_ROOT_SUFFIX == expected_v1_3_1_suffix
    assert (
        persistent_session.session_ledger_root(absolute_output_root)
        == (
            Path(__file__).resolve().parents[1]
            / contract.V1_3_1_PERSISTENT_SESSION_LEDGER_ROOT
        ).resolve()
    )
    assert (
        persistent_session.session_ledger_lock_path(absolute_output_root)
        == (
            Path(__file__).resolve().parents[1]
            / contract.V1_3_1_PERSISTENT_SESSION_LEDGER_LOCK_PATH
        ).resolve()
    )
    assert {
        "plan": persistent_session.PLAN_MESSAGE_TYPE,
        "work": persistent_session.WORK_MESSAGE_TYPE,
        "result": persistent_session.RESULT_MESSAGE_TYPE,
        "receipt": persistent_session.RECEIPT_MESSAGE_TYPE,
    } == {
        "plan": contract.PERSISTENT_SESSION_PLAN_MESSAGE_TYPE,
        "work": contract.PERSISTENT_SESSION_WORK_MESSAGE_TYPE,
        "result": contract.PERSISTENT_SESSION_RESULT_MESSAGE_TYPE,
        "receipt": contract.PERSISTENT_SESSION_RECEIPT_MESSAGE_TYPE,
    }
    assert {
        "launch": persistent_session.LAUNCH_ARTIFACT_TYPE,
        "terminal": persistent_session.TERMINAL_ARTIFACT_TYPE,
    } == {
        "launch": contract.PERSISTENT_SESSION_LAUNCH_ARTIFACT_TYPE,
        "terminal": contract.PERSISTENT_SESSION_TERMINAL_ARTIFACT_TYPE,
    }
    assert {
        "plan": persistent_session.PLAN_ATTESTATION_PURPOSE,
        "work": persistent_session.WORK_ATTESTATION_PURPOSE,
        "result": persistent_session.RESULT_ATTESTATION_PURPOSE,
        "receipt": persistent_session.RECEIPT_ATTESTATION_PURPOSE,
        "launch": persistent_session.LAUNCH_LEDGER_ATTESTATION_PURPOSE,
        "terminal": persistent_session.TERMINAL_LEDGER_ATTESTATION_PURPOSE,
    } == {
        "plan": contract.V1_3_1_PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE,
        "work": contract.V1_3_1_PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE,
        "result": contract.V1_3_1_PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE,
        "receipt": contract.V1_3_1_PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE,
        "launch": contract.V1_3_1_PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE,
        "terminal": contract.V1_3_1_PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE,
    }
    assert namespaces["persistent_session_ledger_root"] == str(
        contract.PERSISTENT_SESSION_LEDGER_ROOT
    )
    assert namespaces["persistent_session_ledger_lock_path"] == str(
        contract.PERSISTENT_SESSION_LEDGER_LOCK_PATH
    )
    assert namespaces["persistent_session_ledger_is_sibling_outside_quality_output_root"] is True
    assert namespaces["persistent_session_message_types"] == {
        "plan": "direct-controller-persistent-session-plan",
        "work": "direct-controller-persistent-session-work-order",
        "result": "direct-controller-persistent-session-work-result",
        "receipt": "direct-controller-persistent-session-receipt",
    }
    assert namespaces["persistent_session_ledger_artifact_types"] == {
        "launch": "direct-controller-persistent-session-launch",
        "terminal": "direct-controller-persistent-session-terminal",
    }
    purposes = namespaces["attestation_purposes"]
    assert {
        key: purposes[key]
        for key in (
            "persistent_session_plan",
            "persistent_session_work",
            "persistent_session_result",
            "persistent_session_receipt",
            "persistent_session_launch_ledger",
            "persistent_session_terminal_ledger",
        )
    } == {
        "persistent_session_plan": (
            "p2-direct-controller-exact-fill-v1-3-persistent-session-plan-v1"
        ),
        "persistent_session_work": (
            "p2-direct-controller-exact-fill-v1-3-persistent-session-work-v1"
        ),
        "persistent_session_result": (
            "p2-direct-controller-exact-fill-v1-3-persistent-session-result-v1"
        ),
        "persistent_session_receipt": (
            "p2-direct-controller-exact-fill-v1-3-persistent-session-receipt-v1"
        ),
        "persistent_session_launch_ledger": (
            "p2-direct-controller-exact-fill-v1-3-persistent-launch-ledger-v1"
        ),
        "persistent_session_terminal_ledger": (
            "p2-direct-controller-exact-fill-v1-3-persistent-terminal-ledger-v1"
        ),
    }

    transport = payload["execution_contract"]["sealed_launch_and_persistent_session"]
    assert transport == contract.expected_transport_and_persistence_contract()
    assert transport["canonical_git_object_launcher_required"] is True
    assert transport["canonical_git_object_launcher_id"] == (
        "p2-direct-controller-git-object-launcher-v1-3"
    )
    assert transport["entrypoint_selector_allowlist"] == ["matrix", "audit", "summary"]
    assert transport["persistent_evaluator_protocol"] == "sealed-hmac-jsonl-model-resident-v1"
    assert transport["durably_evidenced_parent_full_historical_replays_per_launch_authority"] == 1
    assert transport["child_full_historical_evidence_replays_per_session"] == 0
    assert (
        transport["uninterrupted_single_worker_normal_path_checkpoint_model_deserialization_loads"]
        == 10
    )
    assert transport["parent_full_historical_checkpoint_hash_read_bytes"] == "not_measured"
    assert transport["uninterrupted_single_worker_normal_path_conditions"] == [
        "single-worker",
        "complete-9000-shard-matrix",
        "one-successful-session-per-scale-training-seed-cohort",
        "no-controlled-stop",
        "no-interruption-launch-failure-child-eof-parent-commit-failure-or-parent-crash-recovery",
        "no-restart-or-recovery-launch",
    ]
    assert transport["checkpoint_model_deserialization_load_claim_excludes"] == [
        "total-checkpoint-io-bytes",
        "parent-checkpoint-hash-read-bytes",
        "distributed-worker-execution",
        "controlled-stop-execution",
        "restart-or-recovery-execution",
        "failed-or-interrupted-launches",
    ]
    assert transport["durable_runtime_counter_fields_authoritative"] == [
        "launch_attempt_count",
        "checkpoint_model_load_attempt_upper_bound",
        "ready_model_load_count",
        "observed_successful_model_loads",
        "launch_only_interrupted_attempt_count",
        "terminal_count",
        "child_eof_count",
        "published_bundle_reingestion_count",
        "launch_authority_count",
        "durably_evidenced_launch_authority_full_historical_evidence_replay_count",
        "session_triggered_full_historical_evidence_replay_count",
        "normal_no_restart_unique_worker_scale_seed_assignments",
        "normal_path_model_load_bound",
        "normal_path_model_load_bound_applicable",
        "normal_path_model_load_bound_observed_satisfied",
        "additional_controlled_or_recovery_launch_attempts",
        "controlled_stop_session_count",
    ]
    assert transport["durable_runtime_counter_compatibility_aliases"] == {
        "durably_evidenced_parent_full_evidence_replays": (
            "durably_evidenced_launch_authority_full_historical_evidence_replay_count"
        ),
    }
    assert transport["durable_per_session_failure_evidence_authoritative"] == [
        "status",
        "actual_session_argv",
        "child_process_returncode",
    ]


def test_manifest_statistics_and_claim_boundary_do_not_overclaim() -> None:
    payload = contract.build_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )
    statistics = payload["statistical_analysis"]
    gate = payload["confirmatory_success_gate"]
    boundary = payload["claim_boundary"]

    assert statistics["registered_quality_contrast_count"] == 18
    assert statistics["confirmatory_contrast_count"] == 3
    assert statistics["causal_diagnostic_contrast_count"] == 15
    assert statistics["top_p_quality_contrast_count"] == 0
    assert "three registered confirmatory contrasts" in statistics["primary_four_cell_correction"]
    assert gate["required_confirmatory_cells_total"] == 12
    assert gate["required_directional_seed_effects_total"] == 60
    assert gate["top_p_quality_inputs_required"] == 0
    assert (
        "population-significance-at-two-sided-p-less-than-0.05" in boundary["go_does_not_support"]
    )
    assert boundary["forbidden_reporting_label"] == "unchanged-preregistration"


def test_manifest_validator_rejects_arm_seed_and_historical_evidence_drift() -> None:
    payload = contract.build_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )

    arm_drift = copy.deepcopy(payload)
    arm_drift["phases"]["all_arms"].reverse()
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_manifest_payload(arm_drift)

    seed_drift = copy.deepcopy(payload)
    seed_drift["cohort"]["evaluation_seeds"][0] += 1
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_manifest_payload(seed_drift)

    history_drift = copy.deepcopy(payload)
    history_drift["descriptive_feasibility_evidence"]["go_cells"] = 1
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_manifest_payload(history_drift)

    namespace_drift = copy.deepcopy(payload)
    namespace_drift["artifact_namespaces"]["persistent_session_ledger_root"] += "-drift"
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_manifest_payload(namespace_drift)

    transport_drift = copy.deepcopy(payload)
    transport_drift["execution_contract"]["sealed_launch_and_persistent_session"][
        "entrypoint_selector_allowlist"
    ].append("arbitrary")
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_manifest_payload(transport_drift)


def test_manifest_load_fails_closed_before_two_commit_freeze(tmp_path: Path) -> None:
    missing = tmp_path / "not-frozen.json"

    with pytest.raises(RuntimeError, match="quality launch is forbidden"):
        contract.load_manifest(missing)


def test_manifest_load_requires_stable_regular_canonical_bytes(tmp_path: Path) -> None:
    payload = contract.build_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )
    canonical = tmp_path / "manifest.json"
    canonical.write_bytes(contract.canonical_pretty_manifest_bytes(payload))

    assert contract.load_manifest(canonical, verify_implementation=False) == payload

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(contract.canonical_json(payload))
    with pytest.raises(ValueError, match="not canonical pretty JSON"):
        contract.load_manifest(noncanonical, verify_implementation=False)

    symlink = tmp_path / "manifest-link.json"
    symlink.symlink_to(canonical)
    with pytest.raises(RuntimeError, match="absent or not a regular file"):
        contract.load_manifest(symlink, verify_implementation=False)


def test_implementation_inventory_binds_all_inherited_and_new_runtime_paths() -> None:
    assert str(contract.MANIFEST_PATH) not in contract.IMPLEMENTATION_PATHS
    assert contract.PROJECT_DEPENDENCY_SPEC_PATH == contract.IMPLEMENTATION_PATHS[0]
    assert set(v1_2.DIRECT_RESEARCH_IMPLEMENTATION_PATHS).issubset(contract.IMPLEMENTATION_PATHS)
    assert set(contract.V1_3_RESEARCH_IMPLEMENTATION_PATHS).issubset(contract.IMPLEMENTATION_PATHS)
    assert str(contract.V1_2_TOP_P_REPORT_PATH) in contract.PROTOCOL_FREEZE_PATHS
    with pytest.raises(ValueError, match="inventory or ordering drifted"):
        contract.implementation_tree_digest((contract.PROJECT_DEPENDENCY_SPEC_PATH,))
    with pytest.raises(ValueError, match="inventory or ordering drifted"):
        contract.implementation_file_paths((contract.PROJECT_DEPENDENCY_SPEC_PATH,))


def test_no_calibration_only_threshold_generation_api_is_added_to_v1_3_contract() -> None:
    assert "top_p_match_generation_seed" not in contract.__dict__


def test_v1_3_1_activation_amendment_has_distinct_complete_namespaces() -> None:
    namespaces = contract.expected_v1_3_1_artifact_namespaces()

    assert contract.V1_3_1_EXPERIMENT_ID.endswith("v1.3.1")
    assert "v1-3-1" in str(contract.V1_3_1_MANIFEST_PATH)
    assert "v1-3-1" in str(contract.V1_3_1_OUTPUT_ROOT)
    assert "v1-3-1" in str(contract.V1_3_1_ADMISSION_ROOT)
    assert "v1-3-1" in str(contract.V1_3_1_ACTIVATION_ROOT)
    assert contract.V1_3_1_OUTPUT_ROOT != contract.OUTPUT_ROOT
    assert contract.V1_3_1_ADMISSION_ROOT != contract.ADMISSION_ROOT
    assert namespaces["activation_root_exact_member_count"] == 2
    assert namespaces["activation_matrix_lock_path"] == str(
        contract.V1_3_1_ACTIVATION_ROOT / "matrix.lock"
    )
    purposes = namespaces["attestation_purposes"]
    assert set(purposes) == {
        "shard",
        "matrix",
        "worker_ledger",
        "integrity",
        "summary",
        "reuse_admission",
        "preheldout_genesis",
        "quality_start_activation",
        "persistent_session_plan",
        "persistent_session_work",
        "persistent_session_result",
        "persistent_session_receipt",
        "persistent_session_launch_ledger",
        "persistent_session_terminal_ledger",
    }
    assert len(set(purposes.values())) == len(purposes)
    assert all("v1-3-1" in purpose or "v1.3.1" in purpose for purpose in purposes.values())
    assert set(purposes.values()).isdisjoint(
        contract.expected_artifact_namespaces()["attestation_purposes"].values()
    )


def test_v1_3_1_manifest_binds_signed_empty_lineage_and_fixed_start_sequence() -> None:
    payload = contract.build_v1_3_1_manifest_payload(
        attestation_key_id=contract.V1_2_ATTESTATION_KEY_ID,
        implementation_tree_digest="a" * 64,
        implementation_source_commit="b" * 40,
    )

    assert contract.validate_v1_3_1_manifest_payload(copy.deepcopy(payload)) == payload
    assert payload["experiment_id"] == contract.V1_3_1_EXPERIMENT_ID
    assert payload["status"] == contract.V1_3_1_MANIFEST_STATUS
    assert payload["implementation"]["paths"] == list(contract.V1_3_1_IMPLEMENTATION_PATHS)
    lineage = payload["lineage_and_adaptation_disclosure"]["v1_3_signed_empty_lineage"]
    assert lineage == contract.expected_v1_3_1_superseded_empty_lineage()
    assert lineage["quality_state"]["records"] == []
    assert lineage["quality_state"]["completed_shards"] == 0
    assert lineage["reuse_admission"]["sha256"] == (contract.V1_3_SUPERSEDED_ADMISSION_SHA256)
    policy = payload["execution_contract"]["sealed_launch_and_persistent_session"][
        "quality_start_activation"
    ]
    assert policy == contract.expected_v1_3_1_activation_policy()
    assert policy["operational_prefix_max_new_cells"] == 1
    assert policy["zero_prefix_resume_behavior"].startswith("execute-exactly-one")
    assert policy["full_resume_gate"].startswith("authenticated-exact-one")
    assert policy["quality_execution_topology"] == {
        "worker_count": 1,
        "worker_index": 0,
        "distributed_execution_supported": False,
        "reason": "sealed-device-routing-is-not-available-in-v1-3-1",
    }
    assert policy["outcome_values_may_influence_continue_stop_or_configuration"] is False
    assert policy["full_resume_required_after_valid_operational_prefix"] is True
    assert policy["fresh_execution_sequence"][-1].startswith("resume-full-matrix-regardless")

    drift = copy.deepcopy(payload)
    drift["lineage_and_adaptation_disclosure"]["v1_3_signed_empty_lineage"]["reuse_admission"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="content drifted"):
        contract.validate_v1_3_1_manifest_payload(drift)


def test_v1_3_1_inventory_adds_only_the_activation_amendment_report_path() -> None:
    assert contract.V1_3_1_IMPLEMENTATION_PATHS[:-1] == contract.IMPLEMENTATION_PATHS
    assert contract.V1_3_1_IMPLEMENTATION_PATHS[-1] == str(
        contract.V1_3_1_ACTIVATION_AMENDMENT_REPORT_PATH
    )
    assert "2026-07-20" in contract.V1_3_1_IMPLEMENTATION_PATHS[-1]
