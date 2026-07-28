from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_contract_v1_3_5 as contract  # noqa: E402


def _probe_binding(*, worker_count: int = 3) -> dict[str, Any]:
    return {
        "path": "/tmp/adaptive-v4-topology-probe.json",
        "sha256": "1" * 64,
        "bytes": 4096,
        "payload_sha256": "2" * 64,
        "attestation_mac": "3" * 64,
        "candidate_worker_counts": [1, 2, 3, 4],
        "selected_worker_count": worker_count,
        "semantic_equivalence_passed": True,
        "quality_values_accessed": False,
    }


def _manifest(*, worker_count: int = 3, probe_worker_count: int | None = None) -> dict[str, Any]:
    selected_by_probe = worker_count if probe_worker_count is None else probe_worker_count
    return contract.build_v1_3_5_manifest_payload(
        attestation_key_id="a" * 64,
        implementation_tree_digest="b" * 64,
        implementation_source_commit="c" * 40,
        selected_worker_count=worker_count,
        topology_probe_binding=_probe_binding(worker_count=selected_by_probe),
    )


def test_builder_preserves_every_scientific_field() -> None:
    parent = contract._parent_manifest_payload()
    child = _manifest()

    for field in (
        "claim_boundary",
        "cohort",
        "confirmatory_success_gate",
        "descriptive_feasibility_evidence",
        "grid",
        "phases",
        "primary_estimand",
        "schema_version",
        "statistical_analysis",
    ):
        assert child[field] == parent[field]
    assert child["experiment_id"] == contract.V1_3_5_EXPERIMENT_ID
    assert (
        child["lineage_and_adaptation_disclosure"][
            "scientific-grid-arm-estimand-or-success-gate_changed"
        ]
        is False
    )


def test_builder_moves_only_mutable_output_namespaces() -> None:
    child = _manifest(worker_count=4)
    namespaces = child["artifact_namespaces"]
    topology = child["execution_contract"]["sealed_launch_and_persistent_session"][
        "quality_start_activation"
    ]["quality_execution_topology"]

    assert namespaces["output_root"] == str(contract.V1_3_5_OUTPUT_ROOT)
    assert namespaces["activation_root"] == str(contract.V1_3_5_ACTIVATION_ROOT)
    assert namespaces["reuse_admission_path"] == str(contract.base.V1_3_4_REUSE_ADMISSION_PATH)
    assert namespaces["preheldout_genesis_path"] == str(
        contract.base.V1_3_4_PREHELDOUT_GENESIS_PATH
    )
    assert topology["worker_count"] == 4
    assert topology["worker_indices"] == [0, 1, 2, 3]
    assert topology["coordinator_only_publication"] is True


def test_builder_records_explicit_three_worker_override_without_rewriting_probe() -> None:
    child = _manifest(worker_count=3, probe_worker_count=1)
    activation = child["execution_contract"]["sealed_launch_and_persistent_session"][
        "quality_start_activation"
    ]

    assert activation["topology_probe"]["selected_worker_count"] == 1
    assert (
        activation["topology_selection"]
        == contract.V1_3_5_USER_DIRECTED_PARALLEL_OVERRIDE
    )
    assert activation["quality_execution_topology"]["worker_count"] == 3
    assert "balanced-device-block-design" in child["lineage_and_adaptation_disclosure"][
        "amendment_trigger"
    ]


def test_builder_preserves_failed_zero_quality_attempt_in_disjoint_retry_namespace() -> None:
    child = _manifest(worker_count=3, probe_worker_count=1)
    lineage = child["lineage_and_adaptation_disclosure"][
        "v1_3_5_superseded_zero_quality_parallel_attempt"
    ]

    assert lineage == contract.superseded_zero_quality_parallel_attempt()
    assert lineage["quality_state"]["records"] == []
    assert lineage["quality_state"]["completed_shards"] == 0
    assert lineage["quality_state"]["quality_outcomes_materialized"] == 0
    assert lineage["retry"]["quality_outcome_used_to_configure_retry"] is False
    assert lineage["retry"]["output_root"] == str(
        contract.V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT
    )
    assert contract.V1_3_5_OUTPUT_ROOT != (
        contract.V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT
    )
    assert "parallel-retry-1" in lineage["retry"]["output_root"]


def test_builder_preserves_unpublished_schema_failure_without_quality_selection() -> None:
    child = _manifest(worker_count=3, probe_worker_count=1)
    lineage = child["lineage_and_adaptation_disclosure"][
        "v1_3_5_superseded_unpublished_binding_schema_attempt"
    ]

    assert lineage == contract.superseded_unpublished_binding_schema_attempt()
    assert lineage["durable_quality_state"]["matrix_records"] == []
    assert lineage["durable_quality_state"]["published_bundle_count"] == 0
    assert lineage["durable_quality_state"]["orphan_claim_count"] == 1
    assert lineage["volatile_execution_disclosure"][
        "quality_computation_may_have_completed_in_memory"
    ]
    assert not lineage["volatile_execution_disclosure"][
        "quality_values_read_by_supervisor_or_retry_decision"
    ]
    assert lineage["retry"]["output_root"] == str(
        contract.V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT
    )
    assert "parallel-retry-2" in lineage["retry"]["output_root"]


def test_builder_preserves_unpublished_arm_semantics_failure_before_final_namespace() -> None:
    child = _manifest(worker_count=3, probe_worker_count=1)
    lineage = child["lineage_and_adaptation_disclosure"][
        "v1_3_5_superseded_unpublished_arm_semantics_attempt"
    ]

    assert lineage == contract.superseded_unpublished_arm_semantics_attempt()
    assert lineage["ready_only_preflight"]["terminal"]["child_process_returncode"] == 0
    assert lineage["durable_quality_state"]["matrix_records"] == []
    assert lineage["durable_quality_state"]["published_bundle_count"] == 0
    assert lineage["durable_quality_state"]["orphan_claim_count"] == 1
    assert lineage["failure"]["exception"] == "ValueError: Arm semantics drifted."
    assert not lineage["volatile_execution_disclosure"][
        "quality_values_read_by_supervisor_or_retry_decision"
    ]
    assert lineage["retry"]["output_root"] == str(
        contract.V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT
    )
    assert "parallel-final" in lineage["retry"]["output_root"]


def test_builder_preserves_failures_before_final_retry() -> None:
    child = _manifest(worker_count=3, probe_worker_count=1)
    lineage = child["lineage_and_adaptation_disclosure"][
        "v1_3_5_superseded_persistent_reset_attempt"
    ]

    assert lineage == contract.superseded_persistent_reset_attempt()
    assert lineage["closed_world_inventory"]["file_count"] == 45
    assert lineage["durable_quality_state"]["canonical_prefix_shards"] == 3
    assert lineage["durable_quality_state"]["globally_committed_shards"] == 4
    assert lineage["durable_quality_state"]["orphan_claim_count"] == 1
    assert lineage["durable_quality_state"]["complete_uncommitted_bundle_count"] == 1
    assert lineage["failure"]["stage"] == (
        "post-publication-persistent-reset-followed-by-concurrent-drain"
    )
    assert lineage["quality_values_read_by_supervisor_or_retry_decision"] is False
    assert lineage["retry"]["output_root"] == str(
        contract.V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT
    )
    assert "parallel-final-3" in lineage["retry"]["output_root"]
    toctou = child["lineage_and_adaptation_disclosure"][
        "v1_3_5_superseded_live_claim_preflight_attempt"
    ]
    assert toctou == contract.superseded_live_claim_preflight_attempt()
    assert "committed=7;integrity_pass=7;integrity_fail=0" in toctou["durable_state_counts"]
    assert "parallel-final-2" in toctou["output_root"]


def test_builder_rejects_unregistered_probe_override() -> None:
    with pytest.raises(ValueError, match="explicit one-to-three-worker"):
        _manifest(worker_count=2, probe_worker_count=1)


def test_manifest_validation_rejects_topology_or_scientific_tampering() -> None:
    payload = _manifest()
    contract.validate_v1_3_5_manifest_payload(payload, verify_implementation=False)

    topology_tamper = copy.deepcopy(payload)
    topology_tamper["execution_contract"]["sealed_launch_and_persistent_session"][
        "quality_start_activation"
    ]["quality_execution_topology"]["worker_count"] = 2
    with pytest.raises(
        ValueError,
        match="probe binding|canonical builder|explicit one-to-three-worker",
    ):
        contract.validate_v1_3_5_manifest_payload(topology_tamper, verify_implementation=False)

    science_tamper = copy.deepcopy(payload)
    science_tamper["grid"]["replicates"] = [999]
    with pytest.raises(ValueError, match="canonical builder"):
        contract.validate_v1_3_5_manifest_payload(science_tamper, verify_implementation=False)


def test_probe_binding_requires_quality_blind_equivalence() -> None:
    probe = _probe_binding()
    probe["quality_values_accessed"] = True
    with pytest.raises(ValueError, match="Topology probe binding"):
        contract.build_v1_3_5_manifest_payload(
            attestation_key_id="a" * 64,
            implementation_tree_digest="b" * 64,
            implementation_source_commit="c" * 40,
            selected_worker_count=3,
            topology_probe_binding=probe,
        )


def test_parent_manifest_is_bound_by_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = contract._parent_manifest_payload()
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(contract.base, "V1_3_4_MANIFEST_PATH", path)

    with pytest.raises(ValueError, match="predecessor manifest bytes drifted"):
        contract._parent_manifest_payload()


def test_implementation_inventory_expands_package_root_and_matches_head() -> None:
    files = contract.v1_3_5_implementation_file_paths()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert "nano_deepseek_v4/modeling.py" in files
    assert "nano_deepseek_v4" not in files
    assert (
        contract.v1_3_5_implementation_tree_digest()
        == contract.v1_3_5_implementation_tree_digest_at_commit(head)
    )
