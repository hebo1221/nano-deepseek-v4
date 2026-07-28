from __future__ import annotations

import ast
import base64
import hashlib
import importlib.machinery
import inspect
import io
import json
import os
import py_compile
import subprocess
import sys
import typing
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import typing_extensions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_p2_direct_controller_integrity_v1_3 as audit  # noqa: E402
import evaluate_p2_direct_controller_shard_v1_3 as evaluator  # noqa: E402
import generate_p2_direct_controller_v1_3 as generator  # noqa: E402
import p2_direct_controller_contract_v1_3_5 as contract  # noqa: E402
import p2_direct_controller_git_launcher_v1_3 as launcher  # noqa: E402
import run_p2_direct_controller_matrix_v1_3 as matrix  # noqa: E402
import summarize_p2_direct_controller_v1_3 as summary  # noqa: E402

GENERATED_PATHS = {
    "evaluator": SCRIPTS / "evaluate_p2_direct_controller_shard_v1_3.py",
    "matrix": SCRIPTS / "run_p2_direct_controller_matrix_v1_3.py",
    "audit": SCRIPTS / "audit_p2_direct_controller_integrity_v1_3.py",
    "summary": SCRIPTS / "summarize_p2_direct_controller_v1_3.py",
}


def _source(name: str) -> str:
    return GENERATED_PATHS[name].read_text(encoding="utf-8")


def test_persistent_reset_allows_only_initial_allocation_stabilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_allocations = iter((120, 121))
    monkeypatch.setattr(evaluator, "_persistent_model_state", lambda _model: ())
    monkeypatch.setattr(evaluator.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(evaluator.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        evaluator.torch.cuda,
        "memory_allocated",
        lambda _device: next(observed_allocations),
    )
    monkeypatch.setattr(evaluator.gc, "collect", lambda: 0)

    stabilized = evaluator._reset_persistent_model_state(
        object(),
        device=object(),
        expected_state=(),
        expected_allocated_bytes=100,
        allow_initial_allocation_stabilization=True,
    )
    assert stabilized == 120
    with pytest.raises(ValueError, match="allocation baseline"):
        evaluator._reset_persistent_model_state(
            object(),
            device=object(),
            expected_state=(),
            expected_allocated_bytes=stabilized,
            allow_initial_allocation_stabilization=False,
        )


@pytest.mark.parametrize("unknown_temporary", (False, True))
def test_distributed_preflight_closes_live_claim_scan_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unknown_temporary: bool
) -> None:
    coordinate = matrix.coordinates()[0]
    monkeypatch.setattr(matrix, "coordinates", lambda: (coordinate,))
    output_root = (tmp_path / "output").resolve()
    summary = output_root / matrix.MATRIX_SUMMARY_NAME
    output_dir = matrix.shard_output_dir(output_root, coordinate)
    output_dir.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")
    outcomes_name = matrix.canonical_bundle_paths(output_root, coordinate)["outcomes"].name
    late_path = output_dir / (
        "unknown-race.tmp" if unknown_temporary else f".{outcomes_name}.race.tmp"
    )
    original_rglob = Path.rglob

    def racing_rglob(path: Path, pattern: str) -> typing.Any:
        if path == output_root:
            late_path.write_bytes(b"concurrent")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", racing_rglob)
    expectation = pytest.raises(ValueError, match="unregistered orphan") if unknown_temporary else nullcontext()
    with matrix._exclusive_cell_claim(
        output_dir, coordinate=coordinate, launch_nonce="a" * 64, worker_index=0, worker_count=1
    ), expectation:
        matrix._preflight_distributed_output_tree(
            output_root=output_root, matrix_summary=summary, merged_records={}, worker_count=1
        )


def _imports(source: str) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            result.add(node.module)
    return result


def _sealed_common_provenance() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    authority: dict[str, object] = {
        "schema_version": 1,
        "launcher": "p2-direct-controller-git-object-launcher-v1-3-5",
        "sealed_runner": True,
        "sealed_inventory": True,
    }
    runtime: dict[str, object] = {
        "schema_version": 1,
        "implementation": "cpython",
        "cache_tag": "cpython-test",
        "version": [3, 11, 0, "final", 0],
        "executable": "/sealed/python",
        "venv_executable": "/sealed/python",
        "resolved_executable": "/sealed/python",
        "executable_sha256": "1" * 64,
        "executable_metadata": {},
        "base_prefix": "/sealed",
        "site_packages": "/sealed/site-packages",
    }
    source: dict[str, object] = {
        "schema_version": 1,
        "launcher": "p2-direct-controller-git-object-launcher-v1-3-5",
        "repository_root": str(REPOSITORY_ROOT.resolve()),
        "bundle_sha256": "2" * 64,
        "pinned_head_oid": "3" * 40,
        "frozen_source_commit": "4" * 40,
        "implementation_tree_digest": "5" * 64,
        "head_manifest": {},
    }
    return authority, runtime, source


def _sealed_route(selector: str, *, sha_digit: str = "6") -> dict[str, object]:
    paths = {
        "matrix": "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix_v1_3.py",
        "audit": "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity_v1_3.py",
        "summary": "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller_v1_3.py",
    }
    return {
        "schema_version": 1,
        "launcher": "p2-direct-controller-git-object-launcher-v1-3-5",
        "entrypoint_selector": selector,
        "entrypoint_relative_path": paths[selector],
        "source_bundle_sha256": "2" * 64,
        "git_mode": "100644",
        "git_blob_oid": "7" * 40,
        "sha256": sha_digit * 64,
        "bytes": 123,
    }


def test_generator_reproduces_all_committed_v1_3_sources() -> None:
    generator.check_generated_sources()

    expected = generator.generated_sources()
    assert set(expected) == set(GENERATED_PATHS.values())
    assert all(path.read_text(encoding="utf-8") == source for path, source in expected.items())
    assert all(
        source.startswith(
            "from __future__ import annotations\n\n"
            "# Generated by generate_p2_direct_controller_v1_3.py"
        )
        for source in expected.values()
    )
    assert all("V1_3_1" not in source for source in expected.values())
    assert all("v1-3-1" not in source for source in expected.values())
    assert all("V1_3_5_1" not in source for source in expected.values())
    assert all("v1_3_5_1" not in source for source in expected.values())
    evaluator_source = expected[GENERATED_PATHS["evaluator"]]
    matrix_source = expected[GENERATED_PATHS["matrix"]]
    assert 'globals().get("SEALED_SOURCE_PROVENANCE_V1_3_5")' in evaluator_source
    assert 'globals().get("SEALED_LAUNCH_ROUTING_V1_3_5")' in evaluator_source
    assert 'globals().get("SEALED_LAUNCH_ROUTING_V1_3_5")' in matrix_source
    assert "exact-fill-v1-3-worker-" not in matrix_source
    assert "exact-fill-v1-3-5-worker-" in matrix_source


@pytest.mark.parametrize("arm_name", contract.ALL_ARM_NAMES)
def test_success_outcome_accepts_json_round_tripped_arm_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm_name: str,
) -> None:
    observed_config = object()
    monkeypatch.setattr(
        evaluator,
        "_runtime_config_from_payload",
        lambda _payload: observed_config,
    )
    example = {
        "example_index": 0,
        "schedule_index": 7,
        "prefix_length": 2,
        "input_token_count": 4,
        "targets": [1, 2],
    }
    runtime_config = {"fixture": "schema-only"}
    row = evaluator._seal_row(
        evaluator.OUTCOME_SUCCESS_SCHEMA_ID,
        {
            "example_index": 0,
            "arm": arm_name,
            "execution_index": 0,
            "schedule_index": 7,
            "status": "success",
            "config_variant": "schema-only",
            "semantics": evaluator.asdict(contract.EXPECTED_ARM_SEMANTICS[arm_name]),
            "runtime_config": runtime_config,
            "config_sha256": contract.json_digest(runtime_config),
            "prefix_length": 2,
            "decoded_tokens": 2,
            "wall_time_ns": 2,
            "tokens_per_second": 1_000_000_000.0,
            "predictions": [1, 3],
            "correct": [True, False],
            "correct_count": 1,
            "total": 2,
            "accuracy": 0.5,
            "all_queries_correct": False,
            "token_rows_digest": "a" * 64,
        },
    )
    spool = evaluator.DeterministicJsonlGzipSpool(
        kind="outcomes",
        final_path=tmp_path / f"{arm_name}.outcomes.jsonl.gz",
    )
    try:
        spool.write(row)
        finalized = spool.finish()
        reopened = list(
            evaluator._iter_sidecar_rows_from_path(
                finalized.temporary_path,
                kind="outcomes",
                binding=finalized.binding,
            )
        )
        assert len(reopened) == 1
        result = evaluator._validate_success_outcome(
            reopened[0],
            example=example,
            arm_name=arm_name,
            execution_index=0,
            calibration=None,
            arms=None,
            budget="50",
        )
    finally:
        spool.abort()

    assert result is observed_config


def test_generated_operational_sources_contain_no_superseded_live_namespace() -> None:
    stale_tokens = (
        "V1_3_2",
        "v1_3_2",
        "V1.3.2",
        "v1.3.2",
        "V1-3-2",
        "v1-3-2",
        "V1_3_3",
        "v1_3_3",
        "V1.3.3",
        "v1.3.3",
        "V1-3-3",
        "v1-3-3",
    )
    for path, source in generator.generated_sources().items():
        for token in stale_tokens:
            assert token not in source, f"stale operational namespace {token!r} in {path}"


def test_generator_pins_and_preserves_every_v1_2_source_byte_for_byte() -> None:
    for source_name, expected_sha256 in generator.SOURCE_SHA256.items():
        source = SCRIPTS / source_name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256


def test_generated_module_identity_and_paths_are_contract_derived() -> None:
    assert evaluator.EXPERIMENT_ID == contract.V1_3_5_SHARD_EXPERIMENT_ID
    assert evaluator.ATTESTATION_PURPOSE == contract.V1_3_5_SHARD_ATTESTATION_PURPOSE
    assert matrix.EXPERIMENT_ID == contract.V1_3_5_MATRIX_EXPERIMENT_ID
    assert matrix.WORKER_EXPERIMENT_ID == contract.V1_3_5_WORKER_LEDGER_EXPERIMENT_ID
    assert matrix.MATRIX_ATTESTATION_PURPOSE == contract.V1_3_5_MATRIX_ATTESTATION_PURPOSE
    assert matrix.WORKER_ATTESTATION_PURPOSE == contract.V1_3_5_WORKER_LEDGER_ATTESTATION_PURPOSE
    assert audit.EXPERIMENT_ID == contract.V1_3_5_INTEGRITY_EXPERIMENT_ID
    assert audit.ATTESTATION_PURPOSE == contract.V1_3_5_INTEGRITY_ATTESTATION_PURPOSE
    assert summary.EXPERIMENT_ID == contract.V1_3_5_SUMMARY_EXPERIMENT_ID
    assert summary.ATTESTATION_PURPOSE == contract.V1_3_5_SUMMARY_ATTESTATION_PURPOSE
    assert matrix.OUTPUT_ROOT == contract.V1_3_5_OUTPUT_ROOT
    assert matrix.MATRIX_SUMMARY_NAME == contract.MATRIX_SUMMARY_NAME
    assert matrix.MATRIX_SUMMARY == contract.V1_3_5_MATRIX_SUMMARY_PATH
    assert (
        matrix._matrix_lock_path(matrix.OUTPUT_ROOT)
        == contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.resolve()
    )
    assert (
        matrix._worker_ledger_root(matrix.OUTPUT_ROOT)
        == contract.V1_3_5_WORKER_LEDGER_ROOT.resolve()
    )
    assert matrix.REUSE_ADMISSION_PATH == contract.V1_3_5_REUSE_ADMISSION_PATH
    assert matrix.PREHELDOUT_GENESIS_PATH == contract.V1_3_5_PREHELDOUT_GENESIS_PATH
    assert contract.V1_3_5_ADMISSION_ROOT.parent == contract.V1_3_5_OUTPUT_ROOT.parent
    assert contract.V1_3_5_ADMISSION_ROOT != contract.V1_3_5_OUTPUT_ROOT
    assert not contract.V1_3_5_REUSE_ADMISSION_PATH.is_relative_to(contract.V1_3_5_OUTPUT_ROOT)
    assert not contract.V1_3_5_PREHELDOUT_GENESIS_PATH.is_relative_to(contract.V1_3_5_OUTPUT_ROOT)


def test_generated_modules_import_only_the_v1_3_controller_pipeline() -> None:
    assert "p2_direct_controller_contract_v1_3_5" in _imports(_source("evaluator"))
    assert "p2_direct_controller_topology_v1_3_5" in _imports(_source("evaluator"))
    assert "p2_direct_controller_contract" not in _imports(_source("evaluator"))

    assert "p2_direct_controller_contract_v1_3_5" in _imports(_source("matrix"))
    assert "p2_direct_controller_topology_v1_3_5" in _imports(_source("matrix"))
    assert "run_p2_direct_top_p_physical_matrix" not in _imports(_source("matrix"))

    assert "run_p2_direct_controller_matrix_v1_3" in _imports(_source("audit"))
    assert "evaluate_p2_direct_controller_shard_v1_3" in _imports(_source("summary"))
    assert "audit_p2_direct_controller_integrity_v1_3" in _imports(_source("summary"))


def test_external_top_p_quality_inputs_are_absent_from_generated_pipeline() -> None:
    evaluator_source = _source("evaluator")
    matrix_source = _source("matrix")
    audit_source = _source("audit")
    summary_source = _source("summary")

    assert "top_p_match_artifacts" not in evaluator_source
    assert "--top-p-match" not in evaluator_source
    assert "comparator_matches" not in evaluator_source
    assert "strict_matching" not in evaluator_source
    assert '"variable_fill_arms"' not in evaluator_source
    assert '"runtime_soft_lag_snapshot_forbidden_for_variable_fill"' not in evaluator_source
    assert "--reuse-admission" in evaluator_source
    assert '"reuse_admission"' in evaluator_source
    assert "--preheldout-genesis" in evaluator_source
    assert '"preheldout_genesis"' in evaluator_source

    assert "top_p" not in matrix_source.lower()
    assert "top-p" not in matrix_source.lower()
    assert "top_p" not in audit_source.lower()
    assert "top-p" not in audit_source.lower()
    assert "top_p" not in summary_source.lower()
    assert "top-p" not in summary_source.lower()
    assert "--reuse-admission" in matrix_source
    assert "--reuse-admission" in audit_source
    assert "--preheldout-genesis" in matrix_source
    assert "--preheldout-genesis" in audit_source


def test_exact_v1_3_cardinalities_and_contrasts_are_wired_end_to_end() -> None:
    assert matrix.ARM_NAMES == contract.ALL_ARM_NAMES
    assert len(matrix.ARM_NAMES) == 17
    assert matrix.EXPECTED_SHARDS == contract.BUDGET_SHARDS_TOTAL == 9_000
    assert matrix.EXPECTED_OUTCOME_ROWS == contract.QUALITY_OUTCOMES_TOTAL == 3_060_000
    assert contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES == 138_040_000
    assert contract.SYSTEM_SLICES_TOTAL == 15_300
    assert len(summary.ALL_CONTRASTS) == contract.REGISTERED_QUALITY_CONTRAST_COUNT == 18
    assert "all_20_by_17_outcome_rows_verified" in audit.INTEGRITY_CHECK_FIELDS
    assert all("20_by_19" not in field for field in audit.INTEGRITY_CHECK_FIELDS)


def test_matrix_zero_prefix_binds_both_preheldout_artifacts() -> None:
    source = _source("matrix")
    assert '"reuse_admission": dict(reuse_admission.public_binding)' in source
    assert '"preheldout_genesis": dict(preheldout_genesis.public_binding)' in source
    assert '"prerequisites": dict(prerequisites.public_binding)' in source
    single = inspect.getsource(matrix.run_matrix)
    assert single.index("load_and_validate_prestart_prerequisites(") < single.index(
        "_capture_selected_device_context("
    )
    assert single.index("load_and_validate_prerequisites(") < single.index(
        "_capture_selected_device_context("
    )
    assert "load_validated_reuse_admission(" not in source
    assert "load_validated_preheldout_genesis(" not in source


def test_nine_thousand_schedule_positions_are_balanced_at_every_rank() -> None:
    rank_counts = [Counter() for _ in contract.ALL_ARM_NAMES]
    first_counts: Counter[str] = Counter()
    for schedule_index in range(9_000):
        order = evaluator.arm_execution_order(schedule_index)
        assert len(order) == len(set(order)) == 17
        assert set(order) == set(contract.ALL_ARM_NAMES)
        first_counts[order[0]] += 1
        for rank, arm in enumerate(order):
            rank_counts[rank][arm] += 1

    assert set(first_counts.values()) == {529, 530}
    assert all(set(counts.values()) == {529, 530} for counts in rank_counts)
    assert evaluator.arm_execution_order(17) == evaluator.arm_execution_order(0)


def test_evaluator_input_schema_requires_admission_and_genesis_without_top_p() -> None:
    artifact_binding = {
        "path": "/tmp/calibration.json",
        "bytes": 1,
        "sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "attestation_mac": "c" * 64,
        "experiment_id": "calibration",
    }
    reuse_binding = {
        "path": "/tmp/admission.json",
        "bytes": 1,
        "sha256": "d" * 64,
        "payload_sha256": "e" * 64,
        "attestation_mac": "f" * 64,
        "experiment_id": "admission",
        "historical_receipt_sha256": "1" * 64,
        "canonical_nonobservation_sha256": "2" * 64,
        "superseded_failure_lineage_sha256": "c" * 64,
        "superseded_failure_lineage_projection_sha256": "d" * 64,
    }
    genesis_binding = {
        "path": "/tmp/genesis.json",
        "bytes": 1,
        "sha256": "3" * 64,
        "payload_sha256": "4" * 64,
        "attestation_mac": "5" * 64,
        "experiment_id": "genesis",
    }
    activation_binding = {
        "path": "/tmp/activation.json",
        "sha256": "6" * 64,
        "bytes": 1,
        "experiment_id": "activation",
        "payload_sha256": "7" * 64,
        "attestation_mac": "8" * 64,
        "activation_root": "/tmp/activation",
        "matrix_lock_path": "/tmp/activation/matrix.lock",
        "matrix_lock_device": 1,
        "matrix_lock_inode": 2,
        "base_prerequisites_sha256": "9" * 64,
        "sealed_source_bundle_sha256": "a" * 64,
        "sealed_launch_routing_sha256": "b" * 64,
        "selected_worker_count": 3,
        "topology_probe_payload_sha256": "c" * 64,
    }
    producer_binding = evaluator.admission._activation_public_binding(
        Path("/tmp/activation.json"),
        {
            "attestation": {"mac": "8" * 64},
            "matrix_lock_binding": {
                "path": "/tmp/activation/matrix.lock",
                "device": 1,
                "inode": 2,
            },
            "experiment_id": "activation",
            "payload_sha256": "7" * 64,
            "canonical_root": "/tmp/activation",
            "base_prerequisites_sha256": "9" * 64,
            "sealed_source_provenance": {"bundle_sha256": "a" * 64},
            "sealed_launch_routing": {"route": "binding"},
            "selected_worker_count": 3,
            "topology_probe": {"payload_sha256": "c" * 64},
        },
        b"activation",
    )
    assert set(activation_binding) == set(producer_binding)
    source = {
        "source": {},
        "manifest": {},
        "checkpoint": {},
        "training_summary": {},
        "calibration_artifact": artifact_binding,
        "reuse_admission": reuse_binding,
        "preheldout_genesis": genesis_binding,
        "quality_start_activation": activation_binding,
    }
    inputs = {**source, "input_binding_digest": contract.json_digest(source)}

    assert evaluator._validate_inputs_structure(inputs) == inputs
    legacy = dict(inputs)
    legacy["top_p_match_artifacts"] = {}
    legacy["input_binding_digest"] = evaluator._input_binding_digest(legacy)
    with pytest.raises(ValueError, match="schema drifted"):
        evaluator._validate_inputs_structure(legacy)


def test_admission_failure_precedes_every_evaluator_cuda_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AdmissionRejected(RuntimeError):
        pass

    arguments = SimpleNamespace(
        launch_nonce="a" * 64,
        manifest=Path("manifest.json"),
        checkpoint=Path("checkpoint.pt"),
        training_summary=Path("training.json"),
        training_matrix_summary=Path("training-matrix.json"),
        calibration=Path("calibration.json"),
        reuse_admission=Path("admission.json"),
        preheldout_genesis=Path("genesis.json"),
        quality_start_activation=Path("activation.json"),
        scale=contract.SCALES[0],
        training_seed=contract.TRAINING_SEEDS[0],
        budget=contract.BUDGETS[0],
        family=contract.FAMILIES[0],
        context=contract.CONTEXTS[0],
        replicate=contract.REPLICATES[0],
        output=Path("output.json"),
        device="cuda",
        expected_device_identity_type="uuid",
        expected_device_identity="GPU-test",
        dtype=evaluator.DTYPE_NAME,
        persistent_session=False,
    )
    context = SimpleNamespace(manifest_binding={"attestation": {"key_id": "b" * 64}})
    cuda_probes = 0

    def reject_inputs(**_kwargs: object) -> None:
        raise AdmissionRejected

    def probe(*_args: object, **_kwargs: object) -> None:
        nonlocal cuda_probes
        cuda_probes += 1

    monkeypatch.setattr(
        evaluator.argparse.ArgumentParser,
        "parse_args",
        lambda _self: arguments,
    )
    monkeypatch.setattr(evaluator, "_assert_repository_import_origins", lambda: None)
    monkeypatch.setattr(
        evaluator.admission,
        "establish_v1_3_5_quality_context",
        lambda *_a, **_k: context,
    )
    monkeypatch.setattr(
        evaluator.attestation,
        "trust_root_from_inherited_environment",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(evaluator, "establish_evaluator_inputs", reject_inputs)
    monkeypatch.setattr(evaluator, "_runtime_environment", probe)

    with pytest.raises(AdmissionRejected):
        evaluator.main()
    assert cuda_probes == 0


def test_child_uses_scoped_activation_while_parent_owns_full_replay() -> None:
    evaluator_source = _source("evaluator")
    matrix_source = _source("matrix")

    assert evaluator_source.count("load_activated_consumer_authority(") == 2
    assert "load_activated_quality_authority(" not in evaluator_source
    assert "require_activated_consumer_authority(" in evaluator_source
    assert matrix_source.count("load_activated_quality_authority(") == 1
    assert "revalidate_activated_quality_authority(" in matrix_source
    assert "validate_admitted_calibration(" in evaluator_source
    assert "load_admitted_checkpoint(" in evaluator_source


def test_external_cache_loads_one_scoped_authority_for_two_budget_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root = SimpleNamespace(key_id="f" * 64)
    shared_inputs = {
        "source": {"source": "frozen"},
        "manifest": {"manifest": "frozen"},
        "checkpoint": {"checkpoint": "frozen"},
        "training_summary": {"training": "frozen"},
        "calibration_artifact": {"calibration": "frozen"},
        "reuse_admission": {"admission": "frozen"},
        "preheldout_genesis": {"genesis": "frozen"},
        "quality_start_activation": {"activation": "frozen"},
    }
    inputs_by_budget = (
        {**shared_inputs, "input_binding_digest": "a" * 64},
        {**shared_inputs, "input_binding_digest": "b" * 64},
    )
    coordinates = tuple(
        {
            "scale": contract.SCALES[0],
            "training_seed": contract.TRAINING_SEEDS[0],
            "calibration_seed": contract.CALIBRATION_SEEDS[0],
            "evaluation_seed": contract.EVALUATION_SEEDS[0],
            "budget": budget,
            "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS[contract.SCALES[0]][budget],
            "csa_layers": list(contract.DIRECT_CSA_LAYERS_BY_SCALE[contract.SCALES[0]]),
        }
        for budget in contract.BUDGETS
    )
    assert len(coordinates) == 2
    calls = SimpleNamespace(authority=0, validation=0)

    def consumer() -> SimpleNamespace:
        return SimpleNamespace(
            activation=SimpleNamespace(public_binding={"activation": "frozen"}),
            reuse_admission=SimpleNamespace(public_binding={"admission": "frozen"}),
            preheldout_genesis=SimpleNamespace(public_binding={"genesis": "frozen"}),
        )

    def load_authority(*_args: object, **_kwargs: object) -> tuple[object, object]:
        calls.authority += 1
        return SimpleNamespace(), consumer()

    def validate_inputs(
        _inputs: object, coordinate: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        calls.validation += 1
        assert kwargs.get("_consumer_authority") is not None
        budget = coordinate["budget"]
        return ({"budget": budget}, {"arm": budget}, {"metadata": budget})

    monkeypatch.setattr(evaluator, "_load_external_consumer_authority", load_authority)
    monkeypatch.setattr(evaluator, "_validate_external_inputs", validate_inputs)
    cache = evaluator.DirectControllerExternalValidationCache()
    for inputs, coordinate in zip(inputs_by_budget, coordinates, strict=True):
        cache.validated_external_inputs(inputs, coordinate, trust_root=trust_root)
    assert cache.authority_count == 1
    assert cache.entry_count == 2
    assert calls.authority == 1
    assert calls.validation == 2

    for _ in range(5):
        for inputs, coordinate in zip(inputs_by_budget, coordinates, strict=True):
            cache.validated_external_inputs(inputs, coordinate, trust_root=trust_root)
    assert calls.authority == 1
    assert calls.validation == 2

    cache.assert_unchanged(trust_root=trust_root)
    assert calls.authority == 2
    assert calls.validation == 4


@pytest.mark.parametrize("with_callbacks", (False, True))
def test_integrity_iterator_shares_one_nonnull_cache_for_every_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_callbacks: bool,
) -> None:
    paths = (tmp_path / "one.json", tmp_path / "two.json")
    bindings = {
        str(path): {"path": str(path), "sha256": str(index) * 64}
        for index, path in enumerate(paths, start=1)
    }
    inventory = [
        {
            "envelope": bindings[str(path)],
            "matrix_record": {"coordinate_key": path.stem},
        }
        for path in paths
    ]
    cache_instances: list[object] = []
    observed_caches: list[object] = []
    callback_rows: list[tuple[str, str]] = []

    class Cache:
        def __init__(self) -> None:
            self.finalized = 0
            cache_instances.append(self)

        def assert_unchanged(self, *, trust_root: object) -> None:
            assert trust_root is sentinel_trust_root
            self.finalized += 1

    def consume(path: Path, **kwargs: object) -> dict[str, object]:
        cache = kwargs["external_validation_cache"]
        assert cache is not None
        observed_caches.append(cache)
        for name in ("outcome_callback", "token_callback", "failure_callback"):
            callback = kwargs[name]
            assert callable(callback)
            callback({"kind": name})
        return {"path": str(path)}

    sentinel_trust_root = object()
    monkeypatch.setattr(
        audit,
        "_evaluator_module",
        lambda: SimpleNamespace(
            consume_validated_direct_controller_shard=consume,
            DirectControllerExternalValidationCache=Cache,
        ),
    )
    monkeypatch.setattr(
        audit,
        "_file_binding",
        lambda path, **_kwargs: bindings[str(path)],
    )
    callbacks = {}
    if with_callbacks:
        callbacks = {
            "outcome_callback": lambda record, _row: callback_rows.append(
                (record["coordinate_key"], "outcome")
            ),
            "token_callback": lambda record, _row: callback_rows.append(
                (record["coordinate_key"], "token")
            ),
            "failure_callback": lambda record, _row: callback_rows.append(
                (record["coordinate_key"], "failure")
            ),
        }
    yielded = list(
        audit.iter_validated_raw_shards(
            {"bundle_inventory": inventory},
            trust_root=sentinel_trust_root,
            **callbacks,
        )
    )
    assert len(yielded) == 2
    assert len(cache_instances) == 1
    assert cache_instances[0].finalized == 1
    assert observed_caches == [cache_instances[0], cache_instances[0]]
    assert len(callback_rows) == (6 if with_callbacks else 0)


def test_runner_validates_admission_and_genesis_before_device_queries() -> None:
    single = inspect.getsource(matrix.run_matrix)

    assert single.index("load_and_validate_prestart_prerequisites(") < single.index(
        "_capture_selected_device_context("
    )
    assert single.index("load_and_validate_prerequisites(") < single.index(
        "_capture_selected_device_context("
    )
    assert single.index(
        "_validate_quality_mutating_lock_path(", single.index("device_guard_path")
    ) < single.index("_prepare_quality_start_authority(")


def test_child_uses_fresh_pycache_prefix_and_rejects_rogue_package_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nonce = "6" * 64
    prefix = matrix._isolated_evaluator_pycache_prefix(nonce)
    assert prefix.parent == Path("/tmp")
    assert nonce in prefix.name
    assert not prefix.exists()
    command = [
        sys.executable,
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={prefix}",
        "-c",
        "pass",
        "--launch-nonce",
        nonce,
    ]
    with matrix._fresh_evaluator_pycache(command) as active_prefix:
        assert active_prefix == prefix
        assert active_prefix.is_dir()
    assert not prefix.exists()
    matrix_source = _source("matrix")
    assert '"-B"' in matrix_source
    assert '"-X"' in matrix_source
    assert "pycache_prefix=" in matrix_source

    rogue = REPOSITORY_ROOT / "nano_deepseek_v4/_ignored_rogue_v1_3_test.py"
    rogue.write_text("ROGUE = True\n", encoding="utf-8")
    try:
        site_packages = Path(
            str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"])
        )
        monkeypatch.setattr(
            evaluator,
            "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
            str(site_packages),
            raising=False,
        )
        monkeypatch.setattr(
            evaluator.contract,
            "v1_3_5_implementation_file_paths",
            lambda: (
                "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard_v1_3.py",
            ),
        )
        rogue_module = ModuleType("nano_deepseek_v4._ignored_rogue_v1_3_test")
        rogue_module.__file__ = str(rogue)
        with pytest.raises(ValueError, match="outside the frozen implementation inventory"):
            evaluator._assert_repository_import_origins({rogue_module.__name__: rogue_module})
    finally:
        rogue.unlink(missing_ok=True)

    def inventory_row(root: Path, path: Path) -> dict[str, object]:
        data = path.read_bytes()
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        return {
            "path": path.relative_to(root).as_posix(),
            "git_mode": "100755" if path.stat().st_mode & 0o111 else "100644",
            "git_blob_oid": hashlib.sha1(blob).hexdigest(),  # noqa: S324 - Git object ID
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    def run_bootstrap(
        *, root: Path, files: list[dict[str, object]], source: bytes, script: Path
    ) -> subprocess.CompletedProcess[str]:
        manifest_path = root / ".bootstrap-head-manifest.json"
        manifest_created = False
        if not manifest_path.exists():
            manifest_path.write_bytes(b"{}\n")
            manifest_created = True
        manifest_bytes = manifest_path.read_bytes()
        manifest_blob = b"blob " + str(len(manifest_bytes)).encode("ascii") + b"\0" + manifest_bytes
        runtime = launcher._python_runtime_binding(REPOSITORY_ROOT)
        source_provenance = {
            "schema_version": 1,
            "launcher": launcher.LAUNCHER_ID,
            "repository_root": str(root),
            "bundle_sha256": "1" * 64,
            "pinned_head_oid": "2" * 40,
            "frozen_source_commit": "2" * 40,
            "implementation_tree_digest": "3" * 64,
            "head_manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "git_mode": "100644",
                "git_blob_oid": hashlib.sha1(manifest_blob).hexdigest(),  # noqa: S324
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "bytes": len(manifest_bytes),
                "source_base64": base64.b64encode(manifest_bytes).decode("ascii"),
            },
        }
        inventory_source = {
            "repository_root": str(root),
            "files": sorted(files, key=lambda row: str(row["path"])),
            "python_runtime": runtime,
            "source_provenance": source_provenance,
        }
        inventory = {
            **inventory_source,
            "digest": hashlib.sha256(
                json.dumps(inventory_source, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        evaluator_fd = os.memfd_create("v1-3-bootstrap-test-evaluator", os.MFD_CLOEXEC)
        inventory_fd = os.memfd_create("v1-3-bootstrap-test-inventory", os.MFD_CLOEXEC)
        try:
            os.write(evaluator_fd, source)
            os.lseek(evaluator_fd, 0, os.SEEK_SET)
            os.write(
                inventory_fd,
                json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode(),
            )
            os.lseek(inventory_fd, 0, os.SEEK_SET)
            environment = os.environ.copy()
            environment[matrix.EVALUATOR_FD_ENV] = str(evaluator_fd)
            environment[matrix.IMPORT_INVENTORY_FD_ENV] = str(inventory_fd)
            return subprocess.run(
                [
                    str(runtime["executable"]),
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    matrix.EVALUATOR_FD_BOOTSTRAP,
                    str(script),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                pass_fds=(evaluator_fd, inventory_fd),
            )
        finally:
            os.close(inventory_fd)
            os.close(evaluator_fd)
            if manifest_created:
                manifest_path.unlink(missing_ok=True)

    bootstrap_site_packages = Path(
        str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"])
    )
    bootstrap_global_helper = tmp_path / "bootstrap_global_helper.py"
    bootstrap_global_helper.write_text("VALUE = True\n", encoding="utf-8")
    sealed_global_result = run_bootstrap(
        root=tmp_path,
        files=[inventory_row(tmp_path, bootstrap_global_helper)],
        source=b"print(_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5)\n",
        script=tmp_path / "sealed_evaluator.py",
    )
    assert sealed_global_result.returncode == 0, sealed_global_result.stderr
    assert sealed_global_result.stdout.strip() == str(bootstrap_site_packages)

    package = tmp_path / "bootstrap_guard_package"
    package.mkdir()
    package_init = package / "__init__.py"
    package_init.write_text("", encoding="utf-8")
    marker = tmp_path / "rogue-side-effect-observed"
    rogue_submodule = package / "rogue.py"
    rogue_submodule.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    result = run_bootstrap(
        root=tmp_path,
        files=[inventory_row(tmp_path, package_init)],
        source=b"import bootstrap_guard_package.rogue\n",
        script=tmp_path / "sealed_evaluator.py",
    )
    assert result.returncode != 0
    assert "Blocked unresolved frozen repository import" in result.stderr
    assert not marker.exists()

    pyc_marker = tmp_path / "sourceless-side-effect-observed"
    source_path = tmp_path / "sourceless_shadow.py"
    source_path.write_text(
        f"from pathlib import Path\nPath({str(pyc_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    py_compile.compile(
        str(source_path),
        cfile=str(tmp_path / "sourceless_shadow.pyc"),
        doraise=True,
    )
    source_path.unlink()
    result = run_bootstrap(
        root=tmp_path,
        files=[inventory_row(tmp_path, package_init)],
        source=b"import sourceless_shadow\n",
        script=tmp_path / "sealed_evaluator.py",
    )
    assert result.returncode != 0
    assert "Blocked non-frozen repository import before execution" in result.stderr
    assert not pyc_marker.exists()

    editable_marker = tmp_path / "editable-side-effect-observed"
    editable_name = "_ignored_bootstrap_rogue_v1_3_test"
    editable_rogue = REPOSITORY_ROOT / "nano_deepseek_v4" / f"{editable_name}.py"
    editable_rogue.write_text(
        f"from pathlib import Path\nPath({str(editable_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    try:
        tracked_package_paths = subprocess.run(
            ["git", "ls-files", "nano_deepseek_v4/*.py"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        result = run_bootstrap(
            root=REPOSITORY_ROOT,
            files=[
                inventory_row(REPOSITORY_ROOT, REPOSITORY_ROOT / path)
                for path in tracked_package_paths
            ],
            source=f"import nano_deepseek_v4.{editable_name}\n".encode(),
            script=GENERATED_PATHS["evaluator"],
        )
    finally:
        editable_rogue.unlink(missing_ok=True)
    assert result.returncode != 0
    assert "Blocked unresolved frozen repository import" in result.stderr
    assert not editable_marker.exists()

    frozen_name = "_adaptive_v4_deleted_frozen_helper_v13_test"
    frozen_helper = tmp_path / f"{frozen_name}.py"
    frozen_helper.write_text("VALUE = 'sealed-frozen-bytes'\n", encoding="utf-8")
    site_packages = Path(str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"]))
    site_shadow = site_packages / f"{frozen_name}.py"
    site_shadow.write_text("VALUE = 'site-shadow'\n", encoding="utf-8")
    try:
        result = run_bootstrap(
            root=tmp_path,
            files=[inventory_row(tmp_path, frozen_helper)],
            source=(
                f"import os\nos.unlink({str(frozen_helper)!r})\n"
                f"import {frozen_name}\nprint({frozen_name}.VALUE)\n"
            ).encode(),
            script=tmp_path / "sealed_evaluator.py",
        )
    finally:
        site_shadow.unlink(missing_ok=True)
        frozen_helper.unlink(missing_ok=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sealed-frozen-bytes"

    pth_marker = tmp_path / "pth-side-effect-observed"
    explicit_name = "_adaptive_v4_explicit_site_import_v13_test"
    explicit_module = site_packages / f"{explicit_name}.py"
    pth_file = site_packages / "_adaptive_v4_no_site_bootstrap_v13_test.pth"
    explicit_module.write_text("VALUE = 'explicit-site-package'\n", encoding="utf-8")
    pth_file.write_text(
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    try:
        result = run_bootstrap(
            root=tmp_path,
            files=[inventory_row(tmp_path, package_init)],
            source=f"import {explicit_name}\nprint({explicit_name}.VALUE)\n".encode(),
            script=tmp_path / "sealed_evaluator.py",
        )
    finally:
        pth_file.unlink(missing_ok=True)
        explicit_module.unlink(missing_ok=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "explicit-site-package"
    assert not pth_marker.exists()


def test_runtime_import_guard_accepts_only_exact_sealed_site_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_packages = Path(str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"]))
    assert Path(str(typing_extensions.__file__)).resolve().is_relative_to(site_packages)
    monkeypatch.setattr(
        evaluator.contract,
        "v1_3_5_implementation_file_paths",
        lambda: (
            "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard_v1_3.py",
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
        str(site_packages),
        raising=False,
    )

    evaluator._assert_repository_import_origins({"typing_extensions": typing_extensions})

    outside = tmp_path / "outside_site_packages.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    inside_link = site_packages / "_adaptive_v4_v132_symlink_escape.py"
    inside_link.symlink_to(outside)
    try:
        inside_symlink_module = ModuleType("inside_symlink_escape")
        inside_symlink_module.__file__ = str(inside_link)
        with pytest.raises(ValueError, match="escaped its sealed root"):
            evaluator._assert_repository_import_origins(
                {inside_symlink_module.__name__: inside_symlink_module}
            )
    finally:
        inside_link.unlink(missing_ok=True)

    outside_link = tmp_path / "outside_link_into_site_packages.py"
    outside_link.symlink_to(Path(str(typing_extensions.__file__)).resolve())
    outside_symlink_module = ModuleType("outside_symlink_forgery")
    outside_symlink_module.__file__ = str(outside_link)
    with pytest.raises(ValueError, match="escaped its sealed root"):
        evaluator._assert_repository_import_origins(
            {outside_symlink_module.__name__: outside_symlink_module}
        )

    monkeypatch.setattr(
        evaluator,
        "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
        str(REPOSITORY_ROOT.resolve()),
    )
    with pytest.raises(ValueError, match="not the exact verified runtime root"):
        evaluator._assert_repository_import_origins({"typing_extensions": typing_extensions})

    monkeypatch.setattr(
        evaluator,
        "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
        str(site_packages.parent / ".." / site_packages.parent.name / site_packages.name),
    )
    with pytest.raises(ValueError, match="not the exact verified runtime root"):
        evaluator._assert_repository_import_origins({"typing_extensions": typing_extensions})


def test_runtime_import_guard_accepts_originless_virtual_module_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_packages = Path(str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"]))
    monkeypatch.setattr(
        evaluator.contract,
        "v1_3_5_implementation_file_paths",
        lambda: (
            "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard_v1_3.py",
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
        str(site_packages),
        raising=False,
    )

    frozen_path = GENERATED_PATHS["evaluator"].resolve()
    frozen = ModuleType("evaluate_p2_direct_controller_shard_v1_3")
    frozen.__file__ = str(frozen_path)
    frozen_spec = importlib.machinery.ModuleSpec(
        frozen.__name__,
        loader=object(),
        origin=str(frozen_path),
    )
    frozen_spec.has_location = True
    frozen.__spec__ = frozen_spec
    evaluator._assert_repository_import_origins({frozen.__name__: frozen})

    class VirtualModule(ModuleType):
        __file__ = "_virtual_module_compatibility_marker.py"

    virtual = VirtualModule("unrelated.virtual_namespace")
    assert "__file__" not in vars(virtual)
    assert virtual.__spec__ is None
    assert evaluator.torch.ops.__file__ == "_ops.py"
    assert "__file__" not in vars(evaluator.torch.ops)
    assert evaluator.torch.ops.__spec__ is None
    assert evaluator.torch.classes.__file__ == "_classes.py"
    assert "__file__" not in vars(evaluator.torch.classes)
    assert evaluator.torch.classes.__spec__ is None
    typing_io = sys.modules["typing.io"]
    typing_re = sys.modules["typing.re"]

    evaluator._assert_repository_import_origins(
        {
            frozen.__name__: frozen,
            "negative.import.cache": None,
            "typing": typing,
            "typing.io": typing_io,
            "typing.re": typing_re,
            "unrelated.virtual_namespace": virtual,
            "torch.ops": evaluator.torch.ops,
            "torch.classes": evaluator.torch.classes,
        }
    )


def test_runtime_import_guard_rejects_owned_and_disagreeing_origin_claims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    site_packages = Path(str(launcher._python_runtime_binding(REPOSITORY_ROOT)["site_packages"]))
    monkeypatch.setattr(
        evaluator.contract,
        "v1_3_5_implementation_file_paths",
        lambda: (
            "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard_v1_3.py",
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_ADAPTIVE_V4_SEALED_SITE_PACKAGES_V1_3_5",
        str(site_packages),
        raising=False,
    )

    relative_claim = ModuleType("relative_claim")
    relative_claim.__file__ = "_owned_nonexistent_origin.py"
    with pytest.raises(ValueError, match="Imported module origin is not exact"):
        evaluator._assert_repository_import_origins({"relative_claim": relative_claim})

    malformed_file_claim = ModuleType("malformed_file_claim")
    malformed_file_claim.__file__ = Path("_non_string_origin.py")  # type: ignore[assignment]
    with pytest.raises(ValueError, match="own __file__ metadata is malformed"):
        evaluator._assert_repository_import_origins({"malformed_file_claim": malformed_file_claim})

    malformed_spec_claim = ModuleType("malformed_spec_claim")
    malformed_spec_claim.__spec__ = SimpleNamespace(  # type: ignore[assignment]
        has_location=False,
        origin=None,
    )
    with pytest.raises(ValueError, match="own __spec__ metadata is malformed"):
        evaluator._assert_repository_import_origins({"malformed_spec_claim": malformed_spec_claim})

    with pytest.raises(ValueError, match="not a structurally anchored namespace proxy"):
        evaluator._assert_repository_import_origins({"arbitrary.import.registry.entry": object()})

    forged_typing_proxy = type(
        "typing.forged",
        (),
        {"__module__": "typing", "__qualname__": "forged"},
    )
    with pytest.raises(ValueError, match="not a structurally anchored namespace proxy"):
        evaluator._assert_repository_import_origins(
            {"typing": typing, "typing.forged": forged_typing_proxy}
        )

    same_metaclass_forged_proxy = type(sys.modules["typing.io"])(
        "typing.same_metaclass_forged",
        (),
        {
            "__module__": "typing",
            "__qualname__": "same_metaclass_forged",
            "marker": True,
        },
    )
    with pytest.raises(ValueError, match="not a structurally anchored namespace proxy"):
        evaluator._assert_repository_import_origins(
            {
                "typing": typing,
                "typing.same_metaclass_forged": same_metaclass_forged_proxy,
            }
        )

    orphaned_proxy = type(
        "orphaned_namespace",
        (),
        {"__module__": "missing_parent"},
    )
    with pytest.raises(ValueError, match="not a structurally anchored namespace proxy"):
        evaluator._assert_repository_import_origins(
            {"missing_parent.orphaned_namespace": orphaned_proxy}
        )

    rogue = REPOSITORY_ROOT / "nano_deepseek_v4/_ignored_spec_rogue_v1_3_test.py"
    rogue.write_text("ROGUE = True\n", encoding="utf-8")
    try:
        spec_only_claim = ModuleType("nano_deepseek_v4._ignored_spec_rogue_v1_3_test")
        spec = importlib.machinery.ModuleSpec(
            spec_only_claim.__name__,
            loader=object(),
            origin=str(rogue),
        )
        spec.has_location = True
        spec_only_claim.__spec__ = spec
        with pytest.raises(ValueError, match="outside the frozen implementation inventory"):
            evaluator._assert_repository_import_origins({spec_only_claim.__name__: spec_only_claim})
    finally:
        rogue.unlink(missing_ok=True)

    file_origin = tmp_path / "declared-file.py"
    spec_origin = tmp_path / "declared-spec.py"
    file_origin.write_text("VALUE = 'file'\n", encoding="utf-8")
    spec_origin.write_text("VALUE = 'spec'\n", encoding="utf-8")
    disagreeing = ModuleType("disagreeing_provenance")
    disagreeing.__file__ = str(file_origin)
    disagreeing_spec = importlib.machinery.ModuleSpec(
        disagreeing.__name__,
        loader=object(),
        origin=str(spec_origin),
    )
    disagreeing_spec.has_location = True
    disagreeing.__spec__ = disagreeing_spec
    with pytest.raises(ValueError, match="provenance metadata disagrees"):
        evaluator._assert_repository_import_origins({disagreeing.__name__: disagreeing})


def test_summary_cli_defaults_cannot_fall_back_to_the_v1_2_output_tree() -> None:
    source = _source("summary")
    assert "default=integrity_audit.INTEGRITY_OUTPUT" in source
    assert "default=contract.V1_3_5_MATRIX_SUMMARY_PATH" in source
    assert "default=contract.V1_3_5_OUTPUT_ROOT" in source
    assert "confirmatory_comparator_rule" in source
    assert "strongest_fixed_comparator_rule" not in source


def test_entrypoint_clis_forward_explicit_key_path_with_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "external-attestation-key"
    monkeypatch.delenv(matrix.attestation.KEY_PATH_ENV, raising=False)
    matrix_call: dict[str, object] = {}
    audit_call: dict[str, object] = {}

    def fake_run_matrix(**kwargs: object) -> dict[str, object]:
        matrix_call.update(kwargs)
        return {
            "experiment_id": contract.V1_3_5_MATRIX_EXPERIMENT_ID,
            "status": "terminal",
            "integrity_status": "INTEGRITY-PASS",
            "completed_shards": matrix.EXPECTED_SHARDS,
            "canonical_prefix_shards": matrix.EXPECTED_SHARDS,
            "globally_committed_shards": matrix.EXPECTED_SHARDS,
            "payload_sha256": "1" * 64,
        }

    def fake_audit_matrix(**kwargs: object) -> dict[str, object]:
        audit_call.update(kwargs)
        return {
            "experiment_id": audit.EXPERIMENT_ID,
            "status": "terminal",
            "integrity_status": "INTEGRITY-PASS",
            "validated_shards": matrix.EXPECTED_SHARDS,
            "payload_sha256": "2" * 64,
        }

    monkeypatch.setattr(matrix, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix-v1-3",
            "--attestation-key-path",
            str(key_path),
            "--start-mode",
            "resume",
        ],
    )
    assert matrix.main() == 0
    assert matrix_call["attestation_key_path"] == key_path
    capsys.readouterr()

    monkeypatch.setattr(audit, "audit_matrix", fake_audit_matrix)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit-v1-3", "--attestation-key-path", str(key_path)],
    )
    assert audit.main() == 0
    assert audit_call["attestation_key_path"] == key_path
    capsys.readouterr()

    observed_summary: dict[str, object] = {}

    class ExplicitSummaryKeyObserved(RuntimeError):
        pass

    def observe_summary_key(path: Path, **kwargs: object) -> object:
        observed_summary["path"] = path
        observed_summary.update(kwargs)
        raise ExplicitSummaryKeyObserved

    monkeypatch.setattr(summary.attestation, "load_trust_root", observe_summary_key)
    monkeypatch.setattr(
        summary.attestation,
        "trust_root_from_environment",
        lambda **_kwargs: pytest.fail("explicit summary key path fell back to the environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary-v1-3",
            "--attestation-key-path",
            str(key_path),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(ExplicitSummaryKeyObserved):
        summary.main()
    assert observed_summary["path"] == key_path
    assert observed_summary["repository_root"] == summary.REPOSITORY_ROOT


def test_prerequisites_only_cli_has_a_dedicated_nonledger_result_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "key"
    monkeypatch.delenv(matrix.attestation.KEY_PATH_ENV, raising=False)
    monkeypatch.setattr(
        matrix,
        "run_matrix",
        lambda **_kwargs: {
            "experiment_id": contract.V1_3_5_MATRIX_EXPERIMENT_ID,
            "status": "prerequisites_validated",
            "expected_shards": matrix.EXPECTED_SHARDS,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix-v1-3-5",
            "--attestation-key-path",
            str(key_path),
            "--start-mode",
            "prerequisites-only",
        ],
    )
    assert matrix.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "experiment_id": contract.V1_3_5_MATRIX_EXPERIMENT_ID,
        "expected_shards": matrix.EXPECTED_SHARDS,
        "status": "prerequisites_validated",
    }


def test_entrypoint_clis_reject_missing_key_transport_before_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(matrix.attestation.KEY_PATH_ENV, raising=False)
    matrix_root = tmp_path / "matrix-output"
    audit_output = tmp_path / "audit-output.json"
    summary_output = tmp_path / "summary-output.json"

    monkeypatch.setattr(
        matrix,
        "run_matrix",
        lambda **_kwargs: pytest.fail("matrix ran without attestation-key transport"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matrix-v1-3",
            "--output-root",
            str(matrix_root),
            "--start-mode",
            "resume",
        ],
    )
    with pytest.raises(SystemExit) as matrix_exit:
        matrix.main()
    assert matrix_exit.value.code == 2
    assert not matrix_root.exists()

    monkeypatch.setattr(
        audit,
        "audit_matrix",
        lambda **_kwargs: pytest.fail("audit ran without attestation-key transport"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit-v1-3", "--output", str(audit_output)],
    )
    with pytest.raises(SystemExit) as audit_exit:
        audit.main()
    assert audit_exit.value.code == 2
    assert not audit_output.exists()

    monkeypatch.setattr(
        summary,
        "_load_integrity_same_fd",
        lambda *_args, **_kwargs: pytest.fail("summary read inputs without key transport"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary-v1-3", "--output", str(summary_output)],
    )
    with pytest.raises(SystemExit) as summary_exit:
        summary.main()
    assert summary_exit.value.code == 2
    assert not summary_output.exists()


def test_runner_parser_help_and_evaluator_family_flag_are_unambiguous(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["controller-v1-3", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        matrix.main()
    assert exit_info.value.code == 0
    help_output = capsys.readouterr().out
    assert "--max-new-cells" in help_output
    assert "--attestation-key-path" in help_output
    assert inspect.getsource(matrix.build_evaluator_command).count('"--family"') == 1


def test_quality_start_mode_and_stop_limit_truth_table_precedes_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReachedPureLayout(RuntimeError):
        pass

    monkeypatch.setattr(matrix, "_require_launcher_authority", lambda: {})
    monkeypatch.setattr(
        matrix,
        "_validate_matrix_layout",
        lambda **_kwargs: (_ for _ in ()).throw(ReachedPureLayout),
    )
    valid = (("fresh", 1), ("resume", None), ("prerequisites-only", None))
    for start_mode, limit in valid:
        with pytest.raises(ReachedPureLayout):
            matrix.run_matrix(start_mode=start_mode, max_new_cells=limit)

    invalid = (
        ("fresh", None),
        ("fresh", 2),
        ("resume", 1),
        ("prerequisites-only", 1),
        ("unknown", None),
    )
    for start_mode, limit in invalid:
        with pytest.raises(ValueError):
            matrix.run_matrix(start_mode=start_mode, max_new_cells=limit)


def test_v1_3_5_rejects_non_supervisor_or_out_of_range_topology_before_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matrix, "_require_launcher_authority", lambda: {})
    monkeypatch.setattr(matrix, "_validate_matrix_layout", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        matrix,
        "_validate_quality_mutating_lock_path",
        lambda *_args, **_kwargs: pytest.fail("topology rejection reached lock validation"),
    )
    for worker_index, worker_count, coordinator_only in (
        (1, 2, False),
        (0, 2, True),
        (0, 1, True),
        (0, 5, False),
    ):
        with pytest.raises(ValueError, match="one supervisor"):
            matrix.run_matrix(
                start_mode="resume",
                worker_index=worker_index,
                worker_count=worker_count,
                coordinator_only=coordinator_only,
            )


def test_v1_3_5_valid_parallel_topology_dispatches_one_same_gpu_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = SimpleNamespace()
    terminal = {
        "experiment_id": contract.V1_3_5_MATRIX_EXPERIMENT_ID,
        "status": "terminal",
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(matrix, "_require_launcher_authority", lambda: {})
    monkeypatch.setattr(matrix, "_validate_matrix_layout", lambda **_kwargs: layout)
    monkeypatch.setattr(
        matrix, "_validate_quality_mutating_lock_path", lambda *_args, **_kwargs: None
    )

    def run_supervisor(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return terminal

    monkeypatch.setattr(matrix, "_run_same_gpu_supervisor", run_supervisor)

    assert (
        matrix.run_matrix(
            start_mode="resume",
            worker_index=0,
            worker_count=3,
            coordinator_only=False,
        )
        is terminal
    )
    assert observed["layout"] is layout
    assert observed["worker_count"] == 3
    assert observed["start_mode"] == "resume"
    assert observed["max_new_cells"] is None


def test_mixed_device_assignment_is_complete_disjoint_and_factor_balanced() -> None:
    all_coordinates = matrix.coordinates()
    gb10 = matrix._mixed_device_coordinates("gb10")
    rtx4090 = matrix._mixed_device_coordinates("rtx4090")
    assert len(gb10) == 3_600
    assert len(rtx4090) == 5_400
    assert {item.key for item in gb10}.isdisjoint(item.key for item in rtx4090)
    assert {item.key for item in (*gb10, *rtx4090)} == {
        item.key for item in all_coordinates
    }
    assert gb10[0] == all_coordinates[0]
    assert rtx4090[0] == all_coordinates[4]

    strata: dict[tuple[object, ...], Counter[str]] = {}
    for site, items in (("gb10", gb10), ("rtx4090", rtx4090)):
        for item in items:
            key = (
                item.scale,
                item.training_seed,
                item.budget,
                item.family,
                item.context,
            )
            strata.setdefault(key, Counter())[site] += 1
    assert len(strata) == 900
    assert all(counts == {"gb10": 4, "rtx4090": 6} for counts in strata.values())

    factors = (
        ("scale", matrix.FROZEN_SCALES),
        ("training_seed", matrix.FROZEN_TRAINING_SEEDS),
        ("budget", matrix.FROZEN_BUDGETS),
        ("family", matrix.FROZEN_FAMILIES),
        ("context", matrix.FROZEN_CONTEXTS),
    )
    for field, levels in factors:
        for level in levels:
            counts = Counter(
                item.replicate for item in gb10 if getattr(item, field) == level
            )
            assert set(counts) == set(matrix.FROZEN_REPLICATES)
            assert len(set(counts.values())) == 1

    workers = [
        matrix._assigned_coordinates(worker_index=index, worker_count=3)
        for index in range(3)
    ]
    site_count = contract.V1_3_5_MIXED_SITE_COORDINATE_COUNTS[
        contract.V1_3_5_MIXED_DEVICE_SITE
    ]
    assert [len(items) for items in workers] == [site_count // 3] * 3
    assert len({item.key for items in workers for item in items}) == site_count
    assert [
        tuple(item.payload for item in items) for items in workers
    ] == [
        matrix.persistent_session.assigned_coordinates(
            worker_index=index, worker_count=3
        )
        for index in range(3)
    ]


def test_prerequisites_only_does_not_acquire_gpu_activate_or_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = SimpleNamespace(
        output_root=tmp_path / "output",
        matrix_summary=tmp_path / "output/matrix.json",
        training_output_root=tmp_path / "training",
        calibration_output_root=tmp_path / "calibration",
        reuse_admission_path=tmp_path / "admission/reuse.json",
        preheldout_genesis_path=tmp_path / "admission/genesis.json",
    )
    base = SimpleNamespace(public_binding={"base": "validated"})
    prestart = SimpleNamespace(absence_witness={"absence": "validated"})
    monkeypatch.setattr(matrix, "_require_launcher_authority", lambda: {})
    monkeypatch.setattr(matrix, "_validate_matrix_layout", lambda **_kwargs: layout)
    monkeypatch.setattr(
        matrix, "_validate_quality_mutating_lock_path", lambda *_a, **_k: Path("/tmp/lock")
    )
    monkeypatch.setattr(
        matrix,
        "load_and_validate_prestart_prerequisites",
        lambda **_kwargs: (base, prestart),
    )
    for name in (
        "acquire_gpu_lock",
        "_prepare_quality_start_authority",
        "_atomic_write_json",
        "_capture_selected_device_context",
    ):
        monkeypatch.setattr(
            matrix,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"prerequisites-only reached {_name}"
            ),
        )

    before = tuple(tmp_path.iterdir())
    result = matrix.run_matrix(start_mode="prerequisites-only")
    after = tuple(tmp_path.iterdir())
    assert result["status"] == "prerequisites_validated"
    assert result["prerequisites"] == base.public_binding
    assert before == after == ()


def test_one_cell_full_resume_gate_uses_flat_canonical_record_and_integrity_pass() -> None:
    coordinate = matrix._mixed_device_coordinates(
        contract.V1_3_5_MIXED_DEVICE_SITE
    )[0]
    valid = {
        **coordinate.payload,
        "coordinate_key": coordinate.key,
        "integrity_decision": "INTEGRITY-PASS",
    }
    matrix._assert_one_cell_full_resume_gate([valid])

    failed = {**valid, "integrity_decision": "INTEGRITY-FAIL"}
    with pytest.raises(ValueError, match="canonical integrity gate"):
        matrix._assert_one_cell_full_resume_gate([failed])
    with pytest.raises(ValueError, match="canonical integrity gate"):
        matrix._assert_one_cell_full_resume_gate([])
    with pytest.raises(ValueError, match="canonical integrity gate"):
        matrix._assert_one_cell_full_resume_gate([valid, valid])
    nested = {
        "coordinate": coordinate.payload,
        "coordinate_key": coordinate.key,
        "integrity_decision": "INTEGRITY-PASS",
    }
    with pytest.raises(ValueError, match="canonical integrity gate"):
        matrix._assert_one_cell_full_resume_gate([nested])


def test_activation_ledger_boundary_uses_held_lease_and_recovers_interrupted_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    activation_binding = {"activation": "binding"}
    lock_binding = {"path": str(tmp_path / "activation/matrix.lock")}
    activation = SimpleNamespace(
        public_binding=activation_binding,
        matrix_lock_binding=lock_binding,
    )

    class Lease:
        def __init__(self) -> None:
            self.activation = activation
            self.descriptor = 73
            self.assertions = 0

        def assert_held(self) -> None:
            self.assertions += 1

        def fileno(self) -> int:
            return self.descriptor

    prerequisites = SimpleNamespace(
        public_binding={"quality_start_activation": activation_binding},
        activation=activation,
        trust_root=object(),
    )

    def layout_at(name: str) -> SimpleNamespace:
        output = tmp_path / name
        return SimpleNamespace(
            output_root=output,
            matrix_summary=output / matrix.MATRIX_SUMMARY_NAME,
            lock_path=Path(lock_binding["path"]),
        )

    payload = {
        "completed_shards": 0,
        "prerequisites": prerequisites.public_binding,
        "matrix_lock": lock_binding,
        "ready_only_preflight": {"preflight": "binding"},
        "worker_count": 1,
    }
    monkeypatch.setattr(matrix, "_matrix_payload", lambda *_args, **_kwargs: dict(payload))
    monkeypatch.setattr(matrix, "_verify_attested_payload", lambda *_args, **_kwargs: None)

    fresh_layout = layout_at("fresh")
    lease = Lease()
    descriptor_before = lease.fileno()
    observed = matrix._complete_activation_ledger_boundary(
        start_mode="fresh",
        layout=fresh_layout,
        prerequisites=prerequisites,
        activation_lease=lease,
        evaluator_binding={},
        ready_only_preflight=payload["ready_only_preflight"],
        worker_count=1,
        gpu_lease_binding={"gpu": "lease"},
    )
    assert observed == payload
    assert lease.fileno() == descriptor_before
    assert lease.assertions >= 2
    assert json.loads(fresh_layout.matrix_summary.read_text(encoding="utf-8")) == payload

    resumed = matrix._complete_activation_ledger_boundary(
        start_mode="resume",
        layout=fresh_layout,
        prerequisites=prerequisites,
        activation_lease=lease,
        evaluator_binding={},
        ready_only_preflight=payload["ready_only_preflight"],
        worker_count=1,
        gpu_lease_binding={"gpu": "lease"},
    )
    assert resumed == payload

    interrupted_layout = layout_at("interrupted")
    interrupted_layout.output_root.mkdir(mode=0o700)
    temporary = interrupted_layout.output_root / (
        f".{interrupted_layout.matrix_summary.name}.deadbeef.tmp"
    )
    temporary.write_bytes(b"partial")
    temporary.chmod(0o600)
    recovered = matrix._complete_activation_ledger_boundary(
        start_mode="resume",
        layout=interrupted_layout,
        prerequisites=prerequisites,
        activation_lease=lease,
        evaluator_binding={},
        ready_only_preflight=payload["ready_only_preflight"],
        worker_count=1,
        gpu_lease_binding={"gpu": "lease"},
    )
    assert recovered == payload
    assert interrupted_layout.matrix_summary.is_file()
    assert not temporary.exists()


@pytest.mark.parametrize("start_mode", ("fresh", "resume"))
def test_quality_start_authority_closes_untransferred_lease_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start_mode: str,
) -> None:
    layout = SimpleNamespace(
        training_output_root=tmp_path / "training",
        calibration_output_root=tmp_path / "calibration",
        reuse_admission_path=tmp_path / "admission/reuse.json",
        preheldout_genesis_path=tmp_path / "admission/genesis.json",
        output_root=tmp_path / "output",
    )
    state = SimpleNamespace(active=False, created=[], fail_post_acquisition=True)

    class Lease:
        def __init__(self, activation: object) -> None:
            assert not state.active
            state.active = True
            state.created.append(self)
            self.activation = activation
            self.close_count = 0

        def assert_held(self) -> None:
            assert state.active

        def close(self) -> None:
            self.close_count += 1
            assert self.close_count == 1
            state.active = False

    scheduler = SimpleNamespace(assert_held=lambda: None)
    base = SimpleNamespace(trust_root=object(), public_binding={"base": "binding"})
    activation = SimpleNamespace(public_binding={"activation": "binding"})
    frozen = SimpleNamespace(
        trust_root=base.trust_root,
        activation=activation,
        public_binding={
            "frozen": "binding",
            "quality_start_activation": activation.public_binding,
        },
        base_public_binding=base.public_binding,
    )
    monkeypatch.setattr(matrix, "_validated_source_provenance", lambda: {"source": "sealed"})
    monkeypatch.setattr(
        matrix,
        "_validated_launch_routing",
        lambda **_kwargs: {"routing": "sealed"},
    )
    monkeypatch.setattr(
        matrix,
        "load_and_validate_prestart_prerequisites",
        lambda **_kwargs: (base, SimpleNamespace()),
    )

    if start_mode == "fresh":
        monkeypatch.setattr(
            matrix.admission,
            "publish_quality_start_activation",
            lambda **_kwargs: Lease(activation),
        )

        def promote(*_args: object, **_kwargs: object) -> object:
            if state.fail_post_acquisition:
                state.fail_post_acquisition = False
                raise RuntimeError("post-acquisition validation failed")
            return frozen

        monkeypatch.setattr(matrix, "_promote_activated_prerequisites", promote)
    else:
        monkeypatch.setattr(
            matrix.admission,
            "acquire_quality_start_activation_lease",
            lambda *_args, **_kwargs: Lease(activation),
        )
        monkeypatch.setattr(
            matrix,
            "load_and_validate_prerequisites",
            lambda **_kwargs: frozen,
        )

        def revalidate(*_args: object, **_kwargs: object) -> object:
            if state.fail_post_acquisition:
                state.fail_post_acquisition = False
                raise RuntimeError("post-acquisition validation failed")
            return activation

        monkeypatch.setattr(
            matrix.admission,
            "revalidate_activated_quality_authority",
            revalidate,
        )

    arguments = {
        "start_mode": start_mode,
        "layout": layout,
        "manifest_path": tmp_path / "manifest.json",
        "attestation_key_path": None,
        "worker_count": 1,
        "scheduler_lease": scheduler,
    }
    with pytest.raises(RuntimeError, match="post-acquisition validation failed"):
        matrix._prepare_quality_start_authority(**arguments)
    assert len(state.created) == 1
    assert state.created[0].close_count == 1
    assert not state.active

    prerequisites, retry_lease = matrix._prepare_quality_start_authority(**arguments)
    assert prerequisites is frozen
    assert retry_lease is state.created[1]
    assert state.active
    retry_lease.close()
    assert not state.active


def test_runner_rejects_nested_coordinate_scoped_activation_promotion() -> None:
    public_binding = {"base": "binding"}
    base = matrix.ValidatedBasePrerequisites(
        context=SimpleNamespace(),
        trust_root=SimpleNamespace(),
        bundles={},
        public_binding=public_binding,
    )
    scoped = matrix.admission.ValidatedQualityStartActivationV1_3_5(
        _seal=object(),
        payload={"base_prerequisites_binding": public_binding},
        public_binding={"activation": "binding"},
        quality_context=SimpleNamespace(),
        reuse_admission=SimpleNamespace(),
        preheldout_genesis=SimpleNamespace(),
        consumer_coordinate=(contract.SCALES[0], contract.TRAINING_SEEDS[0]),
        root_identity={},
        matrix_lock_binding={},
    )
    with pytest.raises(ValueError, match="full-owner prerequisites"):
        matrix._promote_activated_prerequisites(base, scoped)


def test_selected_device_guard_closes_on_post_acquisition_failure_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = SimpleNamespace(active=False, guards=[])

    class Scheduler:
        path = tmp_path / "scheduler.lock"

        def __init__(self) -> None:
            self.assertions = 0

        def assert_held(self) -> None:
            self.assertions += 1
            if self.assertions == 2:
                raise RuntimeError("scheduler changed after guard acquisition")

    class Guard:
        def __init__(self) -> None:
            assert not state.active
            state.active = True
            state.guards.append(self)
            self.close_count = 0

        def assert_held(self) -> None:
            assert state.active

        def close(self) -> None:
            self.close_count += 1
            assert self.close_count == 1
            state.active = False

    scheduler = Scheduler()
    guard_path = tmp_path / "device.lock"
    monkeypatch.setattr(matrix, "canonical_device_guard_path", lambda _identity: guard_path)
    monkeypatch.setattr(matrix, "acquire_device_guard", lambda *_args, **_kwargs: Guard())
    context = {"selected_device_routing_identity": {"uuid": "GPU-test"}}

    with pytest.raises(RuntimeError, match="scheduler changed"):
        matrix._acquire_selected_device_guard(
            label="test", device_context=context, scheduler_lease=scheduler
        )
    assert state.guards[0].close_count == 1
    assert not state.active

    guard = matrix._acquire_selected_device_guard(
        label="test-retry", device_context=context, scheduler_lease=scheduler
    )
    assert guard is state.guards[1]
    guard.close()
    assert not state.active


def test_mutating_lock_paths_reject_authority_key_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    layout = SimpleNamespace(
        output_root=(tmp_path / "output").resolve(),
        matrix_summary=(tmp_path / "output/matrix.json").resolve(),
        activation_root=(tmp_path / "activation").resolve(),
        activation_path=(tmp_path / "activation/receipt.json").resolve(),
        lock_path=(tmp_path / "activation/matrix.lock").resolve(),
        training_output_root=(tmp_path / "training").resolve(),
        calibration_output_root=(tmp_path / "calibration").resolve(),
        reuse_admission_path=(tmp_path / "admission/reuse.json").resolve(),
        preheldout_genesis_path=(tmp_path / "admission/genesis.json").resolve(),
    )
    manifest_path = (REPOSITORY_ROOT / "pyproject.toml").resolve()
    canonical_scheduler = contract.DIRECT_GPU_SCHEDULER_LOCK_PATH.resolve()
    with pytest.raises(ValueError, match="overlaps"):
        matrix._validate_quality_mutating_lock_path(
            canonical_scheduler,
            layout=layout,
            manifest_path=manifest_path,
            attestation_key_path=canonical_scheduler,
            label="GPU scheduler lock",
            require_canonical_scheduler=True,
        )
    with pytest.raises(ValueError, match="overlaps"):
        matrix._validate_quality_mutating_lock_path(
            matrix.admission.V1_3_5_ACTIVATION_BOOTSTRAP_LOCK_PATH,
            layout=layout,
            manifest_path=manifest_path,
            attestation_key_path=None,
            label="GPU physical-device guard",
            require_canonical_scheduler=False,
        )
    with pytest.raises(ValueError, match="canonical GPU scheduler"):
        matrix._validate_quality_mutating_lock_path(
            manifest_path,
            layout=layout,
            manifest_path=manifest_path,
            attestation_key_path=None,
            label="GPU scheduler lock",
            require_canonical_scheduler=True,
        )

    key_path = tmp_path / "key"
    guard_path = tmp_path / "guard"
    key_path.write_bytes(b"secret")
    key_path.chmod(0o600)
    os.link(key_path, guard_path)
    with pytest.raises(ValueError, match="unsafe pre-existing metadata|aliases"):
        matrix._validate_quality_mutating_lock_path(
            guard_path,
            layout=layout,
            manifest_path=manifest_path,
            attestation_key_path=key_path,
            label="GPU physical-device guard",
            require_canonical_scheduler=False,
        )


def test_source_provenance_distinguishes_pinned_head_from_frozen_source_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "1" * 40
    pinned_head = "2" * 40
    tree_digest = "3" * 64
    manifest_source = json.dumps(
        {
            "implementation": {
                "source_commit": source_commit,
                "tree_digest": tree_digest,
            }
        },
        sort_keys=True,
    ).encode()
    provenance = {
        "schema_version": 1,
        "launcher": "p2-direct-controller-git-object-launcher-v1-3-5",
        "repository_root": str(REPOSITORY_ROOT.resolve()),
        "bundle_sha256": "4" * 64,
        "pinned_head_oid": pinned_head,
        "frozen_source_commit": source_commit,
        "implementation_tree_digest": tree_digest,
        "head_manifest": {
            "path": ".bootstrap-head-manifest.json",
            "git_mode": "100644",
            "git_blob_oid": "5" * 40,
            "sha256": hashlib.sha256(manifest_source).hexdigest(),
            "bytes": len(manifest_source),
            "source_base64": base64.b64encode(manifest_source).decode("ascii"),
        },
    }
    monkeypatch.setattr(matrix, "SEALED_SOURCE_PROVENANCE_V1_3_5", provenance)
    assert matrix._validated_source_provenance() == provenance
    assert source_commit != pinned_head

    mismatched = dict(provenance)
    mismatched["frozen_source_commit"] = "6" * 40
    monkeypatch.setattr(matrix, "SEALED_SOURCE_PROVENANCE_V1_3_5", mismatched)
    with pytest.raises(ValueError, match="decoded HEAD manifest implementation"):
        matrix._validated_source_provenance()


def test_audit_and_summary_routes_are_active_only_for_their_own_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, runtime, source_provenance = _sealed_common_provenance()
    audit_route = _sealed_route("audit")
    summary_route = _sealed_route("summary")
    alternate_audit_route = _sealed_route("audit", sha_digit="8")
    for module in (audit, summary):
        monkeypatch.setattr(module, "SEALED_LAUNCH_AUTHORITY_V1_3_5", authority)
        monkeypatch.setattr(module, "SEALED_PYTHON_RUNTIME_V1_3_5", runtime)
        monkeypatch.setattr(module, "SEALED_SOURCE_PROVENANCE_V1_3_5", source_provenance)
    monkeypatch.setattr(matrix, "SEALED_LAUNCH_AUTHORITY_V1_3_5", None)
    monkeypatch.setattr(matrix, "SEALED_PYTHON_RUNTIME_V1_3_5", None)
    monkeypatch.setattr(matrix, "SEALED_SOURCE_PROVENANCE_V1_3_5", None)
    monkeypatch.setattr(matrix, "SEALED_LAUNCH_ROUTING_V1_3_5", None)

    monkeypatch.setattr(audit, "SEALED_LAUNCH_ROUTING_V1_3_5", audit_route)
    assert audit._active_audit_launch_routing() == audit_route
    assert (
        audit._validate_audit_launch_routing_snapshot(
            audit_route, source_provenance=source_provenance
        )
        == audit_route
    )
    with pytest.raises(ValueError, match="differs from active sealed audit routing"):
        audit._validate_audit_launch_routing_snapshot(
            alternate_audit_route, source_provenance=source_provenance
        )

    monkeypatch.setattr(audit, "SEALED_LAUNCH_ROUTING_V1_3_5", None)
    assert (
        audit._validate_audit_launch_routing_snapshot(
            alternate_audit_route, source_provenance=source_provenance
        )
        == alternate_audit_route
    )
    with pytest.raises(ValueError, match="launch-routing snapshot drifted"):
        audit._validate_audit_launch_routing_snapshot(
            summary_route, source_provenance=source_provenance
        )

    monkeypatch.setattr(summary, "SEALED_LAUNCH_ROUTING_V1_3_5", summary_route)
    assert summary._active_summary_launch_routing() == summary_route
    assert audit.SEALED_LAUNCH_ROUTING_V1_3_5 is None
    assert matrix.SEALED_LAUNCH_ROUTING_V1_3_5 is None


def test_matrix_session_crosscheck_binds_actual_argv_and_recovered_eof() -> None:
    argv = ["/sealed/python", "-I", "--persistent-session"]
    plan = {
        "session_nonce": "1" * 64,
        "launch_authority_nonce": "2" * 64,
        "payload_sha256": "3" * 64,
        "worker_index": 0,
        "worker_count": 1,
        "scale": contract.SCALES[0],
        "training_seed": contract.TRAINING_SEEDS[0],
        "coordinate_count": 2,
        "coordinate_digest": "4" * 64,
    }
    base_row = {
        "session_nonce": plan["session_nonce"],
        "launch_authority_nonce": plan["launch_authority_nonce"],
        "plan_payload_sha256": plan["payload_sha256"],
        "worker_index": 0,
        "worker_count": 1,
        "scale": plan["scale"],
        "training_seed": plan["training_seed"],
        "coordinate_count": 2,
        "coordinate_digest": plan["coordinate_digest"],
        "actual_session_argv": argv,
    }
    result_record = {
        "persistent_session_execution": {
            "plan": plan,
            "session_command": argv,
            "work_order": {"sequence_index": 0, "payload_sha256": "5" * 64},
            "work_result": {"payload_sha256": "6" * 64},
            "published_bundle_reingested_after_child_eof": False,
        }
    }
    eof_record = {
        "persistent_session_execution": {
            "plan": plan,
            "session_command": argv,
            "work_order": {"sequence_index": 1, "payload_sha256": "7" * 64},
            "work_result": None,
            "published_bundle_reingested_after_child_eof": True,
        }
    }
    projection = {
        "sessions": [
            {
                **base_row,
                "status": "parent_crash_recovered",
                "completed_work_payload_sha256": ["5" * 64],
                "completed_result_payload_sha256": ["6" * 64],
                "published_bundle_reingestion_count": 1,
            }
        ]
    }
    matrix._crosscheck_matrix_records_with_session_ledger(
        [result_record, eof_record],
        projection,
        require_terminal_sessions=True,
    )

    tampered = json.loads(json.dumps(result_record))
    tampered["persistent_session_execution"]["session_command"].append("--different")
    with pytest.raises(ValueError, match="argv"):
        matrix._crosscheck_matrix_records_with_session_ledger(
            [tampered, eof_record],
            projection,
            require_terminal_sessions=True,
        )


def test_persistent_scope_cleanup_is_total_and_preserves_original_error() -> None:
    events: list[str] = []

    class FakeClaim:
        def __exit__(self, *_args: object) -> None:
            events.append("claim-exit")

    class FakeEvaluator:
        def __init__(
            self,
            lease: object,
            *,
            fail_close: bool = False,
            fail_abort: bool = False,
        ) -> None:
            self.gpu_lease = lease
            self.closed = False
            self.terminal_published = False
            self.verified_eof_reingested = False
            self.fail_close = fail_close
            self.fail_abort = fail_abort

        def close(self) -> None:
            events.append("close")
            if self.fail_close:
                raise RuntimeError("close-failed")
            self.closed = True
            self.terminal_published = True

        def finalize_eof(self, *, published_bundle_reingested: bool) -> None:
            assert published_bundle_reingested is False
            events.append("finalize-eof")
            self.terminal_published = True

        def abort_after_parent_commit_failure(self) -> None:
            events.append("abort")
            if self.fail_abort:
                raise RuntimeError("abort-failed")
            self.closed = True
            self.terminal_published = True

        def terminate_without_terminal(self) -> None:
            events.append("terminate-without-terminal")
            self.closed = True

    def install(
        lease: object,
        active: FakeEvaluator,
        claim: FakeClaim | None = None,
        *,
        refresher: object | None = None,
    ) -> None:
        key = id(lease)

        def default_refresher(candidate: object | None) -> None:
            events.append("reconcile" if candidate is not None else "refresh")

        with matrix._ACTIVE_PERSISTENT_EVALUATORS_LOCK:
            matrix._ACTIVE_PERSISTENT_EVALUATORS[key] = active
            if claim is not None:
                matrix._ACTIVE_PERSISTENT_PREPARATION_CLAIMS[key] = claim
            matrix._ACTIVE_PERSISTENT_PROJECTION_REFRESHERS[key] = (
                default_refresher if refresher is None else refresher
            )

    try:
        pause_lease = object()
        pause_active = FakeEvaluator(pause_lease)
        install(pause_lease, pause_active, FakeClaim())
        pause = matrix.InfrastructurePauseSignal(
            {"required_headroom_bytes": 2, "filesystem_available_bytes": 1}
        )
        assert (
            matrix._settle_active_persistent_evaluator_scope(
                pause_lease, type(pause), pause, pause.__traceback__
            )
            is False
        )
        assert events == ["claim-exit", "reconcile", "close", "refresh"]

        events.clear()
        error_lease = object()
        error_active = FakeEvaluator(error_lease)
        install(error_lease, error_active)
        original = RuntimeError("preflight-failed")
        matrix._settle_active_persistent_evaluator_scope(
            error_lease, type(original), original, original.__traceback__
        )
        assert events == ["reconcile", "abort", "refresh"]

        events.clear()
        cleanup_lease = object()
        cleanup_active = FakeEvaluator(cleanup_lease, fail_close=True, fail_abort=True)
        install(cleanup_lease, cleanup_active)
        cleanup_pause = matrix.InfrastructurePauseSignal(
            {"required_headroom_bytes": 3, "filesystem_available_bytes": 1}
        )
        matrix._settle_active_persistent_evaluator_scope(
            cleanup_lease,
            type(cleanup_pause),
            cleanup_pause,
            cleanup_pause.__traceback__,
        )
        notes = getattr(cleanup_pause, "__notes__", [])
        assert len(notes) == 2
        assert "close-failed" in notes[0]
        assert "abort-failed" in notes[1]

        events.clear()
        stale_lease = object()
        stale_active = FakeEvaluator(stale_lease)

        def failed_reconciliation(candidate: object | None) -> None:
            events.append("reconcile" if candidate is not None else "refresh")
            if candidate is not None:
                raise RuntimeError("signed-record-reconciliation-failed")

        install(stale_lease, stale_active, refresher=failed_reconciliation)
        stale_error = RuntimeError("outer-failure")
        matrix._settle_active_persistent_evaluator_scope(
            stale_lease,
            type(stale_error),
            stale_error,
            stale_error.__traceback__,
        )
        assert events == ["reconcile", "terminate-without-terminal", "refresh"]
        assert stale_active.terminal_published is False
        assert "signed-record-reconciliation-failed" in getattr(stale_error, "__notes__", [])[0]
    finally:
        with matrix._ACTIVE_PERSISTENT_EVALUATORS_LOCK:
            matrix._ACTIVE_PERSISTENT_EVALUATORS.clear()
            matrix._ACTIVE_PERSISTENT_PREPARATION_CLAIMS.clear()
            matrix._ACTIVE_PERSISTENT_PROJECTION_REFRESHERS.clear()


def test_scope_reconciliation_promotes_sparse_commit_and_exact_eof() -> None:
    work = {"sequence_index": 0, "payload_sha256": "1" * 64}
    result = {"payload_sha256": "2" * 64}

    class FakeActive:
        plan = {"session_nonce": "3" * 64}
        work_orders = [work]
        results = [result]
        committed_count = 0
        closed = False
        terminal_published = False
        eof_work_order: dict[str, object] | None = None
        verified_eof_reingested = False

        def mark_committed(self, *, work_order: object, work_result: object) -> None:
            assert work_order == work
            assert work_result == result
            self.committed_count += 1

    active = FakeActive()
    committed_record = {
        "persistent_session_execution": {
            "plan": active.plan,
            "work_order": work,
            "work_result": result,
            "published_bundle_reingested_after_child_eof": False,
        }
    }
    matrix._reconcile_active_persistent_session_records(active, [committed_record])
    assert active.committed_count == 1

    eof_work = {"sequence_index": 1, "payload_sha256": "4" * 64}
    active.closed = True
    active.eof_work_order = eof_work
    eof_record = {
        "persistent_session_execution": {
            "plan": active.plan,
            "work_order": eof_work,
            "work_result": None,
            "published_bundle_reingested_after_child_eof": True,
        }
    }
    matrix._reconcile_active_persistent_session_records(active, [committed_record, eof_record])
    assert active.verified_eof_reingested is True


def test_persistent_child_ready_and_reap_paths_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matrix, "PERSISTENT_READY_RECEIPT_TIMEOUT_SECONDS", 0.15)
    monkeypatch.setattr(matrix, "PERSISTENT_CHILD_EXIT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(matrix, "PERSISTENT_CHILD_KILL_TIMEOUT_SECONDS", 0.2)

    silent = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        with pytest.raises(ValueError, match="ready receipt timed out"):
            matrix._read_persistent_ready_line_bounded(silent)
        assert type(matrix._reap_launch_failure_child_bounded(silent)) is int
    finally:
        if silent.poll() is None:
            silent.kill()
            silent.wait(timeout=1)
        matrix._close_persistent_process_streams(silent)

    ignores_term = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, lambda *_: None); "
                "print('ready', flush=True); time.sleep(60)"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert matrix._read_persistent_ready_line_bounded(ignores_term) == b"ready\n"
        assert type(matrix._reap_launch_failure_child_bounded(ignores_term)) is int
    finally:
        if ignores_term.poll() is None:
            ignores_term.kill()
            ignores_term.wait(timeout=1)
        matrix._close_persistent_process_streams(ignores_term)

    exits_before_ready = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert matrix._read_persistent_ready_line_bounded(exits_before_ready) == b""
        assert matrix._reap_launch_failure_child_bounded(exits_before_ready) == 7
    finally:
        matrix._close_persistent_process_streams(exits_before_ready)


def test_terminal_authority_failure_prevents_irreversible_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = object.__new__(matrix.PersistentEvaluatorProcess)
    active.terminal_published = False
    publish_calls: list[object] = []

    def reject_authority() -> None:
        raise RuntimeError("terminal-authority-drifted")

    active._assert_terminal_authority = reject_authority
    monkeypatch.setattr(
        matrix.persistent_session,
        "publish_session_terminal",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )
    with pytest.raises(RuntimeError, match="terminal-authority-drifted"):
        active._publish_terminal(
            status="parent_commit_failure",
            final_receipt=None,
            returncode=-9,
        )
    assert publish_calls == []
    assert active.terminal_published is False

    publish_source = inspect.getsource(matrix.PersistentEvaluatorProcess._publish_terminal)
    finish_source = inspect.getsource(matrix.PersistentEvaluatorProcess.finish_eof)
    start_source = inspect.getsource(matrix._start_persistent_evaluator)
    assert publish_source.index("_assert_terminal_authority()") < publish_source.index(
        "persistent_session.publish_session_terminal("
    )
    assert "self._cleanup()" not in finish_source
    assert (
        start_source.index("type(returncode) is int")
        < start_source.index("_assert_terminal_launch_authority(")
        < start_source.index("persistent_session.publish_session_terminal(")
    )


@pytest.mark.parametrize("trailing", ["{}\n", "X"])
def test_persistent_close_rejects_output_after_the_final_receipt(trailing: str) -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.stdin.buffer.read(); "
                "sys.stdout.write('{}\\n' + sys.argv[1]); sys.stdout.flush()"
            ),
            trailing,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    active = object.__new__(matrix.PersistentEvaluatorProcess)
    active.process = process
    active.closed = False
    active.terminal_published = False
    active.committed_count = 0
    active.work_orders = []
    active.results = []
    active.eof_returncode = None
    active._assert_authority = lambda: None
    try:
        with pytest.raises(
            ValueError,
            match="trailing output after its final receipt|JSONL message framing is invalid",
        ):
            active.close()
        assert process.returncode == 0
        assert active.terminal_published is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        matrix._close_persistent_process_streams(process)


def test_persistent_scope_generation_fixes_claim_and_launch_failure_windows() -> None:
    distributed = inspect.getsource(matrix._run_distributed_matrix)
    single = inspect.getsource(matrix.run_matrix)
    start = inspect.getsource(matrix._start_persistent_evaluator)

    assert distributed.index("_register_persistent_preparation_claim(") < distributed.index(
        "try:\n            _clear_persistent_preparation_claim("
    )
    assert distributed.index("try:\n            _clear_persistent_preparation_claim(") < (
        distributed.index("Distributed claim preparation did not produce a launch")
    )
    assert single.index("_register_persistent_preparation_claim(") < single.index(
        "try:\n                _clear_persistent_preparation_claim("
    )
    assert single.index("try:\n                _clear_persistent_preparation_claim(") < (
        single.index("Single-worker cell claim binding is missing")
    )
    assert start.index("publish_session_launch(") < start.index("try:")
    assert start.index("try:") < start.index("_open_evaluator(")
    assert start.index("_open_evaluator(") < start.index('status="launch_failure"')


def test_ready_only_preflight_is_ordered_before_zero_matrix_activation_release_and_claim() -> None:
    single = inspect.getsource(matrix.run_matrix)
    ensure = inspect.getsource(matrix._ensure_ready_only_preflight)
    projection = inspect.getsource(matrix._persistent_plan_projection)

    assert (
        single.index("ready_only_preflight = _ensure_ready_only_preflight(")
        < single.index("activation_boundary = _complete_activation_ledger_boundary(")
        < single.index("activation_lease.close()")
        < single.index("_exclusive_cell_claim(")
    )
    assert ensure.index("_start_persistent_evaluator(") < ensure.index(
        "final_receipt = evaluator.close()"
    )
    assert "evaluator.execute(" not in ensure
    assert "build_work_order(" not in ensure
    assert "ready_only_preflight" in matrix._MATRIX_FIELDS
    assert '"session_role": plan["session_role"]' in projection


def test_distributed_zero_ledger_uses_live_supervisor_gpu_authority_for_preflight() -> None:
    validator = inspect.getsource(matrix.validate_matrix_summary)
    disk_validator = inspect.getsource(matrix._validate_distributed_disk_summary)
    distributed_runner = inspect.getsource(matrix._run_distributed_matrix)

    assert "ready_only_preflight_gpu_binding = observed_gpu_bindings.get(0)" in validator
    assert "ready_only_preflight_gpu_lease_binding is not None" in validator
    assert "gpu_lease_binding=ready_only_preflight_gpu_binding" in validator
    assert "ready_only_preflight_gpu_lease_binding=(" in disk_validator
    assert distributed_runner.count(
        "ready_only_preflight_gpu_lease_binding=current_gpu_binding"
    ) >= 4


class _HeldPreflightLease:
    def __init__(self) -> None:
        self.assertions = 0

    def assert_held(self) -> None:
        self.assertions += 1


def test_ready_only_preflight_runs_zero_work_once_and_leaves_quality_tree_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = matrix._mixed_device_coordinates(
        contract.V1_3_5_MIXED_DEVICE_SITE
    )[0]
    output_root = (tmp_path / "quality").resolve()
    layout = SimpleNamespace(
        output_root=output_root,
        matrix_summary=output_root / matrix.MATRIX_SUMMARY_NAME,
    )
    binding = {"schema_version": 1, "session_role": "ready_only_preflight"}
    projections = [
        {"sessions": []},
        {
            "sessions": [
                {
                    "session_role": matrix.persistent_session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
                    "status": "stopped",
                }
            ]
        },
    ]
    events: list[str] = []

    def load_projection(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("load-projection")
        return projections.pop(0)

    def project_ready(value: dict[str, object]) -> dict[str, object] | None:
        return binding if value["sessions"] else None

    ready = {"status": "stopped", "completed_coordinates": 0}
    closed_stream = SimpleNamespace(closed=True)

    class FakeEvaluator:
        ready_receipt = ready
        closed = False
        terminal_published = False
        eof_returncode = None
        committed_count = 0
        work_orders: list[object] = []
        results: list[object] = []
        reingestion_count = 0
        canonical_descriptor = 12
        pycache_manager: object | None = object()
        process = SimpleNamespace(
            stdin=closed_stream,
            stdout=closed_stream,
            stderr=closed_stream,
        )

        def close(self) -> dict[str, object]:
            events.append("close-zero-work")
            self.closed = True
            self.terminal_published = True
            self.eof_returncode = 0
            self.canonical_descriptor = -1
            self.pycache_manager = None
            return ready

    def start_evaluator(**kwargs: object) -> FakeEvaluator:
        events.append("start-ready-only")
        assert kwargs["session_role"] == (
            matrix.persistent_session.READY_ONLY_PREFLIGHT_SESSION_ROLE
        )
        assert kwargs["plan_coordinates"] == (first,)
        assert kwargs["max_new_cells_stop_limit"] == 1
        assert not layout.matrix_summary.exists()
        return FakeEvaluator()

    def validate_binding(value: object, **_kwargs: object) -> dict[str, object]:
        events.append("validate-binding")
        assert value == binding
        return binding

    monkeypatch.setattr(
        matrix.persistent_session, "load_session_ledger_projection", load_projection
    )
    monkeypatch.setattr(matrix.persistent_session, "ready_only_preflight_binding", project_ready)
    monkeypatch.setattr(matrix, "_start_persistent_evaluator", start_evaluator)
    monkeypatch.setattr(matrix, "_validate_ready_only_preflight_snapshot", validate_binding)
    activation = _HeldPreflightLease()
    gpu = _HeldPreflightLease()
    device = _HeldPreflightLease()
    prerequisites = SimpleNamespace(
        trust_root=object(),
        public_binding={},
        bundles={
            (first.scale, first.training_seed, first.budget): SimpleNamespace(
                binding={"input_binding_digest": "a" * 64}
            )
        },
    )

    observed = matrix._ensure_ready_only_preflight(
        start_mode="fresh",
        layout=layout,
        manifest_path=tmp_path / "manifest.json",
        canonical=tmp_path / "evaluator.py",
        evaluator_snapshot=SimpleNamespace(),
        evaluator_binding={},
        prerequisites=prerequisites,
        activation_lease=activation,
        launch_authority_nonce="b" * 64,
        gpu_lease_binding={},
        gpu_lease=gpu,
        device_guard_lease=device,
    )
    assert observed == binding
    assert events == [
        "load-projection",
        "start-ready-only",
        "close-zero-work",
        "load-projection",
        "validate-binding",
    ]
    assert activation.assertions >= 2
    assert gpu.assertions >= 2
    assert device.assertions >= 2
    assert not output_root.exists()


def test_ready_only_preflight_adopts_valid_proof_and_failure_cannot_publish_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = (tmp_path / "quality").resolve()
    layout = SimpleNamespace(
        output_root=output_root,
        matrix_summary=output_root / matrix.MATRIX_SUMMARY_NAME,
    )
    proof = {"schema_version": 1, "session_role": "ready_only_preflight"}
    projection = {
        "sessions": [
            {
                "session_role": matrix.persistent_session.READY_ONLY_PREFLIGHT_SESSION_ROLE,
                "status": "stopped",
            }
        ]
    }
    monkeypatch.setattr(
        matrix.persistent_session,
        "load_session_ledger_projection",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setattr(
        matrix.persistent_session,
        "ready_only_preflight_binding",
        lambda _projection: proof,
    )
    monkeypatch.setattr(
        matrix,
        "_validate_ready_only_preflight_snapshot",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        matrix,
        "_start_persistent_evaluator",
        lambda **_kwargs: pytest.fail("valid preflight was rerun"),
    )
    lease = _HeldPreflightLease()
    first = matrix.coordinates()[0]
    prerequisites = SimpleNamespace(
        trust_root=object(),
        bundles={
            (first.scale, first.training_seed, first.budget): SimpleNamespace(
                binding={"input_binding_digest": "a" * 64}
            )
        },
        public_binding={},
    )
    adopted = matrix._ensure_ready_only_preflight(
        start_mode="resume",
        layout=layout,
        manifest_path=tmp_path / "manifest.json",
        canonical=tmp_path / "evaluator.py",
        evaluator_snapshot=SimpleNamespace(),
        evaluator_binding={},
        prerequisites=prerequisites,
        activation_lease=lease,
        launch_authority_nonce="c" * 64,
        gpu_lease_binding={},
        gpu_lease=lease,
        device_guard_lease=lease,
    )
    assert adopted == proof
    assert not layout.matrix_summary.exists()

    monkeypatch.setattr(
        matrix.persistent_session,
        "load_session_ledger_projection",
        lambda *_args, **_kwargs: {"sessions": []},
    )
    monkeypatch.setattr(
        matrix.persistent_session,
        "ready_only_preflight_binding",
        lambda _projection: None,
    )

    def fail_start(**_kwargs: object) -> object:
        raise RuntimeError("preflight-launch-failed")

    monkeypatch.setattr(matrix, "_start_persistent_evaluator", fail_start)
    with pytest.raises(RuntimeError, match="preflight-launch-failed"):
        matrix._ensure_ready_only_preflight(
            start_mode="resume",
            layout=layout,
            manifest_path=tmp_path / "manifest.json",
            canonical=tmp_path / "evaluator.py",
            evaluator_snapshot=SimpleNamespace(),
            evaluator_binding={},
            prerequisites=prerequisites,
            activation_lease=lease,
            launch_authority_nonce="d" * 64,
            gpu_lease_binding={},
            gpu_lease=lease,
            device_guard_lease=lease,
        )
    assert not layout.matrix_summary.exists()
    assert not output_root.exists()

    events: list[str] = []

    class CloseFailure:
        terminal_published = False

        def close(self) -> None:
            events.append("close-failed")
            raise RuntimeError("preflight-close-failed")

        def abort_after_parent_commit_failure(self) -> None:
            events.append("failure-terminal")
            self.terminal_published = True

    monkeypatch.setattr(matrix, "_start_persistent_evaluator", lambda **_kwargs: CloseFailure())
    with pytest.raises(RuntimeError, match="preflight-close-failed"):
        matrix._ensure_ready_only_preflight(
            start_mode="resume",
            layout=layout,
            manifest_path=tmp_path / "manifest.json",
            canonical=tmp_path / "evaluator.py",
            evaluator_snapshot=SimpleNamespace(),
            evaluator_binding={},
            prerequisites=prerequisites,
            activation_lease=lease,
            launch_authority_nonce="e" * 64,
            gpu_lease_binding={},
            gpu_lease=lease,
            device_guard_lease=lease,
        )
    assert events == ["close-failed", "failure-terminal"]
    assert not layout.matrix_summary.exists()
    assert not output_root.exists()


def test_ready_only_matrix_binding_rejects_tamper_before_authority_crosscheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconstructed = {"schema_version": 1, "session_role": "ready_only_preflight"}
    monkeypatch.setattr(
        matrix.persistent_session,
        "ready_only_preflight_binding",
        lambda _projection: reconstructed,
    )
    with pytest.raises(ValueError, match="differs from its authenticated ledger"):
        matrix._validate_ready_only_preflight_snapshot(
            {**reconstructed, "session_role": "quality"},
            session_projection={},
            output_root=Path("/tmp/quality"),
            prerequisites=SimpleNamespace(),
            evaluator_binding={},
            gpu_lease_binding={},
        )


def test_evaluator_ready_only_role_accepts_only_eof_before_any_work_or_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    coordinate = dict(contract.quality_coordinates()[0])
    output_root = (tmp_path / "quality").resolve()
    output = output_root / "forbidden-envelope.json"
    plan = {
        "session_nonce": "1" * 64,
        "session_role": contract.V1_3_5_READY_ONLY_PREFLIGHT_SESSION_ROLE,
        "scale": coordinate["scale"],
        "training_seed": coordinate["training_seed"],
        "coordinates": [coordinate],
        "coordinate_count": 1,
        "input_binding_digest": "2" * 64,
        "output_root": str(output_root),
    }
    args = SimpleNamespace(
        launch_nonce=plan["session_nonce"],
        scale=coordinate["scale"],
        training_seed=coordinate["training_seed"],
        budget=coordinate["budget"],
        family=coordinate["family"],
        context=coordinate["context"],
        replicate=coordinate["replicate"],
        output=output,
        checkpoint=tmp_path / "checkpoint.pt",
        training_summary=tmp_path / "training.json",
        training_matrix_summary=tmp_path / "training-matrix.json",
        calibration=tmp_path / "calibration.json",
        reuse_admission=tmp_path / "reuse.json",
        preheldout_genesis=tmp_path / "genesis.json",
        quality_start_activation=tmp_path / "activation.json",
        manifest=tmp_path / "manifest.json",
        device="cuda:0",
        expected_device_identity_type="uuid",
        expected_device_identity="GPU-test",
    )
    monkeypatch.setattr(
        evaluator.persistent_session,
        "read_sealed_plan_fd",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        evaluator,
        "establish_evaluator_inputs",
        lambda **_kwargs: (
            {"input_binding_digest": plan["input_binding_digest"]},
            object(),
            object(),
            {},
            {},
            object(),
        ),
    )
    monkeypatch.setattr(
        evaluator.admission, "assert_quality_context_unchanged", lambda _context: None
    )
    monkeypatch.setattr(evaluator, "_runtime_environment", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluator, "_load_checkpoint_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluator, "_persistent_model_state", lambda _model: ())
    monkeypatch.setattr(evaluator.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(evaluator.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(evaluator.torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(evaluator.gc, "collect", lambda: 0)
    ready = {"status": "stopped", "completed_coordinates": 0}
    monkeypatch.setattr(
        evaluator.persistent_session,
        "build_session_receipt",
        lambda *_args, **_kwargs: ready,
    )
    monkeypatch.setattr(
        evaluator.persistent_session,
        "validate_work_order",
        lambda *_args, **_kwargs: pytest.fail("preflight validated a work order"),
    )
    monkeypatch.setattr(
        evaluator,
        "publish_direct_controller_shard_bundle",
        lambda *_args, **_kwargs: pytest.fail("preflight invoked the workload publisher"),
    )

    messages: list[dict[str, object]] = []
    monkeypatch.setattr(evaluator, "_write_persistent_message", messages.append)
    monkeypatch.setenv(evaluator.PERSISTENT_PLAN_FD_ENV, "9")
    monkeypatch.setattr(
        evaluator.sys,
        "stdin",
        SimpleNamespace(buffer=io.BytesIO(b'{"work":"forbidden"}\n')),
    )
    with pytest.raises(ValueError, match="rejects every work order"):
        evaluator._run_persistent_session(args, quality_context=object(), trust_root=object())
    assert messages == [ready]
    assert not output_root.exists()

    messages.clear()
    monkeypatch.setenv(evaluator.PERSISTENT_PLAN_FD_ENV, "9")
    monkeypatch.setattr(evaluator.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"")))
    evaluator._run_persistent_session(args, quality_context=object(), trust_root=object())
    assert messages == [ready, ready]
    assert messages[0] is messages[1]
    assert not output_root.exists()

    child = inspect.getsource(evaluator._run_persistent_session)
    role_gate = child.index(
        'plan["session_role"] == persistent_session.READY_ONLY_PREFLIGHT_SESSION_ROLE'
    )
    assert role_gate < child.index("persistent_session.validate_work_order(")
    assert role_gate < child.index("publish_direct_controller_shard_bundle(")
