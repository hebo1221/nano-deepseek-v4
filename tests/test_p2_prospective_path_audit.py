from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_prospective_path_audit as runner  # noqa: E402
import summarize_p2_prospective_path_audit as summary  # noqa: E402


def _success_observation(
    *,
    path: str = "tiered-tokenwise",
    role: str = "primary",
    prediction: int = 7,
    query_logits_sha256: str = "1" * 64,
    pin_actions: int = 1,
) -> dict[str, Any]:
    corner = runner.PATH_CORNER_BY_NAME[path]
    payload: dict[str, Any] = {
        "status": "success",
        "path": path,
        "corner": corner,
        "chunk_size": runner.PATH_BY_NAME[path].chunk_size,
        "tiered": runner.PATH_BY_NAME[path].tiered,
        "observation_role": role,
        "repeat_index": int(role == "integrity-replay"),
        "execution_ordinal": 0,
        "conversation_ids": ["conversation:0"],
        "predictions": [[prediction]],
        "prediction_sha256": runner.json_digest([[prediction]]),
        "prediction_shape": [1, 1],
        "query_input_positions": [8],
        "query_logits": {"sha256": query_logits_sha256},
        "post_prefix_trace": {
            "first_input_position": 8,
            "last_input_position": 8,
            "top1_token_ids": [[prediction]],
            "top1_sha256": "2" * 64,
            "logits_sha256": query_logits_sha256,
            "trace_chunks": [
                [
                    {
                        "chunk_index": 0,
                        "start_trace_token_index": 0,
                        "stop_trace_token_index_exclusive": 1,
                        "sha256": "3" * 64,
                    }
                ]
            ],
        },
        "controller": {
            "action_count": 1,
            "path_neutral_action_sha256": "4" * 64,
            "path_neutral_selected_position_sha256": "5" * 64,
            "path_neutral_pin_set_sha256": "6" * 64,
            "path_neutral_fallback_action_sha256": "7" * 64,
            "rows": [{"budget_violations": 0}],
            "rows_sha256": "8" * 64,
            "runtime_replay_digest": "9" * 64,
            "logical_counters": {
                "selected_queries": 1,
                "finalized_control_points": 1,
                "fallback_control_points": 0,
                "peak_selected_blocks": 1,
            },
            "action_chunks": [
                {
                    "chunk_index": 0,
                    "first_identity": {
                        "batch_index": 0,
                        "query_position": 8,
                        "layer_index": 2,
                    },
                    "last_identity": {
                        "batch_index": 0,
                        "query_position": 8,
                        "layer_index": 2,
                    },
                    "actions": 1,
                    "sha256": "a" * 64,
                }
            ],
            "pin_exposure": {
                "actions_with_nonempty_pins": pin_actions,
                "pinned_position_occurrences": pin_actions,
                "distinct_pinned_end_positions": [3] if pin_actions else [],
                "distinct_pinned_end_positions_sha256": "b" * 64,
            },
        },
        "budget_violations": 0,
        "physical_hot_budget_blocks_by_layer": {2: 2},
        "accounting": {"hot_resident_bytes": 10},
        "tier": {"hot_blocks": 2},
        "wall_ms": 1.0,
    }
    unsigned = dict(payload)
    payload["observation_sha256"] = runner.json_digest(unsigned)
    return payload


def _minimal_integrity_payload(*, pin_actions: int, protected: bool) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    configurations: dict[str, Any] = {}
    for arm in runner.STAGE_A_ARMS:
        primary = _success_observation(pin_actions=pin_actions)
        replay = _success_observation(role="integrity-replay", pin_actions=pin_actions)
        comparison = runner.compare_observations(primary, replay, mode="sequential-tiered-repeat")
        arms[arm] = {
            "path_order": [runner.INTEGRITY_PATH_NAME, runner.INTEGRITY_PATH_NAME],
            "observations": [primary, replay],
            "sequential_tiered_repeat_comparison": comparison,
        }
        configurations[arm] = {"config_sha256": f"{arm}:config"}
    return {
        "coordinate": {"family": "instruction-persistence"},
        "workload": {"protected_end_positions": [3] if protected else []},
        "stage_a_exact_controls": {"passed": True},
        "configurations": configurations,
        "arms": arms,
    }


def test_frozen_stage_a_grid_is_exactly_3600_plus_5400_runs() -> None:
    assert runner.EXPECTED_COORDINATES == 900
    assert runner.EXPECTED_ARM_COORDINATES == 1_800
    assert runner.EXPECTED_PHASE_RUNS == {
        "integrity": 3_600,
        "cross-corner": 5_400,
    }
    assert runner.EXPECTED_TOTAL_RUNS == 9_000
    assert runner.EXPECTED_WORKER_COORDINATES == 90
    assert runner.EXPECTED_WORKER_RUNS == {
        "integrity": 360,
        "cross-corner": 540,
    }


def test_generation_seed_formula_is_frozen_and_collision_free() -> None:
    assert (
        runner.generation_seed("s55", 6_071_401, runner.PAPER_GRADE_WORKLOAD_FAMILIES[0], 80)
        == 9_251_801
    )
    seeds = {
        runner.generation_seed(scale, training_seed, family, context)
        for scale in runner.SCALES
        for training_seed in runner.TRAINING_SEEDS
        for family in runner.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in runner.CONTEXTS
    }
    assert len(seeds) == 450


def test_coordinate_and_arm_ordinals_cover_the_full_grid_without_collision() -> None:
    coordinates = {
        runner.coordinate_ordinal(scale, training_seed, budget, family, context)
        for scale in runner.SCALES
        for training_seed in runner.TRAINING_SEEDS
        for budget in runner.BUDGETS
        for family in runner.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in runner.CONTEXTS
    }
    arms = {
        runner.arm_coordinate_ordinal(scale, training_seed, budget, family, context, arm)
        for scale in runner.SCALES
        for training_seed in runner.TRAINING_SEEDS
        for budget in runner.BUDGETS
        for family in runner.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in runner.CONTEXTS
        for arm in runner.STAGE_A_ARMS
    }
    assert coordinates == set(range(900))
    assert arms == set(range(1_800))


def test_frozen_manifest_matches_the_implemented_two_phase_run_set() -> None:
    manifest = runner._validate_frozen_contract()
    panel = manifest["path_split_diagnostic"]["prospective_equivalence_panel"]

    assert panel["expected_stage_a_total_path_runs"] == 9_000
    assert panel["expected_stage_a_arm_corner_records"] == 7_200
    assert panel["expected_stage_a_sequential_tiered_replay_records"] == 1_800
    assert panel["arms_by_stage"]["stage_a_clean_layer_identity"] == list(runner.STAGE_A_ARMS)
    assert (
        manifest["path_split_diagnostic"]["sequential_tiered_eligibility_gate"][
            "cross_corner_results_used_as_blocker"
        ]
        is False
    )


def test_structural_no_go_blocks_prospective_gpu_spend(tmp_path: Path) -> None:
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": runner.STAGE_A_IDENTIFIABILITY_EXPERIMENT_ID,
        "status": "terminal_structural_no_go",
        "terminal": True,
        "observation_boundary": {"outcome_independent": True},
        "preflight": {
            "outcomes_or_807_inputs_inspected": False,
            "current_stage_a_quality_execution_permitted": False,
        },
        "decision": {
            "launch_current_3600_run_prospective_integrity_phase": False,
        },
    }
    report["payload_sha256"] = runner.payload_digest(report)
    path = tmp_path / "identifiability.json"
    path.write_text(json.dumps(report))

    with pytest.raises(RuntimeError, match="structurally NO-GO"):
        runner.require_current_stage_a_structural_go(path)


def test_phase_paths_are_disjoint_resume_shards() -> None:
    integrity = runner.coordinate_path(
        "integrity",
        "s151",
        6_071_405,
        "4x",
        "instruction-persistence",
        1024,
    )
    cross = runner.coordinate_path(
        "cross-corner",
        "s151",
        6_071_405,
        "4x",
        "instruction-persistence",
        1024,
    )
    assert integrity != cross
    assert integrity.is_relative_to(runner.OUTPUT_ROOT)
    assert cross.is_relative_to(runner.OUTPUT_ROOT)
    assert "seed-6071405" in str(integrity)


def test_repeat_comparison_is_strict_but_cross_corner_logit_drift_is_descriptive() -> None:
    reference = _success_observation()
    repeat = _success_observation(role="integrity-replay")
    assert runner.compare_observations(reference, repeat, mode="sequential-tiered-repeat")["passed"]

    logit_drift = _success_observation(path="resident-tokenwise", query_logits_sha256="f" * 64)
    cross = runner.compare_observations(reference, logit_drift, mode="cross-corner")
    assert cross["query_logits_identical"] is False
    assert cross["passed"] is True

    prediction_drift = _success_observation(path="resident-tokenwise", prediction=8)
    mismatch = runner.compare_observations(reference, prediction_drift, mode="cross-corner")
    assert mismatch["passed"] is False
    assert mismatch["first_divergence"]["prediction"]["index"] == 0


def test_error_observation_fails_closed_without_becoming_a_stage_a_cross_blocker() -> None:
    reference = _success_observation()
    error = {
        "status": "error",
        "path": "tiered-chunk2",
        "corner": "chunked-tiered",
        "observation_role": "primary-corner",
        "error": {"sha256": "e" * 64},
    }
    comparison = runner.compare_observations(reference, error, mode="cross-corner")
    assert comparison["passed"] is False
    assert comparison["candidate_error_sha256"] == "e" * 64


def test_payload_digest_detects_mutation() -> None:
    payload = {"schema_version": 1, "value": [1, 2]}
    payload["payload_sha256"] = runner.payload_digest(payload)
    assert payload["payload_sha256"] == runner.payload_digest(payload)

    payload["value"].append(3)
    assert payload["payload_sha256"] != runner.payload_digest(payload)


def test_exclusive_writer_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "stage-a/integrity/cell.json"
    runner._write_json_exclusive(output, {"value": 1}, output_root=tmp_path)
    assert json.loads(output.read_text()) == {"value": 1}

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_json_exclusive(output, {"value": 2}, output_root=tmp_path)
    assert json.loads(output.read_text()) == {"value": 1}


def test_zero_pin_exposure_is_reported_as_no_integrity_evidence() -> None:
    zero = _minimal_integrity_payload(pin_actions=0, protected=True)
    collected, details = summary._collect_integrity([zero])
    pin = collected["pin_exposure"]

    assert pin["passed"] is False
    assert pin["failing_family_arm_configurations"] == 2
    assert details["pin_exposure_failures"]["count"] == 2
    family = pin["by_family"]["instruction-persistence"]
    assert family["passed"] is False
    assert family["evidence_interpretation"].startswith("no pin-integrity evidence")


def test_nonzero_pin_exposure_is_reported_by_family_and_configuration() -> None:
    exposed = _minimal_integrity_payload(pin_actions=2, protected=True)
    collected, details = summary._collect_integrity([exposed])
    family = collected["pin_exposure"]["by_family"]["instruction-persistence"]

    assert collected["pin_exposure"]["passed"] is True
    assert details["pin_exposure_failures"]["count"] == 0
    assert family["distinct_arm_configurations_requiring_exposure"] == 2
    assert family["configurations_with_nonzero_pin_actions"] == 2
    assert family["primary_actions_with_nonempty_pins"] == 4


def test_cross_corner_rotation_is_exactly_balanced_globally() -> None:
    counts = {
        position: {name: 0 for name in runner.CROSS_CORNER_PATH_NAMES}
        for position in range(len(runner.CROSS_CORNER_PATH_NAMES))
    }
    for ordinal in range(runner.EXPECTED_ARM_COORDINATES):
        rotation = ordinal % len(runner.CROSS_CORNER_PATH_NAMES)
        order = (
            *runner.CROSS_CORNER_PATH_NAMES[rotation:],
            *runner.CROSS_CORNER_PATH_NAMES[:rotation],
        )
        for position, name in enumerate(order):
            counts[position][name] += 1

    assert {value for position in counts.values() for value in position.values()} == {600}


def test_observation_digest_covers_compact_controller_evidence() -> None:
    observation = _success_observation()
    assert runner._observation_digest_valid(observation)

    mutated = deepcopy(observation)
    mutated["controller"]["pin_exposure"]["actions_with_nonempty_pins"] = 0
    assert not runner._observation_digest_valid(mutated)
