from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_p2_targeted_stage_a_identifiability as identifiability  # noqa: E402
import audit_p2_targeted_stage_a_integrity as integrity  # noqa: E402
import evaluate_p2_targeted_stage_a_shard as stage_a  # noqa: E402
import summarize_p2_targeted_stage_a as summary  # noqa: E402
from freeze_p2_causal_factorial_arms import build_arm_configs  # noqa: E402


def test_stage_a_manifest_freezes_the_full_exploratory_grid_and_clean_pair() -> None:
    payload = stage_a.load_manifest()

    assert payload["experiment_id"] == stage_a.MANIFEST_EXPERIMENT_ID
    assert len(stage_a.expected_coordinates()) == 900
    assert len(set(stage_a.expected_coordinates())) == 900
    assert stage_a.EXPECTED_STAGE_A_CONVERSATIONS == 36_000
    assert stage_a.EXPECTED_PROSPECTIVE_RUNS == 3_600
    assert stage_a.ARMS == ("calibrated+pins", "shuffled-quota+pins")


def test_current_stage_a_is_structurally_unidentified_before_outcome_access() -> None:
    preflight = stage_a.structural_identifiability_preflight()

    assert preflight["identified_records"] == 10
    assert preflight["unidentified_records"] == 10
    assert preflight["all_four_scale_budget_cells_identified"] is False
    assert preflight["current_stage_a_quality_execution_permitted"] is False
    assert preflight["current_arm_prospective_integrity_spend_recommended"] is False
    assert preflight["outcomes_or_807_inputs_inspected"] is False
    assert {
        (cell["scale"], cell["budget"]): cell["unidentified_training_seeds"]
        for cell in preflight["cells"]
    } == {
        ("s55", "2x"): [6_071_403, 6_071_404],
        ("s55", "4x"): [6_071_404],
        ("s151", "2x"): [6_071_401, 6_071_402, 6_071_403, 6_071_404],
        ("s151", "4x"): [6_071_401, 6_071_402, 6_071_404],
    }


def test_identifiability_report_is_terminal_no_go_and_outcome_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_a,
        "source_state",
        lambda: {
            "commit": "f" * 40,
            "dirty": False,
            "implementation_digest": "a" * 64,
        },
    )

    report = identifiability.build_report()

    assert report["status"] == "terminal_structural_no_go"
    assert report["decision"]["launch_current_3600_run_prospective_integrity_phase"] is False
    assert report["decision"]["launch_current_36000_arm_conversation_quality_matrix"] is False
    assert report["observation_boundary"]["outcome_independent"] is True
    assert report["payload_sha256"] == identifiability._payload_digest(report)


def _actual_clean_configs() -> tuple[Any, Any, dict[str, Any]]:
    path = Path(
        "artifacts/adaptive_v4_memory/paper_grade/calibration_matrix/"
        "s55/seed-6071401/p1-layer-quotas.json"
    )
    calibration = json.loads(path.read_text())
    arms, metadata = build_arm_configs(calibration, "2x", fixed_match=None)
    return arms[stage_a.ARMS[0]].configs[0], arms[stage_a.ARMS[1]].configs[0], metadata


def test_clean_config_contract_requires_exact_multiset_total_pins_and_fallback() -> None:
    calibrated, shuffled, _metadata = _actual_clean_configs()

    contract = stage_a.clean_config_contract(calibrated, shuffled)

    assert contract["passed"] is True
    assert contract["mapping_identified"] is True
    assert contract["checks"] == {
        "same_layer_id_set": True,
        "same_sorted_layer_quota_multiset": True,
        "same_sorted_dense_quota_multiset": True,
        "same_total_configured_blocks": True,
        "same_total_dense_blocks": True,
        "same_nonquota_configuration": True,
        "protected_pins_enabled_both": True,
        "fallback_disabled_both": True,
    }


def _prospective_artifact(tmp_path: Path) -> Path:
    audit: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": stage_a.PROSPECTIVE_INTEGRITY_ID,
        "status": "terminal",
        "phase": "integrity",
        "expected_coordinate_artifacts": stage_a.EXPECTED_COORDINATES,
        "observed_coordinate_artifacts": stage_a.EXPECTED_COORDINATES,
        "expected_path_runs": stage_a.EXPECTED_PROSPECTIVE_RUNS,
        "observed_path_runs": stage_a.EXPECTED_PROSPECTIVE_RUNS,
        "implementation_digest": "a" * 64,
        "auditor_implementation_digest": "b" * 64,
        "source": {"dirty": False, "implementation_digest": "a" * 64},
        "frozen_manifest": {
            "path": str(stage_a.MANIFEST_PATH),
            "sha256": stage_a.sha256(stage_a.MANIFEST_PATH),
        },
        "artifact_sets": {
            "integrity": {
                "coordinate_artifacts": stage_a.EXPECTED_COORDINATES,
                "worker_receipts": len(stage_a.SCALES) * len(stage_a.TRAINING_SEEDS),
            }
        },
        "workload_grid": {"passed": True},
        "sequential_tiered_eligibility": {
            "expected_repeat_comparisons": stage_a.EXPECTED_COORDINATES * len(stage_a.ARMS),
            "observed_repeat_comparisons": stage_a.EXPECTED_COORDINATES * len(stage_a.ARMS),
            "failing_repeat_comparisons": 0,
            "error_observations": 0,
            "budget_violations": 0,
            "exact_control_failures": 0,
            "pin_exposure": {
                "passed": True,
                "failing_family_arm_configurations": 0,
            },
            "sequential_tiered_eligibility_passed": True,
            "cross_corner_results_used_as_blocker": False,
        },
        "targeted_stage_a_eligible": True,
        "cross_corner_interchangeability": None,
        "cross_corner_results_used_as_stage_a_blocker": False,
        "failure_details": {
            "integrity": {
                "failed_comparisons": {"count": 0},
                "errors": {"count": 0},
                "pin_exposure_failures": {"count": 0},
            }
        },
    }
    audit["payload_sha256"] = stage_a.prospective.payload_digest(audit)
    path = tmp_path / "prospective.summary.json"
    path.write_text(json.dumps(audit))
    return path


def test_prospective_integrity_requires_all_3600_same_path_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _prospective_artifact(tmp_path)
    monkeypatch.setattr(stage_a.prospective, "implementation_digest", lambda: "a" * 64)
    monkeypatch.setattr(
        stage_a.prospective_summary, "auditor_implementation_digest", lambda: "b" * 64
    )
    monkeypatch.setattr(stage_a.prospective, "integrity_summary_path", lambda: path)

    payload = stage_a.require_prospective_integrity(path, verify_raw_records=False)

    assert payload["observed_path_runs"] == 3_600
    payload["observed_path_runs"] -= 1
    payload["payload_sha256"] = stage_a.prospective.payload_digest(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="terminal integrity audit"):
        stage_a.require_prospective_integrity(path, verify_raw_records=False)


def _dependency(tmp_path: Path, name: str) -> dict[str, Any]:
    path = tmp_path / name
    path.write_text(name)
    return {"path": str(path), "sha256": stage_a.sha256(path)}


def _fake_raw_shard(tmp_path: Path) -> tuple[dict[str, Any], tuple[str, int, str, str, int, int]]:
    coordinate = stage_a.expected_coordinates()[0]
    scale, training_seed, budget, family, context, replicate = coordinate
    calibrated, shuffled, metadata = _actual_clean_configs()
    configs = {
        stage_a.ARMS[0]: asdict(calibrated),
        stage_a.ARMS[1]: asdict(shuffled),
    }
    config_digests = {arm: stage_a.json_digest(config) for arm, config in configs.items()}
    contract = stage_a.clean_config_contract(calibrated, shuffled)
    prospective = _dependency(tmp_path, "prospective.summary.json")
    prospective.update(
        {
            "experiment_id": stage_a.PROSPECTIVE_INTEGRITY_ID,
            "observed_path_runs": stage_a.EXPECTED_PROSPECTIVE_RUNS,
        }
    )
    checkpoint = _dependency(tmp_path, "checkpoint.pt")
    checkpoint["bytes"] = Path(checkpoint["path"]).stat().st_size
    calibration = _dependency(tmp_path, "calibration.json")
    calibration.update({"calibration_seed": 7071401, "calibration_digest": "b" * 64})
    batches: list[dict[str, Any]] = []
    for local_batch in range(stage_a.BATCHES_PER_SHARD):
        schedule = stage_a.causal.schedule_batch_index(
            family=family,
            context=context,
            replicate=replicate,
            local_batch_index=local_batch,
        )
        rotation = schedule % len(stage_a.ARMS)
        order = (*stage_a.ARMS[rotation:], *stage_a.ARMS[:rotation])
        ids = [
            f"{family}:{context}:{index}"
            for index in range(
                local_batch * stage_a.BATCH_SIZE,
                (local_batch + 1) * stage_a.BATCH_SIZE,
            )
        ]
        digests = {
            conversation: {
                "action_sha256": "1" * 64,
                "selected_position_sha256": "2" * 64,
                "pin_set_sha256": "3" * 64,
                "fallback_action_sha256": "4" * 64,
            }
            for conversation in ids
        }
        exposure = {
            conversation: {"pinned_blocks_sum": 0, "actions_with_pins": 0} for conversation in ids
        }
        runs = {}
        for execution_index, arm in enumerate(order):
            predictions = [[1] for _ in ids]
            rows = [{"budget_violations": 0} for _ in ids]
            runs[arm] = {
                "arm": arm,
                "execution_index": execution_index,
                "execution_path": "sequential-tiered",
                "chunk_size": 1,
                "tiered": True,
                "config_sha256": config_digests[arm],
                "predictions": predictions,
                "prediction_sha256": stage_a.json_digest(predictions),
                "controller": {
                    "action_count": 1,
                    "semantic_action_sha256": "5" * 64,
                    "digests_by_conversation": copy.deepcopy(digests),
                    "pin_exposure_by_conversation": copy.deepcopy(exposure),
                    "rows": rows,
                    "rows_sha256": stage_a.json_digest(rows),
                    "logical_counters": {},
                },
                "physical_hot_budget_blocks_by_layer": {
                    str(layer): value * stage_a.BATCH_SIZE
                    for layer, value in configs[arm]["layer_budgets"]
                },
                "accounting": {
                    "state_bytes": 200,
                    "hca_bytes": 0,
                    "csa_bytes": 0,
                    "index_bytes": 0,
                    "logical_cache_bytes": 200,
                    "hot_resident_bytes": 200,
                    "cold_resident_bytes": 0,
                },
                "tier": {
                    "logical_blocks": 50,
                    "hot_blocks": 50,
                    "hot_bytes": 100,
                    "host_bytes": 0,
                    "h2d_bytes": 0,
                    "d2h_bytes": 0,
                    "late_misses": 0,
                    "evictions": 0,
                },
                "wall_ms": 1.0,
            }
        protected: list[int] = []
        batches.append(
            {
                "local_batch_index": local_batch,
                "schedule_batch_index": schedule,
                "execution_order": list(order),
                "workload": {
                    "conversation_ids": ids,
                    "input_ids": {"shape": [stage_a.BATCH_SIZE, context], "sha256": "6" * 64},
                    "query_positions": {"values": [[context - 1] for _ in ids], "sha256": "7" * 64},
                    "targets": {"values": [[1] for _ in ids], "sha256": "8" * 64},
                    "protected_end_positions": protected,
                    "protected_end_positions_sha256": stage_a.json_digest(protected),
                },
                "arm_runs": runs,
                "exact_controls": {
                    "config_contract_passed": True,
                    "same_quota_multiset": True,
                    "same_total_configured_blocks": True,
                    "pin_set_digests_identical": True,
                    "fallback_action_digests_identical": True,
                    "no_budget_violations": True,
                    "measured_tier_hot_bytes_identical": True,
                    "measured_tier_hot_blocks_identical": True,
                    "measured_hot_resident_bytes_identical": True,
                },
                "paired_measured_hot_memory": {
                    "tier_hot_bytes_by_arm": {arm: 100 for arm in stage_a.ARMS},
                    "tier_hot_blocks_by_arm": {arm: 50 for arm in stage_a.ARMS},
                    "hot_resident_bytes_by_arm": {arm: 200 for arm in stage_a.ARMS},
                    "tier_hot_bytes_identical": True,
                    "tier_hot_blocks_identical": True,
                    "hot_resident_bytes_identical": True,
                    "passed": True,
                },
                "integrity_evidence": {
                    "pin_set_mismatch_count": 0,
                    "first_pin_set_mismatch_conversation": None,
                    "fallback_action_mismatch_count": 0,
                    "first_fallback_action_mismatch_conversation": None,
                    "budget_violation_rows_by_arm": {arm: 0 for arm in stage_a.ARMS},
                },
                "integrity_passed": True,
            }
        )
    raw = {
        "schema_version": 1,
        "experiment_id": stage_a.EXPERIMENT_ID,
        "status": "raw_unscored_integrity_pending",
        "scale": scale,
        "training_seed": training_seed,
        "evaluation_seed_namespace": "reused_post_core_held_out_807_exploratory",
        "evaluation_seed": stage_a.core._evaluation_seed(training_seed),
        "generation_seed": stage_a.core._generation_seed(
            stage_a.core._evaluation_seed(training_seed), family, context, replicate
        ),
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": replicate,
        "examples": stage_a.EXAMPLES_PER_SHARD,
        "batch_size": stage_a.BATCH_SIZE,
        "arms": list(stage_a.ARMS),
        "execution_corner": "sequential-tiered",
        "manifest": {
            "path": str(stage_a.MANIFEST_PATH),
            "sha256": stage_a.sha256(stage_a.MANIFEST_PATH),
        },
        "original_arm_manifest": {
            "path": str(stage_a.ORIGINAL_CAUSAL_MANIFEST_PATH),
            "sha256": stage_a.sha256(stage_a.ORIGINAL_CAUSAL_MANIFEST_PATH),
        },
        "prospective_integrity_artifact": prospective,
        "checkpoint": checkpoint,
        "calibration_artifact": calibration,
        "arm_contract": {
            **contract,
            "shuffle_offset": metadata["shuffle_offset"],
            "shuffled_is_structurally_identical": metadata["shuffled_is_structurally_identical"],
            "configs": configs,
            "config_sha256": config_digests,
        },
        "records_digest": stage_a.json_digest(batches),
        "batches": batches,
        "all_batch_integrity_passed": True,
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "outcome_embargo": {
            "quality_accuracy_computed": False,
            "paired_effect_computed": False,
            "outcomes_may_be_read_only_after_terminal_integrity_audit": True,
        },
        "leakage_and_reuse_disclosure": {
            "calibration_seed_used_for_quality": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_arms": True,
            "reuses_post_core_807_series_inputs": True,
            "reuse_is_explicitly_exploratory_not_fresh_confirmatory_evidence": True,
        },
    }
    return raw, coordinate


def test_integrity_validator_is_outcome_blind_and_checks_compact_digests(
    tmp_path: Path,
) -> None:
    raw, coordinate = _fake_raw_shard(tmp_path)

    metadata = integrity.validate_raw_shard(
        raw, coordinate=coordinate, implementation_digest="implementation"
    )

    assert metadata["mapping_identified"] is True
    assert metadata["protected_position_batches"] == 0
    assert metadata["pinned_blocks_by_arm"] == {arm: 0 for arm in stage_a.ARMS}
    leaked = copy.deepcopy(raw)
    leaked["batches"][0]["arm_runs"][stage_a.ARMS[0]]["correct"] = [[True]]
    with pytest.raises(ValueError, match="Outcome-derived field"):
        integrity.validate_raw_shard(
            leaked, coordinate=coordinate, implementation_digest="implementation"
        )


def test_integrity_rejects_paired_measured_hot_memory_mismatch(tmp_path: Path) -> None:
    raw, coordinate = _fake_raw_shard(tmp_path)
    batch = raw["batches"][0]
    candidate = stage_a.ARMS[0]
    batch["arm_runs"][candidate]["tier"]["hot_bytes"] = 101
    batch["paired_measured_hot_memory"] = {
        "tier_hot_bytes_by_arm": {
            stage_a.ARMS[0]: 101,
            stage_a.ARMS[1]: 100,
        },
        "tier_hot_blocks_by_arm": {arm: 50 for arm in stage_a.ARMS},
        "hot_resident_bytes_by_arm": {arm: 200 for arm in stage_a.ARMS},
        "tier_hot_bytes_identical": False,
        "tier_hot_blocks_identical": True,
        "hot_resident_bytes_identical": True,
        "passed": False,
    }
    batch["exact_controls"]["measured_tier_hot_bytes_identical"] = False
    batch["integrity_passed"] = False
    raw["all_batch_integrity_passed"] = False
    raw["records_digest"] = stage_a.json_digest(raw["batches"])

    with pytest.raises(ValueError, match="integrity failure"):
        integrity.validate_raw_shard(
            raw, coordinate=coordinate, implementation_digest="implementation"
        )


def test_quality_outcomes_are_paired_only_after_integrity_structure_passes(
    tmp_path: Path,
) -> None:
    raw, coordinate = _fake_raw_shard(tmp_path)
    integrity.validate_raw_shard(raw, coordinate=coordinate, implementation_digest="implementation")

    records = summary.extract_paired_differences(raw)

    assert len(records) == stage_a.EXAMPLES_PER_SHARD
    assert all(record["paired_difference"] == 0.0 for record in records)


def test_expected_pin_family_requires_nontrivial_exposure_for_both_configs() -> None:
    coordinates_per_family = (
        len(stage_a.SCALES)
        * len(stage_a.TRAINING_SEEDS)
        * len(stage_a.BUDGETS)
        * len(stage_a.CONTEXTS)
    )
    values = {
        family: {
            "coordinates": coordinates_per_family,
            "protected_position_batches": (
                coordinates_per_family * stage_a.BATCHES_PER_SHARD
                if family == "instruction-persistence"
                else 0
            ),
            f"{stage_a.ARMS[0]}_pinned_blocks": (1 if family == "instruction-persistence" else 0),
            f"{stage_a.ARMS[1]}_pinned_blocks": (1 if family == "instruction-persistence" else 0),
        }
        for family in stage_a.PAPER_GRADE_WORKLOAD_FAMILIES
    }

    passed = integrity.validate_pin_exposure_by_family(values)
    assert all(row["passed"] for row in passed)
    values["instruction-persistence"][f"{stage_a.ARMS[1]}_pinned_blocks"] = 0
    failed = integrity.validate_pin_exposure_by_family(values)
    instruction = next(row for row in failed if row["family"] == "instruction-persistence")
    assert instruction["passed"] is False
    assert instruction["both_configs_exercised_pins_when_expected"] is False


def test_stage_a_bootstrap_sign_flip_and_gate_are_exact_and_deterministic() -> None:
    by_seed = [
        {
            "scale": "s151",
            "budget": "2x",
            "training_seed": seed,
            "eligible_paired_conversations": 900,
            "mean_difference": value,
        }
        for seed, value in zip(stage_a.TRAINING_SEEDS, (0.01, 0.02, 0.03, 0.04, 0.05), strict=True)
    ]

    first = summary.cell_statistics(by_seed, scale="s151", budget="2x", cell_offset=0)
    second = summary.cell_statistics(by_seed, scale="s151", budget="2x", cell_offset=0)

    assert first == second
    assert first["pooled_effect"] == pytest.approx(0.03)
    assert first["positive_seed_effects"] == 5
    assert first["bootstrap"]["seed"] == 9_171_803
    assert first["bootstrap"]["resamples"] == 10_000
    assert all(first["gate_checks"].values())
    assert summary.exact_seed_sign_flip([0.0] * 5) == 1.0


def test_exclusive_stage_a_write_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    stage_a.write_json_exclusive(path, {"first": True})

    with pytest.raises(FileExistsError):
        stage_a.write_json_exclusive(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}
