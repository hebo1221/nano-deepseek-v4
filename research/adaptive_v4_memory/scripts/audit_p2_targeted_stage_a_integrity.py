from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import evaluate_p2_targeted_stage_a_shard as stage_a

INTEGRITY_EXPERIMENT_ID = "p2-targeted-stage-a-quality-integrity-audit-v1"
DEFAULT_OUTPUT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2-targeted-stage-a-quality.integrity.summary.json"
)
OUTCOME_DERIVED_KEYS = {
    "accuracy",
    "correct",
    "correct_count",
    "conversation_outcome",
    "paired_difference",
    "mean_difference",
    "effect",
    "p_value",
}
TIER_MEASUREMENT_KEYS = {
    "logical_blocks",
    "hot_blocks",
    "hot_bytes",
    "host_bytes",
    "h2d_bytes",
    "d2h_bytes",
    "late_misses",
    "evictions",
}
ACCOUNTING_MEASUREMENT_KEYS = {
    "state_bytes",
    "hca_bytes",
    "csa_bytes",
    "index_bytes",
    "logical_cache_bytes",
    "hot_resident_bytes",
    "cold_resident_bytes",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_outcome_embargo(value: Any, *, path: str = "raw") -> None:
    """Reject precomputed quality fields without comparing targets to predictions."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in OUTCOME_DERIVED_KEYS:
                raise ValueError(f"Outcome-derived field appeared before integrity: {path}.{key}")
            _assert_outcome_embargo(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_outcome_embargo(item, path=f"{path}[{index}]")


def _expected_conversation_ids(family: str, context: int, local_batch: int) -> list[str]:
    first = local_batch * stage_a.BATCH_SIZE
    return [f"{family}:{context}:{index}" for index in range(first, first + stage_a.BATCH_SIZE)]


def _validate_sha_map(values: object, expected_keys: set[str], label: str) -> None:
    _require(isinstance(values, dict), f"{label} is not a mapping.")
    assert isinstance(values, dict)
    _require(set(values) == expected_keys, f"{label} coverage drifted.")
    _require(
        all(stage_a._is_sha256(value) for value in values.values()),
        f"{label} contains a malformed digest.",
    )


def validate_pin_exposure_by_family(
    pin_exposure: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Require material pin exposure in the one frozen protected-pin family."""

    expected_family_coordinates = (
        len(stage_a.SCALES)
        * len(stage_a.TRAINING_SEEDS)
        * len(stage_a.BUDGETS)
        * len(stage_a.CONTEXTS)
    )
    result: list[dict[str, Any]] = []
    for family in stage_a.PAPER_GRADE_WORKLOAD_FAMILIES:
        values = pin_exposure.get(family, {})
        expected_to_exercise_pins = family == "instruction-persistence"
        protected_batches = values.get("protected_position_batches", 0)
        arms_exercised = {arm: values.get(f"{arm}_pinned_blocks", 0) > 0 for arm in stage_a.ARMS}
        passed = values.get("coordinates", 0) == expected_family_coordinates and (
            protected_batches == expected_family_coordinates * stage_a.BATCHES_PER_SHARD
            and all(arms_exercised.values())
            if expected_to_exercise_pins
            else protected_batches == 0
        )
        result.append(
            {
                "family": family,
                "coordinates": values.get("coordinates", 0),
                "expected_to_exercise_protected_pins": expected_to_exercise_pins,
                "protected_position_batches": protected_batches,
                "pinned_blocks_by_arm": {
                    arm: values.get(f"{arm}_pinned_blocks", 0) for arm in stage_a.ARMS
                },
                "both_configs_exercised_pins_when_expected": (
                    all(arms_exercised.values()) if expected_to_exercise_pins else None
                ),
                "passed": passed,
            }
        )
    return result


def validate_raw_shard(
    raw: dict[str, Any],
    *,
    coordinate: tuple[str, int, str, str, int, int],
    implementation_digest: str,
    manifest_path: Path = stage_a.MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate provenance and controls while leaving outcomes unscored."""

    scale, training_seed, budget, family, context, replicate = coordinate
    evaluation_seed = stage_a.core._evaluation_seed(training_seed)
    generation_seed = stage_a.core._generation_seed(evaluation_seed, family, context, replicate)
    _require(raw.get("schema_version") == 1, "Wrong Stage-A raw schema version.")
    _require(raw.get("experiment_id") == stage_a.EXPERIMENT_ID, "Wrong Stage-A raw id.")
    _require(
        raw.get("status") == "raw_unscored_integrity_pending",
        "Stage-A raw status drifted.",
    )
    for key, expected in {
        "scale": scale,
        "training_seed": training_seed,
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": replicate,
        "evaluation_seed": evaluation_seed,
        "generation_seed": generation_seed,
    }.items():
        _require(raw.get(key) == expected, f"Stage-A coordinate drifted: {key}.")
    _require(
        raw.get("evaluation_seed_namespace") == "reused_post_core_held_out_807_exploratory",
        "Stage-A 807-series reuse disclosure drifted.",
    )
    _require(raw.get("examples") == stage_a.EXAMPLES_PER_SHARD, "Shard size drifted.")
    _require(raw.get("batch_size") == stage_a.BATCH_SIZE, "Batch size drifted.")
    _require(tuple(raw.get("arms", ())) == stage_a.ARMS, "Stage-A arm order drifted.")
    _require(
        raw.get("execution_corner") == "sequential-tiered",
        "Stage-A execution corner drifted.",
    )
    _require(
        raw.get("manifest")
        == {"path": str(manifest_path), "sha256": stage_a.sha256(manifest_path)},
        "Stage-A manifest binding drifted.",
    )
    _require(
        raw.get("original_arm_manifest")
        == {
            "path": str(stage_a.ORIGINAL_CAUSAL_MANIFEST_PATH),
            "sha256": stage_a.sha256(stage_a.ORIGINAL_CAUSAL_MANIFEST_PATH),
        },
        "Original arm-manifest binding drifted.",
    )
    source = raw.get("source", {})
    _require(source.get("dirty") is False, "Dirty Stage-A raw source.")
    _require(
        source.get("implementation_digest") == implementation_digest,
        "Stage-A implementation digest drifted.",
    )
    _require(
        raw.get("outcome_embargo")
        == {
            "quality_accuracy_computed": False,
            "paired_effect_computed": False,
            "outcomes_may_be_read_only_after_terminal_integrity_audit": True,
        },
        "Stage-A outcome embargo drifted.",
    )
    _require(
        raw.get("leakage_and_reuse_disclosure")
        == {
            "calibration_seed_used_for_quality": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_arms": True,
            "reuses_post_core_807_series_inputs": True,
            "reuse_is_explicitly_exploratory_not_fresh_confirmatory_evidence": True,
        },
        "Stage-A leakage/reuse disclosure drifted.",
    )
    _assert_outcome_embargo(raw)

    arm_contract = raw.get("arm_contract", {})
    _require(arm_contract.get("passed") is True, "Stage-A config contract failed.")
    required_config_checks = {
        "same_layer_id_set",
        "same_sorted_layer_quota_multiset",
        "same_sorted_dense_quota_multiset",
        "same_total_configured_blocks",
        "same_total_dense_blocks",
        "same_nonquota_configuration",
        "protected_pins_enabled_both",
        "fallback_disabled_both",
    }
    checks = arm_contract.get("checks", {})
    _require(
        set(checks) == required_config_checks and all(checks.values()),
        "Stage-A exact config checks drifted.",
    )
    configs = arm_contract.get("configs", {})
    config_digests = arm_contract.get("config_sha256", {})
    _require(set(configs) == set(stage_a.ARMS), "Stage-A configs are incomplete.")
    _validate_sha_map(config_digests, set(stage_a.ARMS), "Config digests")
    for arm in stage_a.ARMS:
        _require(
            stage_a.json_digest(configs[arm]) == config_digests[arm],
            f"Stage-A {arm} config digest drifted.",
        )
    calibrated_values = [value for _layer, value in configs[stage_a.ARMS[0]]["layer_budgets"]]
    shuffled_values = [value for _layer, value in configs[stage_a.ARMS[1]]["layer_budgets"]]
    _require(
        sorted(calibrated_values) == sorted(shuffled_values)
        and sum(calibrated_values) == sum(shuffled_values),
        "Stage-A quota multiset or total drifted.",
    )
    _require(
        arm_contract.get("mapping_identified")
        is (configs[stage_a.ARMS[0]]["layer_budgets"] != configs[stage_a.ARMS[1]]["layer_budgets"]),
        "Stage-A mapping-identification label drifted.",
    )

    batches = raw.get("batches", [])
    _require(isinstance(batches, list), "Stage-A batches are not a list.")
    _require(len(batches) == stage_a.BATCHES_PER_SHARD, "Stage-A batch count drifted.")
    _require(raw.get("records_digest") == stage_a.json_digest(batches), "Raw digest drifted.")
    _require(
        raw.get("all_batch_integrity_passed") is True,
        "Stage-A shard recorded an integrity failure.",
    )
    observed_batch_indices: set[int] = set()
    observed_conversations: set[str] = set()
    for batch in batches:
        local_batch = batch.get("local_batch_index")
        _require(type(local_batch) is int, "Stage-A local batch index drifted.")
        assert isinstance(local_batch, int)
        _require(local_batch not in observed_batch_indices, "Duplicate Stage-A local batch.")
        observed_batch_indices.add(local_batch)
        schedule = stage_a.causal.schedule_batch_index(
            family=family,
            context=context,
            replicate=replicate,
            local_batch_index=local_batch,
        )
        _require(
            batch.get("schedule_batch_index") == schedule,
            "Stage-A global schedule coordinate drifted.",
        )
        rotation = schedule % len(stage_a.ARMS)
        order = (*stage_a.ARMS[rotation:], *stage_a.ARMS[:rotation])
        _require(tuple(batch.get("execution_order", ())) == order, "Arm rotation drifted.")
        workload = batch.get("workload", {})
        conversation_ids = _expected_conversation_ids(family, context, local_batch)
        _require(
            workload.get("conversation_ids") == conversation_ids,
            "Stage-A conversation identities drifted.",
        )
        _require(
            not observed_conversations.intersection(conversation_ids),
            "Duplicate Stage-A conversation identity.",
        )
        observed_conversations.update(conversation_ids)
        input_metadata = workload.get("input_ids", {})
        _require(
            input_metadata.get("shape") == [stage_a.BATCH_SIZE, context]
            and stage_a._is_sha256(input_metadata.get("sha256")),
            "Stage-A input tensor metadata drifted.",
        )
        for label in ("query_positions", "targets"):
            metadata = workload.get(label, {})
            values = metadata.get("values")
            _require(
                isinstance(values, list)
                and len(values) == stage_a.BATCH_SIZE
                and all(isinstance(row, list) and row for row in values)
                and stage_a._is_sha256(metadata.get("sha256")),
                f"Stage-A {label} tensor metadata drifted.",
            )
        _require(
            stage_a.json_digest(workload.get("protected_end_positions"))
            == workload.get("protected_end_positions_sha256"),
            "Protected-position digest drifted.",
        )

        arm_runs = batch.get("arm_runs", {})
        _require(set(arm_runs) == set(stage_a.ARMS), "Stage-A arm-run coverage drifted.")
        for execution_index, arm in enumerate(order):
            run = arm_runs[arm]
            _require(
                run.get("arm") == arm
                and run.get("execution_index") == execution_index
                and run.get("execution_path") == "sequential-tiered"
                and run.get("chunk_size") == 1
                and run.get("tiered") is True
                and run.get("config_sha256") == config_digests[arm],
                f"Stage-A {arm} execution metadata drifted.",
            )
            predictions = run.get("predictions")
            _require(
                isinstance(predictions, list)
                and len(predictions) == stage_a.BATCH_SIZE
                and all(isinstance(row, list) and row for row in predictions)
                and stage_a.json_digest(predictions) == run.get("prediction_sha256"),
                f"Stage-A {arm} prediction artifact drifted.",
            )
            controller = run.get("controller", {})
            rows = controller.get("rows", [])
            _require(
                isinstance(rows, list)
                and len(rows) == stage_a.BATCH_SIZE
                and stage_a.json_digest(rows) == controller.get("rows_sha256")
                and all(row.get("budget_violations") == 0 for row in rows),
                f"Stage-A {arm} controller budget audit failed.",
            )
            digests = controller.get("digests_by_conversation", {})
            _require(set(digests) == set(conversation_ids), "Controller digest coverage drifted.")
            for values in digests.values():
                _validate_sha_map(
                    values,
                    {
                        "action_sha256",
                        "selected_position_sha256",
                        "pin_set_sha256",
                        "fallback_action_sha256",
                    },
                    "Controller action digests",
                )
            exposure = controller.get("pin_exposure_by_conversation", {})
            _require(
                set(exposure) == set(conversation_ids)
                and all(
                    set(values) == {"pinned_blocks_sum", "actions_with_pins"}
                    and all(type(value) is int and value >= 0 for value in values.values())
                    for values in exposure.values()
                ),
                "Controller pin-exposure accounting drifted.",
            )
            hot_budget = run.get("physical_hot_budget_blocks_by_layer", {})
            expected_hot = {
                str(layer): value * stage_a.BATCH_SIZE
                for layer, value in configs[arm]["layer_budgets"]
            }
            _require(hot_budget == expected_hot, "Physical hot-budget map drifted.")
            _require(
                isinstance(run.get("accounting"), dict)
                and set(run["accounting"]) == ACCOUNTING_MEASUREMENT_KEYS
                and all(type(value) is int and value >= 0 for value in run["accounting"].values())
                and isinstance(run.get("tier"), dict)
                and set(run["tier"]) == TIER_MEASUREMENT_KEYS
                and all(type(value) is int and value >= 0 for value in run["tier"].values())
                and isinstance(run.get("wall_ms"), (int, float))
                and not isinstance(run.get("wall_ms"), bool)
                and cast(float, run["wall_ms"]) > 0,
                f"Stage-A {arm} physical measurements drifted.",
            )
        controls = batch.get("exact_controls", {})
        expected_control_keys = {
            "config_contract_passed",
            "same_quota_multiset",
            "same_total_configured_blocks",
            "pin_set_digests_identical",
            "fallback_action_digests_identical",
            "no_budget_violations",
            "measured_tier_hot_bytes_identical",
            "measured_tier_hot_blocks_identical",
            "measured_hot_resident_bytes_identical",
        }
        _require(
            set(controls) == expected_control_keys
            and all(controls.values())
            and batch.get("integrity_passed") is True,
            "Stage-A paired exact controls failed.",
        )
        calibrated_digests = arm_runs[stage_a.ARMS[0]]["controller"]["digests_by_conversation"]
        shuffled_digests = arm_runs[stage_a.ARMS[1]]["controller"]["digests_by_conversation"]
        _require(
            all(
                calibrated_digests[conversation]["pin_set_sha256"]
                == shuffled_digests[conversation]["pin_set_sha256"]
                and calibrated_digests[conversation]["fallback_action_sha256"]
                == shuffled_digests[conversation]["fallback_action_sha256"]
                for conversation in conversation_ids
            ),
            "Stage-A runtime pin/fallback control drifted.",
        )
        measured_memory = batch.get("paired_measured_hot_memory", {})
        tier_hot_bytes = {arm: arm_runs[arm]["tier"]["hot_bytes"] for arm in stage_a.ARMS}
        tier_hot_blocks = {arm: arm_runs[arm]["tier"]["hot_blocks"] for arm in stage_a.ARMS}
        hot_resident_bytes = {
            arm: arm_runs[arm]["accounting"]["hot_resident_bytes"] for arm in stage_a.ARMS
        }
        expected_measured_memory = {
            "tier_hot_bytes_by_arm": tier_hot_bytes,
            "tier_hot_blocks_by_arm": tier_hot_blocks,
            "hot_resident_bytes_by_arm": hot_resident_bytes,
            "tier_hot_bytes_identical": len(set(tier_hot_bytes.values())) == 1,
            "tier_hot_blocks_identical": len(set(tier_hot_blocks.values())) == 1,
            "hot_resident_bytes_identical": len(set(hot_resident_bytes.values())) == 1,
            "passed": len(set(tier_hot_bytes.values())) == 1
            and len(set(tier_hot_blocks.values())) == 1
            and len(set(hot_resident_bytes.values())) == 1,
        }
        _require(
            measured_memory == expected_measured_memory and measured_memory["passed"] is True,
            "Stage-A paired measured hot-memory control failed.",
        )
        evidence = batch.get("integrity_evidence", {})
        _require(
            evidence
            == {
                "pin_set_mismatch_count": 0,
                "first_pin_set_mismatch_conversation": None,
                "fallback_action_mismatch_count": 0,
                "first_fallback_action_mismatch_conversation": None,
                "budget_violation_rows_by_arm": {arm: 0 for arm in stage_a.ARMS},
            },
            "Stage-A compact first/aggregate integrity evidence drifted.",
        )
    _require(
        observed_batch_indices == set(range(stage_a.BATCHES_PER_SHARD)),
        "Stage-A local batch coverage drifted.",
    )
    _require(
        len(observed_conversations) == stage_a.EXAMPLES_PER_SHARD,
        "Stage-A conversation coverage drifted.",
    )
    prerequisite = raw.get("prospective_integrity_artifact", {})
    _require(
        prerequisite.get("experiment_id") == stage_a.PROSPECTIVE_INTEGRITY_ID
        and prerequisite.get("observed_path_runs") == stage_a.EXPECTED_PROSPECTIVE_RUNS
        and stage_a._is_sha256(prerequisite.get("sha256")),
        "Stage-A prospective integrity binding drifted.",
    )
    dependencies = [
        raw["manifest"],
        raw["original_arm_manifest"],
        prerequisite,
        raw.get("checkpoint", {}),
        raw.get("calibration_artifact", {}),
    ]
    for dependency in dependencies:
        dependency_path = Path(str(dependency.get("path", "")))
        _require(
            dependency_path.is_file() and stage_a._is_sha256(dependency.get("sha256")),
            "Stage-A dependency metadata is invalid.",
        )
    protected_position_batches = sum(
        bool(batch["workload"]["protected_end_positions"]) for batch in batches
    )
    pinned_blocks_by_arm = {
        arm: sum(
            values["pinned_blocks_sum"]
            for batch in batches
            for values in batch["arm_runs"][arm]["controller"][
                "pin_exposure_by_conversation"
            ].values()
        )
        for arm in stage_a.ARMS
    }
    _require(
        protected_position_batches == 0
        or all(count > 0 for count in pinned_blocks_by_arm.values()),
        "A protected Stage-A coordinate did not exercise pins in every arm/config.",
    )
    return {
        "scale": scale,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": replicate,
        "records_digest": raw["records_digest"],
        "mapping_identified": arm_contract["mapping_identified"],
        "protected_position_batches": protected_position_batches,
        "pinned_blocks_by_arm": pinned_blocks_by_arm,
        "config_sha256_by_arm": config_digests,
        "paired_measured_hot_memory": [
            {
                "local_batch_index": batch["local_batch_index"],
                **batch["paired_measured_hot_memory"],
            }
            for batch in batches
        ],
        "dependencies": dependencies,
    }


def audit_matrix(
    *,
    raw_root: Path,
    manifest_path: Path = stage_a.MANIFEST_PATH,
) -> dict[str, Any]:
    stage_a.load_manifest(manifest_path)
    source = stage_a.source_state()
    _require(source.get("dirty") is False, "Stage-A integrity audit requires clean source.")
    implementation_digest = cast(str, source["implementation_digest"])
    expected = stage_a.expected_coordinates()
    expected_paths = {
        coordinate: stage_a.output_path(
            raw_root,
            scale=coordinate[0],
            training_seed=coordinate[1],
            budget=coordinate[2],
            family=coordinate[3],
            context=coordinate[4],
        )
        for coordinate in expected
    }
    found_paths = set(raw_root.rglob("*.json")) if raw_root.is_dir() else set()
    _require(
        found_paths == set(expected_paths.values()),
        (
            "Stage-A raw shard coverage is not terminal: "
            f"expected {len(expected_paths)}, found {len(found_paths)}."
        ),
    )
    runs: list[dict[str, Any]] = []
    dependency_digests: dict[Path, str] = {}
    identified_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    prospective_paths: set[tuple[str, str]] = set()
    pin_exposure: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pin_exposure_by_config: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    measured_hot_memory_pairs: list[dict[str, Any]] = []
    for coordinate in expected:
        path = expected_paths[coordinate]
        raw = json.loads(path.read_text())
        metadata = validate_raw_shard(
            raw,
            coordinate=coordinate,
            implementation_digest=implementation_digest,
            manifest_path=manifest_path,
        )
        for dependency in metadata.pop("dependencies"):
            dependency_path = Path(str(dependency["path"]))
            expected_sha = str(dependency["sha256"])
            previous = dependency_digests.setdefault(dependency_path, expected_sha)
            _require(previous == expected_sha, "Inconsistent Stage-A dependency digest.")
        prospective = raw["prospective_integrity_artifact"]
        prospective_paths.add((prospective["path"], prospective["sha256"]))
        family = cast(str, metadata["family"])
        pin_exposure[family]["coordinates"] += 1
        pin_exposure[family]["protected_position_batches"] += cast(
            int, metadata["protected_position_batches"]
        )
        for arm, count in metadata["pinned_blocks_by_arm"].items():
            pin_exposure[family][f"{arm}_pinned_blocks"] += count
            config_sha256 = metadata["config_sha256_by_arm"][arm]
            group = pin_exposure_by_config[(family, arm, config_sha256)]
            group["coordinates"] += 1
            group["protected_position_batches"] += cast(int, metadata["protected_position_batches"])
            group["pinned_blocks"] += count
        if metadata["mapping_identified"]:
            identified_counts[
                (metadata["scale"], metadata["budget"], metadata["training_seed"])
            ] += stage_a.EXAMPLES_PER_SHARD
        measured_hot_memory_pairs.extend(
            {
                "scale": metadata["scale"],
                "training_seed": metadata["training_seed"],
                "budget": metadata["budget"],
                "family": metadata["family"],
                "context": metadata["context"],
                **pair,
            }
            for pair in metadata["paired_measured_hot_memory"]
        )
        runs.append(
            {
                **metadata,
                "raw_artifact": {"path": str(path), "sha256": stage_a.sha256(path)},
            }
        )
    _require(len(prospective_paths) == 1, "Prospective Stage-A prerequisite drifted across shards.")
    prospective_path, prospective_sha = next(iter(prospective_paths))
    _require(
        stage_a.sha256(Path(prospective_path)) == prospective_sha,
        "Prospective integrity audit digest drifted.",
    )
    stage_a.require_prospective_integrity(
        Path(prospective_path), manifest_path=manifest_path, verify_raw_records=True
    )
    for path, expected_sha in dependency_digests.items():
        _require(stage_a.sha256(path) == expected_sha, f"Dependency drifted: {path}")
    expected_seed_cells = {
        (scale, budget, seed)
        for scale in stage_a.SCALES
        for budget in stage_a.BUDGETS
        for seed in stage_a.TRAINING_SEEDS
    }
    identification = [
        {
            "scale": scale,
            "budget": budget,
            "training_seed": seed,
            "identified_conversations": identified_counts.get((scale, budget, seed), 0),
            "minimum_required": 200,
            "passed": identified_counts.get((scale, budget, seed), 0) >= 200,
        }
        for scale, budget, seed in sorted(expected_seed_cells)
    ]
    _require(
        all(row["passed"] for row in identification),
        "Uniform-mapping exclusions leave an underpowered Stage-A seed cell.",
    )
    pin_exposure_by_family = validate_pin_exposure_by_family(
        {family: dict(values) for family, values in pin_exposure.items()}
    )
    _require(
        all(row["passed"] for row in pin_exposure_by_family),
        "Protected-position/pin exposure is absent or drifted in an expected family/config.",
    )
    pin_exposure_config_records = [
        {
            "family": family,
            "arm": arm,
            "config_sha256": config_sha256,
            **dict(values),
            "exposure_required": values["protected_position_batches"] > 0,
            "passed": values["protected_position_batches"] == 0 or values["pinned_blocks"] > 0,
        }
        for (family, arm, config_sha256), values in sorted(pin_exposure_by_config.items())
    ]
    _require(
        all(row["passed"] for row in pin_exposure_config_records),
        "Protected-position pin exposure failed for a family/arm/config group.",
    )
    expected_memory_pairs = stage_a.EXPECTED_COORDINATES * stage_a.BATCHES_PER_SHARD
    _require(
        len(measured_hot_memory_pairs) == expected_memory_pairs
        and all(pair["passed"] is True for pair in measured_hot_memory_pairs),
        "Stage-A measured hot-memory pair coverage or equality drifted.",
    )
    runs.sort(
        key=lambda row: (
            row["scale"],
            row["training_seed"],
            row["budget"],
            row["family"],
            row["context"],
            row["replicate"],
        )
    )
    for run in runs:
        raw_metadata = run["raw_artifact"]
        _require(
            stage_a.sha256(Path(raw_metadata["path"])) == raw_metadata["sha256"],
            "A Stage-A raw shard changed during the integrity audit.",
        )
    end_source = stage_a.source_state()
    _require(end_source == source, "Stage-A source changed during the integrity audit.")
    return {
        "schema_version": 1,
        "experiment_id": INTEGRITY_EXPERIMENT_ID,
        "status": "terminal_pass",
        "terminal": True,
        "stage": "A",
        "claim_scope": "Integrity/provenance audit only; no quality outcome was evaluated.",
        "manifest": {"path": str(manifest_path), "sha256": stage_a.sha256(manifest_path)},
        "source": source,
        "raw_root": str(raw_root),
        "expected_shards": stage_a.EXPECTED_COORDINATES,
        "observed_shards": len(runs),
        "expected_arm_conversations": stage_a.EXPECTED_STAGE_A_CONVERSATIONS,
        "coordinate_digest": stage_a.coordinate_digest(),
        "raw_matrix_digest": stage_a.json_digest(runs),
        "runs": runs,
        "uniform_mapping_identification": identification,
        "pin_exposure_by_family": pin_exposure_by_family,
        "pin_exposure_by_family_arm_config": pin_exposure_config_records,
        "paired_measured_hot_memory": {
            "expected_batch_pairs": expected_memory_pairs,
            "observed_batch_pairs": len(measured_hot_memory_pairs),
            "tier_hot_byte_mismatches": sum(
                pair["tier_hot_bytes_identical"] is not True for pair in measured_hot_memory_pairs
            ),
            "tier_hot_block_mismatches": sum(
                pair["tier_hot_blocks_identical"] is not True for pair in measured_hot_memory_pairs
            ),
            "hot_resident_byte_mismatches": sum(
                pair["hot_resident_bytes_identical"] is not True
                for pair in measured_hot_memory_pairs
            ),
            "records_sha256": stage_a.json_digest(measured_hot_memory_pairs),
            "exact_equality_required_before_outcome_access": True,
            "passed": True,
        },
        "audit": {
            "all_required_coordinates_present": True,
            "all_raw_artifact_digests_verified": True,
            "all_dependency_digests_verified": True,
            "prospective_3600_run_sequential_tiered_integrity_verified": True,
            "all_exact_quota_multiset_and_total_controls_verified": True,
            "all_pin_set_and_fallback_controls_verified": True,
            "paired_measured_hot_memory_exactly_equal": True,
            "protected_pin_exposure_verified_by_family_and_config": True,
            "no_budget_violations": True,
            "all_uniform_mapping_exclusions_predeclared_and_sufficient": True,
            "post_core_807_input_reuse_disclosed": True,
            "outcomes_inspected": False,
            "accuracy_computed": False,
            "paired_effect_computed": False,
        },
        "outcome_unlock": {
            "permitted": True,
            "scope": "Stage-A exploratory quality summary only",
            "requires_this_exact_integrity_artifact_sha256": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal outcome-blind integrity audit for Stage-A raw shards."
    )
    parser.add_argument("--raw-root", type=Path, default=stage_a.OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=stage_a.MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage-A integrity output already exists: {args.output}")
    payload = audit_matrix(raw_root=args.raw_root, manifest_path=args.manifest)
    stage_a.write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "observed_shards": payload["observed_shards"],
                "outcomes_inspected": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
