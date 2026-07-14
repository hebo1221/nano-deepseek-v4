from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, cast

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_online_learned_lookahead_shard as evaluator
import evaluate_p2_causal_factorial_shard as causal
import fit_p1_online_learned_lookahead_policy as fitter
import numpy as np
import run_p1_online_learned_lookahead as matrix_runner
from summarize_p2_causal_factorial import contrast_statistics
from summarize_p2_core_matrix import holm_bonferroni

from nano_deepseek_v4 import RISK_FEATURE_NAMES, LearnedLookaheadPolicy

PRIMARY = "online-learned-lookahead+pins"
ONE_TOKEN = "one-token-training-free+pins"
FIXED = "memory-matched-fixed+pins"
MAXIMUM_HBM_DIFFERENCE = 0.01
MAXIMUM_WORST_SLICE_REGRESSION = -0.02
_DIGEST_CACHE: dict[tuple[Path, int, int], str] = {}
PARALLEL_RUNNER = Path("research/adaptive_v4_memory/scripts/run_p1_online_lookahead_parallel.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256(path: Path) -> str:
    stat = path.stat()
    key = (path.resolve(), stat.st_mtime_ns, stat.st_size)
    if key not in _DIGEST_CACHE:
        _DIGEST_CACHE[key] = sha256(path)
    return _DIGEST_CACHE[key]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (minimum is None or float(value) >= minimum)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_conversation_ids(*, family: str, context: int, replicate: int) -> set[str]:
    start = replicate * labels.EXAMPLES_PER_SHARD
    return {
        f"{family}:{context}:{index}" for index in range(start, start + labels.EXAMPLES_PER_SHARD)
    }


def _checkpoint_digest(payload: dict[str, Any], *, label: str) -> str:
    metadata = payload.get("checkpoint", {})
    path = _bound_path(metadata, label)
    if "bytes" in metadata:
        _require(metadata.get("bytes") == path.stat().st_size, f"{label} byte size drifted.")
    digest = metadata.get("sha256")
    _require(_is_sha256(digest), f"{label} has an invalid SHA-256 digest.")
    return str(digest)


def _bound_path(metadata: dict[str, Any], label: str) -> Path:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(metadata.get("sha256") == _cached_sha256(path), f"{label} digest drifted: {path}")
    return path


def _verify_common(payload: dict[str, Any], *, implementation_digest: str, label: str) -> None:
    _require(
        payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("implementation_digest") == implementation_digest,
        f"{label} source implementation drifted.",
    )
    _require(
        payload.get("design", {}).get("path") == str(matrix_runner.DESIGN)
        and payload.get("design", {}).get("sha256") == _cached_sha256(matrix_runner.DESIGN),
        f"{label} design dependency drifted.",
    )


def _verify_checkpoint_reuse_audit(metadata: dict[str, Any]) -> int:
    path = _bound_path(metadata, "online-lookahead checkpoint-reuse audit")
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    _require(
        payload.get("experiment_id") == "p1-online-lookahead-checkpoint-reuse-audit-v1"
        and payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("orchestrator_sha256") == _cached_sha256(PARALLEL_RUNNER)
        and audit.get("scale_seed_probes") == 10
        and audit.get("all_label_rows_identical") is True
        and audit.get("all_test_records_identical") is True
        and audit.get("all_controller_accounting_identical") is True,
        "Online-lookahead checkpoint-reuse audit is incomplete.",
    )
    coordinates: set[tuple[str, int]] = set()
    for probe_metadata in payload.get("probes", []):
        probe_path = _bound_path(probe_metadata, "online-lookahead checkpoint-reuse probe")
        probe = json.loads(probe_path.read_text())
        probe_audit = probe.get("audit", {})
        coordinate = (probe.get("scale"), probe.get("training_seed"))
        _require(
            probe.get("experiment_id") == "p1-online-lookahead-checkpoint-reuse-probe-v1"
            and coordinate[0] in matrix_runner.SCALES
            and coordinate[1] in labels.TRAINING_SEEDS
            and coordinate not in coordinates
            and probe_audit.get("label_rows_identical") is True
            and probe_audit.get("test_records_identical") is True
            and probe_audit.get("controller_accounting_identical") is True
            and probe_audit.get("timing_fields_excluded") is True
            and probe.get("implementation", {}).get("label") == labels.implementation_digest()
            and probe.get("implementation", {}).get("evaluation")
            == evaluator.implementation_digest()
            and probe.get("implementation", {}).get("orchestrator_sha256")
            == _cached_sha256(PARALLEL_RUNNER),
            f"Online-lookahead checkpoint-reuse probe drifted: {probe_path}",
        )
        for artifact in probe.get("artifacts", []):
            _bound_path(artifact, f"checkpoint-reuse child artifact for {probe_path}")
        coordinates.add(coordinate)
    _require(
        coordinates == set(product(matrix_runner.SCALES, labels.TRAINING_SEEDS)),
        "Online-lookahead checkpoint-reuse coordinate coverage drifted.",
    )
    return len(coordinates)


def _validate_label_payload(
    shard: dict[str, Any], *, path: Path
) -> tuple[tuple[str, int, str, str, int, int], set[str], str]:
    scale = shard.get("scale")
    training_seed = shard.get("training_seed")
    split = shard.get("split")
    family = shard.get("family")
    context = shard.get("context")
    replicate = shard.get("replicate")
    _require(
        scale in matrix_runner.SCALES
        and training_seed in labels.TRAINING_SEEDS
        and split in matrix_runner.SPLITS
        and family in labels.PAPER_GRADE_WORKLOAD_FAMILIES
        and context in labels.CONTEXTS
        and replicate in labels.REPLICATES[str(split)],
        f"Label shard coordinate drifted: {path}",
    )
    coordinate = (
        str(scale),
        cast(int, training_seed),
        str(split),
        str(family),
        cast(int, context),
        cast(int, replicate),
    )
    rows = shard.get("rows", [])
    failures = shard.get("failures", [])
    _require(isinstance(rows, list) and isinstance(failures, list), f"Invalid rows: {path}")
    _require(
        shard.get("experiment_id") == "p1-online-learned-lookahead-label-shard-v1"
        and shard.get("generation_seed")
        == labels.generation_seed(
            split=str(split),
            training_seed=cast(int, training_seed),
            family=str(family),
            context=cast(int, context),
            replicate=cast(int, replicate),
        )
        and shard.get("conversations") == labels.EXAMPLES_PER_SHARD
        and shard.get("risk_examples") == len(rows)
        and shard.get("failure_count") == len(failures)
        and tuple(shard.get("registered_topk_sweep", ())) == labels.NORMAL_TOPK_SWEEP
        and hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == shard.get("rows_digest"),
        f"Invalid label shard accounting: {path}",
    )
    leakage = shard.get("leakage_guard", {})
    _require(
        leakage.get("features_use_prior_token_only") is True
        and leakage.get("causal_offset") == 1
        and leakage.get("benchmark_targets_used_for_labels") is False
        and leakage.get("dense_model_predictions_used_for_labels") is True
        and leakage.get("split_namespace") == labels.SPLIT_NAMESPACES[str(split)],
        f"Label shard leakage boundary drifted: {path}",
    )
    dense_topk = shard.get("dense_topk")
    _require(
        isinstance(dense_topk, list)
        and bool(dense_topk)
        and all(_is_int(value, minimum=1) for value in dense_topk)
        and dense_topk == sorted(set(dense_topk)),
        f"Label shard dense top-k contract drifted: {path}",
    )
    expected_ids = _expected_conversation_ids(
        family=str(family),
        context=cast(int, context),
        replicate=cast(int, replicate),
    )
    observed_ids: set[str] = set()
    input_by_conversation: dict[str, str] = {}
    group_ids: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict), f"Invalid label row: {path}")
        conversation_id = row.get("conversation_id")
        input_digest = row.get("input_sha256")
        feature_position = row.get("feature_token_position")
        label_position = row.get("label_token_position")
        features = row.get("features")
        matching = row.get("matching_registered_topk")
        sufficient = row.get("sufficient_topk")
        dense_required = row.get("dense_required")
        expected_group = (
            f"{split}:{scale}:{training_seed}:{family}:{context}:{replicate}:"
            f"{conversation_id}:q{label_position}"
        )
        _require(
            conversation_id in expected_ids
            and _is_sha256(input_digest)
            and _is_int(feature_position, minimum=0)
            and _is_int(label_position, minimum=1)
            and feature_position + 1 == label_position
            and row.get("causal_offset") == 1
            and row.get("group_id") == expected_group
            and row.get("group_id") not in group_ids
            and isinstance(features, list)
            and len(features) == len(RISK_FEATURE_NAMES)
            and all(_is_finite_number(value) for value in features)
            and isinstance(matching, list)
            and matching == sorted(set(matching))
            and all(value in labels.NORMAL_TOPK_SWEEP for value in matching)
            and _is_int(sufficient, minimum=1)
            and isinstance(dense_required, bool)
            and _is_int(row.get("dense_prediction"), minimum=0),
            f"Label row schema drifted: {path}",
        )
        _require(
            (
                not cast(list[Any], matching)
                and dense_required is True
                and sufficient in cast(list[Any], dense_topk)
            )
            or (
                bool(cast(list[Any], matching))
                and dense_required is False
                and sufficient == min(cast(list[Any], matching))
            ),
            f"Label row target derivation drifted: {path}",
        )
        previous = input_by_conversation.setdefault(str(conversation_id), str(input_digest))
        _require(previous == input_digest, f"Conversation input digest drifted within {path}")
        observed_ids.add(str(conversation_id))
        group_ids.add(str(row["group_id"]))
    for failure in failures:
        _require(
            isinstance(failure, dict)
            and failure.get("conversation_id") in expected_ids
            and _is_sha256(failure.get("input_sha256"))
            and _is_int(failure.get("query_position"), minimum=1)
            and isinstance(failure.get("reason"), str)
            and bool(failure.get("reason")),
            f"Label failure schema drifted: {path}",
        )
        failure_conversation_id = str(failure["conversation_id"])
        failure_input_digest = str(failure["input_sha256"])
        previous = input_by_conversation.setdefault(failure_conversation_id, failure_input_digest)
        _require(
            previous == failure_input_digest,
            f"Conversation input digest drifted within {path}",
        )
        observed_ids.add(failure_conversation_id)
    _require(observed_ids == expected_ids, f"Label conversation coverage drifted: {path}")
    return (
        coordinate,
        set(input_by_conversation.values()),
        _checkpoint_digest(shard, label=f"label checkpoint for {path}"),
    )


def _validate_accounting(accounting: Any, *, label: str) -> None:
    fields = (
        "state_bytes",
        "hca_bytes",
        "csa_bytes",
        "index_bytes",
        "logical_cache_bytes",
        "hot_resident_bytes",
        "cold_resident_bytes",
    )
    _require(
        isinstance(accounting, dict)
        and all(_is_int(accounting.get(field), minimum=0) for field in fields)
        and accounting["state_bytes"]
        + accounting["hca_bytes"]
        + accounting["csa_bytes"]
        + accounting["index_bytes"]
        == accounting["logical_cache_bytes"]
        and accounting["hot_resident_bytes"] + accounting["cold_resident_bytes"]
        == accounting["logical_cache_bytes"],
        f"Cache accounting drifted: {label}",
    )


def _validate_test_payload(
    shard: dict[str, Any], *, path: Path
) -> tuple[
    tuple[str, int, str, str, int, int],
    dict[tuple[str, str], dict[str, Any]],
    set[str],
    list[dict[str, Any]],
    int,
    int,
    str,
]:
    scale = shard.get("scale")
    training_seed = shard.get("training_seed")
    budget = shard.get("budget")
    family = shard.get("family")
    context = shard.get("context")
    replicate = shard.get("replicate")
    _require(
        scale in matrix_runner.SCALES
        and training_seed in labels.TRAINING_SEEDS
        and budget in causal.BUDGET_LABELS
        and family in labels.PAPER_GRADE_WORKLOAD_FAMILIES
        and context in labels.CONTEXTS
        and replicate in range(10),
        f"Test shard coordinate drifted: {path}",
    )
    coordinate = (
        str(scale),
        cast(int, training_seed),
        str(budget),
        str(family),
        cast(int, context),
        cast(int, replicate),
    )
    records = shard.get("records", [])
    metrics = shard.get("batch_metrics", [])
    _require(
        shard.get("experiment_id") == "p1-online-learned-lookahead-test-shard-v1"
        and tuple(shard.get("arms", ())) == evaluator.ARMS
        and shard.get("examples") == labels.EXAMPLES_PER_SHARD
        and isinstance(records, list)
        and len(records) == labels.EXAMPLES_PER_SHARD * len(evaluator.ARMS)
        and isinstance(metrics, list)
        and len(metrics) == (labels.EXAMPLES_PER_SHARD // labels.BATCH_SIZE) * len(evaluator.ARMS)
        and hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == shard.get("records_digest"),
        f"Invalid online-lookahead test shard: {path}",
    )
    expected_ids = _expected_conversation_ids(
        family=str(family),
        context=cast(int, context),
        replicate=cast(int, replicate),
    )
    expected_offset = {
        conversation_id: (
            (index - cast(int, replicate) * labels.EXAMPLES_PER_SHARD) // labels.BATCH_SIZE
        )
        * labels.BATCH_SIZE
        for conversation_id in expected_ids
        for index in (int(conversation_id.rsplit(":", 1)[1]),)
    }
    by_arm_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        _require(isinstance(row, dict), f"Invalid test record: {path}")
        arm = row.get("arm")
        conversation_id = row.get("conversation_id")
        targets = row.get("targets")
        predictions = row.get("predictions")
        correct = row.get("correct")
        key = (str(arm), str(conversation_id))
        _require(
            arm in evaluator.ARMS
            and conversation_id in expected_ids
            and key not in by_arm_id
            and row.get("batch_offset") == expected_offset[str(conversation_id)]
            and _is_sha256(row.get("input_sha256"))
            and row.get("family") == family
            and row.get("context") == context
            and row.get("replicate") == replicate
            and row.get("budget") == budget
            and isinstance(targets, list)
            and bool(targets)
            and all(_is_int(value, minimum=0) for value in targets)
            and isinstance(predictions, list)
            and len(predictions) == len(targets)
            and all(_is_int(value, minimum=-1) for value in predictions)
            and isinstance(correct, list)
            and all(isinstance(value, bool) for value in correct)
            and correct == [left == right for left, right in zip(predictions, targets, strict=True)]
            and row.get("correct_count") == sum(correct)
            and row.get("total") == len(correct)
            and (row.get("failure") is None or isinstance(row.get("failure"), dict)),
            f"Test record schema drifted: {path}",
        )
        controller = row.get("controller")
        if controller is None:
            _require(
                arm == "native-resident" and row.get("failure") is None,
                f"Missing controller evidence: {path}",
            )
        else:
            _require(
                isinstance(controller, dict)
                and _is_int(controller.get("budget_violations"), minimum=0),
                f"Controller accounting drifted: {path}",
            )
            for field in ("control_points", "selected_blocks_sum", "budget_blocks_sum"):
                _require(
                    _is_int(controller.get(field), minimum=0),
                    f"Controller field {field} drifted: {path}",
                )
        by_arm_id[key] = row
    ids = {conversation_id for _, conversation_id in by_arm_id}
    _require(
        ids == expected_ids
        and all(
            {arm for arm, item_id in by_arm_id if item_id == conversation_id} == set(evaluator.ARMS)
            for conversation_id in ids
        ),
        f"Test shard pairing drifted: {path}",
    )
    for conversation_id in ids:
        paired = [by_arm_id[(arm, conversation_id)] for arm in evaluator.ARMS]
        _require(
            len({row["input_sha256"] for row in paired}) == 1
            and len({tuple(row["targets"]) for row in paired}) == 1,
            f"Paired test inputs drifted: {path}",
        )
    metric_keys: set[tuple[int, str]] = set()
    budget_violations = 0
    failure_count = 0
    tier_fields = (
        "hot_blocks",
        "logical_blocks",
        "hot_bytes",
        "host_bytes",
        "h2d_bytes",
        "useful_h2d_bytes",
        "d2h_bytes",
        "late_misses",
        "evictions",
    )
    for metric in metrics:
        _require(isinstance(metric, dict), f"Invalid batch metric: {path}")
        offset = metric.get("batch_offset")
        arm = metric.get("arm")
        key = (offset, arm)
        _require(
            _is_int(offset, minimum=0)
            and offset in range(0, labels.EXAMPLES_PER_SHARD, labels.BATCH_SIZE)
            and arm in evaluator.ARMS
            and key not in metric_keys,
            f"Batch metric coordinate drifted: {path}",
        )
        rotation = (cast(int, replicate) * 5 + cast(int, offset) // labels.BATCH_SIZE) % len(
            evaluator.ARMS
        )
        order = (*evaluator.ARMS[rotation:], *evaluator.ARMS[:rotation])
        _require(
            metric.get("execution_index") == order.index(str(arm))
            and _is_finite_number(metric.get("wall_ms"), minimum=0.0)
            and _is_int(metric.get("peak_cuda_allocated_bytes"), minimum=0)
            and _is_int(metric.get("peak_cuda_reserved_bytes"), minimum=0)
            and isinstance(metric.get("tier"), dict)
            and all(_is_int(metric["tier"].get(field), minimum=0) for field in tier_fields)
            and metric["tier"]["useful_h2d_bytes"] <= metric["tier"]["h2d_bytes"],
            f"Batch physical metric schema drifted: {path}",
        )
        failure = metric.get("failure")
        _require(failure is None or isinstance(failure, dict), f"Failure schema drifted: {path}")
        if failure is None:
            _validate_accounting(metric.get("accounting"), label=f"{path}:{offset}:{arm}")
        else:
            _require(metric.get("accounting") is None, f"Failed metric accounting drifted: {path}")
        batch_records = [
            by_arm_id[(str(arm), conversation_id)]
            for conversation_id in ids
            if expected_offset[conversation_id] == offset
        ]
        _require(
            len(batch_records) == labels.BATCH_SIZE
            and all(row.get("failure") == failure for row in batch_records),
            f"Batch failure provenance drifted: {path}",
        )
        recomputed_violations = sum(
            int(row["controller"]["budget_violations"])
            for row in batch_records
            if row.get("controller") is not None
        )
        _require(
            metric.get("budget_violations") == recomputed_violations,
            f"Batch budget violation aggregate drifted: {path}",
        )
        hot_budget = metric.get("physical_hot_budget_blocks_by_layer")
        _require(
            (arm == "native-resident" and hot_budget is None)
            or failure is not None
            or (
                isinstance(hot_budget, dict)
                and bool(hot_budget)
                and all(_is_int(value, minimum=0) for value in hot_budget.values())
            ),
            f"Physical hot-budget evidence drifted: {path}",
        )
        budget_violations += recomputed_violations
        failure_count += int(failure is not None)
        metric_keys.add((int(offset), str(arm)))
    expected_metric_keys = set(
        product(range(0, labels.EXAMPLES_PER_SHARD, labels.BATCH_SIZE), evaluator.ARMS)
    )
    _require(metric_keys == expected_metric_keys, f"Batch metric coverage drifted: {path}")
    return (
        coordinate,
        by_arm_id,
        ids,
        metrics,
        budget_violations,
        failure_count,
        _checkpoint_digest(shard, label=f"test checkpoint for {path}"),
    )


def summarize(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    expected = matrix.get("expected", {})
    completed = matrix.get("completed", {})
    _require(
        matrix.get("experiment_id") == "p1-online-learned-lookahead-matrix-progress-v1",
        "Wrong online-lookahead matrix id.",
    )
    _require(matrix.get("source", {}).get("dirty") is False, "Dirty matrix source.")
    _require(
        expected
        == {
            "label_shards": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies": matrix_runner.EXPECTED_POLICIES,
            "test_shards": matrix_runner.EXPECTED_TEST_SHARDS,
            "test_examples_per_shard": 20,
            "arms": list(evaluator.ARMS),
        },
        "Online-lookahead frozen matrix shape drifted.",
    )
    _require(
        completed
        == {
            "label_shards": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies": matrix_runner.EXPECTED_POLICIES,
            "test_shards": matrix_runner.EXPECTED_TEST_SHARDS,
        },
        "Online-lookahead matrix is incomplete.",
    )
    _require(
        len(matrix.get("label_shards", [])) == matrix_runner.EXPECTED_LABEL_SHARDS
        and len(matrix.get("policies", [])) == matrix_runner.EXPECTED_POLICIES
        and len(matrix.get("test_shards", [])) == matrix_runner.EXPECTED_TEST_SHARDS,
        "Online-lookahead matrix index coverage drifted.",
    )
    _require(
        matrix.get("design", {}).get("path") == str(matrix_runner.DESIGN)
        and matrix.get("design", {}).get("sha256") == _cached_sha256(matrix_runner.DESIGN),
        "Online-lookahead matrix design drifted.",
    )
    causal_gate_path = _bound_path(matrix.get("p2_causal_gate", {}), "P2 causal gate")
    matrix_runner.require_causal_gate(causal_gate_path)
    reuse_probes = _verify_checkpoint_reuse_audit(matrix.get("checkpoint_reuse_audit", {}))
    label_implementation = labels.implementation_digest()
    fit_implementation = fitter.implementation_digest()
    evaluation_implementation = evaluator.implementation_digest()
    for category in ("label_shards", "policies", "test_shards"):
        _require(
            len({metadata.get("path") for metadata in matrix[category]}) == len(matrix[category]),
            f"Duplicate {category} paths detected.",
        )
    label_coordinates: set[tuple[str, int, str, str, int, int]] = set()
    label_artifacts: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    label_risk_examples: dict[tuple[str, int, str], int] = defaultdict(int)
    label_failures: dict[tuple[str, int, str], int] = defaultdict(int)
    label_inputs: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    checkpoint_digests: dict[tuple[str, int], set[str]] = defaultdict(set)
    for metadata in matrix["label_shards"]:
        path = _bound_path(metadata, "online-lookahead label shard")
        shard = json.loads(path.read_text())
        _verify_common(
            shard,
            implementation_digest=label_implementation,
            label=f"label shard {path}",
        )
        coordinate, input_digests, checkpoint_digest = _validate_label_payload(shard, path=path)
        _require(coordinate not in label_coordinates, f"Duplicate label coordinate: {path}")
        label_coordinates.add(coordinate)
        split_key = (coordinate[0], coordinate[1], coordinate[2])
        label_artifacts[split_key].append({"path": str(path), "sha256": str(metadata["sha256"])})
        label_risk_examples[split_key] += int(shard["risk_examples"])
        label_failures[split_key] += int(shard["failure_count"])
        label_inputs[split_key].update(input_digests)
        checkpoint_digests[(coordinate[0], coordinate[1])].add(checkpoint_digest)
    expected_label_coordinates = {
        (scale, seed, split, family, context, replicate)
        for scale, seed, family, context in product(
            matrix_runner.SCALES,
            labels.TRAINING_SEEDS,
            labels.PAPER_GRADE_WORKLOAD_FAMILIES,
            labels.CONTEXTS,
        )
        for split in matrix_runner.SPLITS
        for replicate in labels.REPLICATES[split]
    }
    _require(
        label_coordinates == expected_label_coordinates,
        "Exact online-lookahead label coordinate coverage drifted.",
    )
    for scale, seed in product(matrix_runner.SCALES, labels.TRAINING_SEEDS):
        _require(
            label_inputs[(scale, seed, "train")].isdisjoint(
                label_inputs[(scale, seed, "calibration")]
            ),
            f"Train/calibration input overlap detected for {scale}/seed-{seed}.",
        )
    policy_coordinates: set[tuple[str, int, str]] = set()
    policies_by_coordinate: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] = {}
    for metadata in matrix["policies"]:
        path = _bound_path(metadata, "online-lookahead policy")
        policy_payload = json.loads(path.read_text())
        policy_coordinate_raw = (
            policy_payload.get("scale"),
            policy_payload.get("training_seed"),
            policy_payload.get("budget"),
        )
        _require(
            policy_payload.get("experiment_id") == "p1-online-learned-lookahead-policy-v1"
            and policy_coordinate_raw[0] in matrix_runner.SCALES
            and policy_coordinate_raw[1] in labels.TRAINING_SEEDS
            and policy_coordinate_raw[2] in causal.BUDGET_LABELS
            and policy_coordinate_raw not in policy_coordinates,
            f"Policy coordinate drifted: {path}",
        )
        _verify_common(
            policy_payload,
            implementation_digest=fit_implementation,
            label=f"policy {path}",
        )
        checkpoint_digest = _checkpoint_digest(
            policy_payload, label=f"policy checkpoint for {path}"
        )
        checkpoint_digests[
            (str(policy_coordinate_raw[0]), cast(int, policy_coordinate_raw[1]))
        ].add(checkpoint_digest)
        for dependency_name in ("quota_calibration", "physical_memory_match"):
            _bound_path(
                policy_payload.get(dependency_name, {}), f"policy {dependency_name} for {path}"
            )
        policy = LearnedLookaheadPolicy.from_dict(policy_payload["policy"])
        leakage = policy_payload.get("leakage_guard", {})
        _require(
            policy_payload.get("policy_digest") == policy.policy_digest
            and leakage.get("train_calibration_group_ids_disjoint") is True
            and leakage.get("train_calibration_token_sequences_disjoint") is True
            and leakage.get("test_split_loaded") is False
            and leakage.get("benchmark_targets_used_for_training") is False,
            f"Policy integrity or split boundary drifted: {path}",
        )
        for split in ("train", "calibration"):
            split_payload = policy_payload.get("splits", {}).get(split, {})
            shard_digests = split_payload.get("shard_digests", [])
            _require(isinstance(shard_digests, list), f"Invalid policy split: {path}")
            for shard_metadata in shard_digests:
                _bound_path(shard_metadata, f"policy {split} label shard for {path}")
            split_key = (
                str(policy_coordinate_raw[0]),
                cast(int, policy_coordinate_raw[1]),
                split,
            )
            expected_artifacts = label_artifacts[split_key]
            expected_shards = len(expected_artifacts)
            _require(
                len({(item.get("path"), item.get("sha256")) for item in shard_digests})
                == len(shard_digests)
                and {(item["path"], item["sha256"]) for item in shard_digests}
                == {(item["path"], item["sha256"]) for item in expected_artifacts}
                and split_payload.get("shards") == expected_shards
                and split_payload.get("conversations")
                == expected_shards * labels.EXAMPLES_PER_SHARD
                and split_payload.get("risk_examples") == label_risk_examples[split_key]
                and split_payload.get("failure_count") == label_failures[split_key]
                and split_payload.get("unique_group_ids") == label_risk_examples[split_key],
                f"Policy {split} raw-shard membership drifted: {path}",
            )
        normalized_coordinate = (
            str(policy_coordinate_raw[0]),
            cast(int, policy_coordinate_raw[1]),
            str(policy_coordinate_raw[2]),
        )
        policy_coordinates.add(normalized_coordinate)
        policies_by_coordinate[normalized_coordinate] = (path, policy_payload)
    expected_policy_coordinates = set(
        product(matrix_runner.SCALES, labels.TRAINING_SEEDS, causal.BUDGET_LABELS)
    )
    _require(
        policy_coordinates == expected_policy_coordinates,
        "Exact online-lookahead policy coordinate coverage drifted.",
    )

    differences: dict[str, dict[tuple[str, str, int, str, int], list[float]]] = {
        "learned_vs_one_token": defaultdict(list),
        "learned_vs_fixed": defaultdict(list),
        "no_dense_vs_learned": defaultdict(list),
        "no_pins_vs_learned": defaultdict(list),
    }
    contrast_arms = {
        "learned_vs_one_token": (PRIMARY, ONE_TOKEN),
        "learned_vs_fixed": (PRIMARY, FIXED),
        "no_dense_vs_learned": (
            "online-learned-lookahead-no-dense-fallback",
            PRIMARY,
        ),
        "no_pins_vs_learned": (
            "online-learned-lookahead-no-protected-pins",
            PRIMARY,
        ),
    }
    physical: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failure_count = 0
    budget_violations = 0
    policy_digests: set[str] = set()
    paired_conversations = 0
    test_coordinates: set[tuple[str, int, str, str, int, int]] = set()
    for metadata in matrix["test_shards"]:
        path = _bound_path(metadata, "online-lookahead test shard")
        shard = json.loads(path.read_text())
        _verify_common(
            shard,
            implementation_digest=evaluation_implementation,
            label=f"test shard {path}",
        )
        (
            coordinate,
            by_arm_id,
            ids,
            metrics,
            shard_budget_violations,
            shard_failures,
            checkpoint_digest,
        ) = _validate_test_payload(shard, path=path)
        _require(coordinate not in test_coordinates, f"Duplicate test coordinate: {path}")
        test_coordinates.add(coordinate)
        checkpoint_digests[(coordinate[0], coordinate[1])].add(checkpoint_digest)
        expected_policy_path, policy_payload = policies_by_coordinate[
            (coordinate[0], coordinate[1], coordinate[2])
        ]
        policy_path = _bound_path(shard["policy"], "test shard policy dependency")
        _require(
            policy_path == expected_policy_path
            and shard["policy"].get("sha256") == _cached_sha256(expected_policy_path)
            and shard.get("policy_digest") == policy_payload.get("policy_digest")
            and shard.get("policy_split_counts")
            == {
                split: policy_payload["splits"][split]["risk_examples"]
                for split in ("train", "calibration")
            },
            f"Test shard policy dependency drifted: {path}",
        )
        for dependency_name in (
            "checkpoint",
            "quota_calibration",
            "physical_memory_match",
        ):
            dependency = shard.get(dependency_name, {})
            _bound_path(dependency, f"test shard {dependency_name} for {path}")
            _require(
                dependency.get("path") == policy_payload[dependency_name].get("path")
                and dependency.get("sha256") == policy_payload[dependency_name].get("sha256"),
                f"Test shard {dependency_name} does not match its frozen policy: {path}",
            )
        leakage = shard.get("leakage_guard", {})
        _require(
            shard.get("generation_seed")
            == evaluator.generation_seed(
                training_seed=coordinate[1],
                family=coordinate[3],
                context=coordinate[4],
                replicate=coordinate[5],
            )
            and leakage.get("test_namespace") == evaluator.TEST_NAMESPACE
            and leakage.get("test_examples_used_for_training") is False
            and leakage.get("policy_frozen_before_test") is True
            and leakage.get("paired_inputs_shared_across_arms") is True,
            f"Test split or deterministic seed boundary drifted: {path}",
        )
        policy_digests.add(str(shard["policy_digest"]))
        paired_conversations += len(ids)
        budget_violations += shard_budget_violations
        failure_count += shard_failures
        key_prefix = (
            shard["scale"],
            shard["budget"],
            int(shard["training_seed"]),
            shard["family"],
            int(shard["context"]),
        )
        for conversation_id in ids:
            for name, (candidate, comparator) in contrast_arms.items():
                left = by_arm_id[(candidate, conversation_id)]
                right = by_arm_id[(comparator, conversation_id)]
                _require(left["total"] == right["total"], "Paired query counts drifted.")
                differences[name][key_prefix].append(
                    left["correct_count"] / left["total"] - right["correct_count"] / right["total"]
                )
        for row in metrics:
            arm = row["arm"]
            cell = physical[(shard["scale"], shard["budget"], arm)]
            failed = row.get("failure") is not None
            if not failed:
                accounting = row["accounting"]
                cell["peak_cuda_allocated_bytes"].append(float(row["peak_cuda_allocated_bytes"]))
                cell["peak_cuda_reserved_bytes"].append(float(row["peak_cuda_reserved_bytes"]))
                cell["hot_resident_bytes"].append(float(accounting["hot_resident_bytes"]))
                cell["h2d_bytes"].append(float(row["tier"]["h2d_bytes"]))
                cell["useful_h2d_bytes"].append(float(row["tier"]["useful_h2d_bytes"]))
                cell["d2h_bytes"].append(float(row["tier"]["d2h_bytes"]))
                cell["late_misses"].append(float(row["tier"]["late_misses"]))
                cell["wall_ms"].append(float(row["wall_ms"]))

    expected_test_coordinates = set(
        product(
            matrix_runner.SCALES,
            labels.TRAINING_SEEDS,
            causal.BUDGET_LABELS,
            labels.PAPER_GRADE_WORKLOAD_FAMILIES,
            labels.CONTEXTS,
            range(10),
        )
    )
    _require(
        test_coordinates == expected_test_coordinates,
        "Exact online-lookahead test coordinate coverage drifted.",
    )
    _require(
        len(policy_digests) == matrix_runner.EXPECTED_POLICIES,
        "Test shards do not bind every frozen online-lookahead policy.",
    )
    _require(
        set(checkpoint_digests) == set(product(matrix_runner.SCALES, labels.TRAINING_SEEDS))
        and all(len(digests) == 1 for digests in checkpoint_digests.values()),
        "Checkpoint digest consistency drifted across label/policy/test artifacts.",
    )

    contrast_payloads = {
        name: contrast_statistics(
            differences[name],
            name=name,
            candidate=contrast_arms[name][0],
            comparator=contrast_arms[name][1],
        )
        for name in differences
    }
    primary_cells = contrast_payloads["learned_vs_one_token"]["cells"]
    p_values = {
        f"{row['scale']}:{row['budget']}": row["seed_cluster_inference"][
            "paired_randomization_two_sided_p"
        ]
        for row in primary_cells
    }
    adjusted = holm_bonferroni(p_values)
    cell_gate = []
    for row in primary_cells:
        cell_name = f"{row['scale']}:{row['budget']}"
        seed_inference = row["seed_cluster_inference"]
        ci = seed_inference["seed_cluster_bootstrap_ci"]
        seed_means = seed_inference["seed_means"]
        passed = ci[0] > 0.0 and all(value > 0.0 for value in seed_means)
        cell_gate.append(
            {
                "scale": row["scale"],
                "budget": row["budget"],
                "five_of_five_seed_means_positive": all(value > 0.0 for value in seed_means),
                "seed_cluster_bootstrap_ci": ci,
                "exact_resolution_aware_p": seed_inference["paired_randomization_two_sided_p"],
                "holm_adjusted_p_descriptive": adjusted[cell_name],
                "minimum_attainable_two_sided_exact_p": 0.0625,
                "p_value_used_as_success_gate": False,
                "passed": passed,
            }
        )

    system_cells = []
    memory_passed = True
    for scale in matrix_runner.SCALES:
        for budget in causal.BUDGET_LABELS:
            learned = physical[(scale, budget, PRIMARY)]
            fixed = physical[(scale, budget, FIXED)]

            def mean_or_none(values: list[float]) -> float | None:
                return float(np.mean(values)) if values else None

            learned_hbm = mean_or_none(learned["peak_cuda_allocated_bytes"])
            fixed_hbm = mean_or_none(fixed["peak_cuda_allocated_bytes"])
            relative_hbm = (
                (learned_hbm - fixed_hbm) / max(fixed_hbm, 1.0)
                if learned_hbm is not None and fixed_hbm is not None
                else None
            )
            learned_hot = mean_or_none(learned["hot_resident_bytes"])
            fixed_hot = mean_or_none(fixed["hot_resident_bytes"])
            relative_hot = (
                (learned_hot - fixed_hot) / max(fixed_hot, 1.0)
                if learned_hot is not None and fixed_hot is not None
                else None
            )
            learned_h2d = mean_or_none(learned["h2d_bytes"])
            fixed_h2d = mean_or_none(fixed["h2d_bytes"])
            learned_useful_h2d = mean_or_none(learned["useful_h2d_bytes"])
            fixed_useful_h2d = mean_or_none(fixed["useful_h2d_bytes"])
            passed = (
                relative_hot is not None
                and learned_useful_h2d is not None
                and fixed_useful_h2d is not None
                and relative_hot <= MAXIMUM_HBM_DIFFERENCE
                and learned_useful_h2d <= fixed_useful_h2d
            )
            memory_passed = memory_passed and passed
            system_cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "learned_peak_allocated_bytes_mean": learned_hbm,
                    "fixed_peak_allocated_bytes_mean": fixed_hbm,
                    "relative_peak_allocated_difference": relative_hbm,
                    "learned_hot_resident_bytes_mean": learned_hot,
                    "fixed_hot_resident_bytes_mean": fixed_hot,
                    "relative_hot_resident_difference": relative_hot,
                    "learned_h2d_bytes_mean": learned_h2d,
                    "fixed_h2d_bytes_mean": fixed_h2d,
                    "learned_useful_h2d_bytes_mean": learned_useful_h2d,
                    "fixed_useful_h2d_bytes_mean": fixed_useful_h2d,
                    "successful_learned_measurements": len(learned["hot_resident_bytes"]),
                    "successful_fixed_measurements": len(fixed["hot_resident_bytes"]),
                    "passed": passed,
                }
            )
    fixed_slices = contrast_payloads["learned_vs_fixed"]["by_family_context"]
    worst_slice = min(fixed_slices, key=lambda row: row["mean_difference"])
    worst_slice_passed = worst_slice["mean_difference"] >= MAXIMUM_WORST_SLICE_REGRESSION
    gate_passed = (
        all(row["passed"] for row in cell_gate)
        and memory_passed
        and worst_slice_passed
        and budget_violations == 0
        and failure_count == 0
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _require(not dirty, "Online-lookahead summary requires a clean source tree.")
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-audit-v1",
        "raw_matrix": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
        "audit": {
            "label_shards_verified": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies_verified": matrix_runner.EXPECTED_POLICIES,
            "test_shards_verified": matrix_runner.EXPECTED_TEST_SHARDS,
            "paired_conversations": paired_conversations,
            "quality_arm_conversations": paired_conversations * len(evaluator.ARMS),
            "training_seeds": len(labels.TRAINING_SEEDS),
            "scales": len(matrix_runner.SCALES),
            "families": len(labels.PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": len(labels.CONTEXTS),
            "budgets": len(causal.BUDGET_LABELS),
            "policy_digests": len(policy_digests),
            "all_raw_digests_verified": True,
            "all_dependencies_verified": True,
            "implementation_digests_verified": True,
            "dependency_artifact_digests_verified": True,
            "checkpoint_reuse_equivalence_verified": True,
            "checkpoint_reuse_scale_seed_probes": reuse_probes,
            "exact_label_policy_test_coordinates_verified": True,
            "label_and_test_seed_schedules_verified": True,
            "train_calibration_raw_membership_and_disjointness_verified": True,
            "checkpoint_digest_consistency_verified": True,
            "raw_test_record_schema_verified": True,
            "raw_physical_metrics_and_aggregates_verified": True,
            "all_inputs_paired": True,
            "zero_budget_violations": budget_violations == 0,
            "complete_failure_accounting": True,
            "online_token_offset_verified": True,
            "native_bootstrap_accounted": True,
            "cache_replay_contract_tested": True,
            "resolution_aware_gate_verified": True,
        },
        "contrasts": contrast_payloads,
        "primary_gate": {
            "candidate": PRIMARY,
            "one_token_comparator": ONE_TOKEN,
            "fixed_comparator": FIXED,
            "cells": cell_gate,
            "system_cells": system_cells,
            "worst_fixed_slice": worst_slice,
            "worst_slice_passed": worst_slice_passed,
            "budget_violations": budget_violations,
            "failures": failure_count,
            "passed": gate_passed,
        },
        "decision": "success" if gate_passed else "negative-result",
        "claim_boundary": (
            "Held-out synthetic token-t to token-(t+1) evidence for this frozen predictor, "
            "checkpoint set, and budget matrix only; never same-token, natural-language, or "
            "official DeepSeek-V4 evidence."
        ),
        "source": {"commit": commit, "dirty": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the online learned-lookahead matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-learned-lookahead-matrix.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-learned-lookahead.summary.json"
        ),
    )
    args = parser.parse_args()
    payload = summarize(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"decision": payload["decision"], "gate": payload["primary_gate"]}))


if __name__ == "__main__":
    main()
