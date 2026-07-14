from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, cast

import evaluate_p2_causal_factorial_shard as causal
import p2_seed_extension as extension
import summarize_p2_causal_factorial as causal_summary

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

DESIGN_PATH = Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json")
MEMORY_RAW_EXPERIMENT_ID = "p2-seed-extension-causal-hot-memory-match-v1"
MEMORY_AUDIT_EXPERIMENT_ID = "p2-seed-extension-causal-hot-memory-match-audit-v1"
EQUIVALENCE_RAW_EXPERIMENT_ID = "p2-seed-extension-causal-equivalence-v1"
EQUIVALENCE_AUDIT_EXPERIMENT_ID = "p2-seed-extension-causal-equivalence-audit-v1"
SHARD_EXPERIMENT_ID = "p2-seed-extension-causal-factorial-shard-v1"
MATRIX_EXPERIMENT_ID = "p2-seed-extension-causal-factorial-matrix-v1"
AUDIT_EXPERIMENT_ID = "p2-seed-extension-causal-ablation-audit-v1"
COMBINED_AUDIT_EXPERIMENT_ID = "p2-nine-seed-causal-ablation-audit-v1"
PRIMARY_EXPECTED_SHARDS = 9_000
EXPECTED_SHARDS = (
    len(extension.SCALES)
    * len(extension.EXTENSION_TRAINING_SEEDS)
    * len(causal.BUDGET_LABELS)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(causal.CONTEXTS)
    * len(causal.REPLICATES)
)
PRIMARY_IMPLEMENTATION_PATHS = tuple(causal.IMPLEMENTATION_PATHS)
IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *PRIMARY_IMPLEMENTATION_PATHS,
            str(extension.MANIFEST_PATH),
            "research/adaptive_v4_memory/scripts/p2_seed_extension.py",
            "research/adaptive_v4_memory/scripts/p2_seed_extension_causal.py",
            "research/adaptive_v4_memory/scripts/run_p2_seed_extension_prerequisites.py",
            "research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py",
            "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py",
            "research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal_prerequisites.py",
            "research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py",
        )
    )
)
_PRIMARY_VERIFY_RAW_METADATA = causal_summary.verify_raw_metadata
_PRIMARY_MEMORY_MATCH = causal._memory_match
_PRIMARY_EQUIVALENCE = causal._equivalence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest_paths(paths: tuple[str, ...]) -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_paths = {
        line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line
    }
    missing = [
        path
        for path in paths
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked causal extension implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def primary_contract_digest() -> str:
    return _digest_paths(PRIMARY_IMPLEMENTATION_PATHS)


def implementation_digest() -> str:
    return _digest_paths(IMPLEMENTATION_PATHS)


@contextmanager
def bind_registry(
    training_seeds: tuple[int, ...] = extension.EXTENSION_TRAINING_SEEDS,
    calibration_seeds: tuple[int, ...] = extension.EXTENSION_CALIBRATION_SEEDS,
    evaluation_seeds: tuple[int, ...] = extension.EXTENSION_EVALUATION_SEEDS,
) -> Iterator[None]:
    causal_module = cast(Any, causal)
    old = (
        causal_module.TRAINING_SEEDS,
        causal_module.EVALUATION_SEEDS,
        causal_module.IMPLEMENTATION_PATHS,
    )
    with extension.bind_seed_registry(
        training_seeds, calibration_seeds, evaluation_seeds
    ):
        causal_module.TRAINING_SEEDS = training_seeds
        causal_module.EVALUATION_SEEDS = evaluation_seeds
        causal_module.IMPLEMENTATION_PATHS = IMPLEMENTATION_PATHS
        try:
            yield
        finally:
            (
                causal_module.TRAINING_SEEDS,
                causal_module.EVALUATION_SEEDS,
                causal_module.IMPLEMENTATION_PATHS,
            ) = old


def source_state() -> dict[str, str | bool]:
    return {
        "commit": extension.head(),
        "dirty": extension.dirty(),
        "implementation_digest": implementation_digest(),
        "base_contract_digest": primary_contract_digest(),
    }


def extension_metadata() -> dict[str, Any]:
    return {
        **extension.extension_metadata(),
        "causal_design": {"path": str(DESIGN_PATH), "sha256": extension.sha256(DESIGN_PATH)},
    }


def require_extension_core_audit(matrix_path: Path, audit_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    expected = set(
        product(
            extension.SCALES,
            extension.EXTENSION_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            causal.CONTEXTS,
            causal.REPLICATES,
        )
    )
    observed = {
        (
            run.get("scale"),
            run.get("training_seed"),
            run.get("family"),
            run.get("context"),
            run.get("replicate"),
        )
        for run in matrix.get("runs", [])
    }
    _require(
        matrix.get("experiment_id") == extension.CORE_MATRIX_EXPERIMENT_ID
        and matrix.get("completed_shards") == extension.EXPECTED_CORE_SHARDS
        and observed == expected
        and matrix.get("implementation_digest") == extension.implementation_digest(),
        "The complete four-seed extension core matrix is required before causality.",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = audit.get("audit", {})
    raw = audit.get("raw_matrix", {})
    _require(
        audit.get("experiment_id") == extension.CORE_AUDIT_EXPERIMENT_ID
        and raw.get("path") == str(matrix_path)
        and raw.get("sha256") == extension.sha256(matrix_path)
        and checks.get("all_raw_shards_verified") is True
        and checks.get("all_dependency_digests_verified") is True
        and checks.get("all_record_digests_verified") is True
        and checks.get("no_budget_violations") is True
        and checks.get("unique_shards") == extension.EXPECTED_CORE_SHARDS,
        "The full digest-bound extension core audit is required before causality.",
    )
    return cast(dict[str, Any], audit)


def require_primary_causal_audit(matrix_path: Path, audit_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    expected = set(
        product(
            extension.SCALES,
            extension.PRIMARY_TRAINING_SEEDS,
            causal.BUDGET_LABELS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            causal.CONTEXTS,
            causal.REPLICATES,
        )
    )
    observed = {
        (
            run.get("scale"),
            run.get("training_seed"),
            run.get("budget"),
            run.get("family"),
            run.get("context"),
            run.get("replicate"),
        )
        for run in matrix.get("runs", [])
    }
    _require(
        matrix.get("experiment_id") == "p2-causal-factorial-matrix-progress-v1"
        and matrix.get("completed_shards") == PRIMARY_EXPECTED_SHARDS
        and observed == expected
        and matrix.get("implementation_digest") == primary_contract_digest(),
        "The complete five-seed primary causal matrix is required.",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = audit.get("audit", {})
    raw = audit.get("raw_matrix", {})
    _require(
        audit.get("experiment_id") == "p2-causal-ablation-audit-v1"
        and raw.get("path") == str(matrix_path)
        and raw.get("sha256") == extension.sha256(matrix_path)
        and checks.get("all_raw_shards_verified") is True
        and checks.get("all_dependency_digests_verified") is True
        and checks.get("all_record_digests_verified") is True
        and checks.get("no_budget_violations") is True
        and checks.get("all_physical_predictions_identical") is True
        and checks.get("exact_config_reuse_verified") is True
        and checks.get("unique_shards") == PRIMARY_EXPECTED_SHARDS,
        "The full digest-bound primary causal audit is required.",
    )
    return cast(dict[str, Any], audit)


def _dependency(metadata: dict[str, Any], name: str) -> Path:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {name}: {path}")
    _require(metadata.get("sha256") == extension.sha256(path), f"Drifted {name}: {path}")
    return path


def memory_match(
    path: Path, *, scale: str, training_seed: int, calibration_path: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_path = _dependency(payload.get("raw_artifact", {}), "extension memory-match raw")
    matches = payload.get("matches", {})
    calibration_metadata = payload.get("calibration_artifact", {})
    _require(
        payload.get("experiment_id") == MEMORY_AUDIT_EXPERIMENT_ID
        and payload.get("scale") == scale
        and payload.get("training_seed") == training_seed
        and payload.get("implementation_digest") == implementation_digest()
        and payload.get("all_budget_cells_matched") is True
        and set(matches) == set(causal.BUDGET_LABELS)
        and all(
            item.get("passed") is True
            and item.get("relative_difference", 1.0) <= 0.01
            and item.get("mixture_denominator") == causal.MEMORY_MATCH_MIXTURE_DENOMINATOR
            for item in matches.values()
        )
        and calibration_metadata.get("path") == str(calibration_path)
        and calibration_metadata.get("sha256") == extension.sha256(calibration_path),
        "Extension physical hot-memory match drifted.",
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    protocol = raw.get("protocol", {})
    _require(
        raw.get("experiment_id") == MEMORY_RAW_EXPERIMENT_ID
        and raw.get("scale") == scale
        and raw.get("training_seed") == training_seed
        and raw.get("source", {}).get("dirty") is False
        and raw.get("source", {}).get("implementation_digest") == implementation_digest()
        and raw.get("matches") == matches
        and raw.get("calibration_artifact") == calibration_metadata
        and protocol.get("predictions_captured") is False
        and protocol.get("quality_targets_used_for_matching") is False
        and protocol.get("examples_per_family_context")
        == causal.MEMORY_MATCH_EXAMPLES_PER_FAMILY_CONTEXT
        and protocol.get("heldout_mixture_denominator")
        == causal.MEMORY_MATCH_MIXTURE_DENOMINATOR
        and tuple(protocol.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES
        and tuple(protocol.get("contexts", ())) == causal.CONTEXTS,
        "Extension physical hot-memory raw audit failed.",
    )
    return cast(dict[str, Any], payload)


def equivalence(path: Path, *, scale: str, training_seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_path = _dependency(payload.get("raw_artifact", {}), "extension causal equivalence raw")
    validation = payload.get("validation", {})
    _require(
        payload.get("experiment_id") == EQUIVALENCE_AUDIT_EXPERIMENT_ID
        and payload.get("scale") == scale
        and payload.get("training_seed") == training_seed
        and payload.get("implementation_digest") == implementation_digest()
        and tuple(validation.get("budgets", ())) == causal.BUDGET_LABELS
        and tuple(validation.get("arms", ())) == causal.ALL_ARM_NAMES
        and validation.get("chunk_size") == causal.CHUNK_SIZE_BY_SCALE[scale]
        and validation.get("examples_per_family_context")
        == causal.EQUIVALENCE_EXAMPLES_PER_FAMILY_CONTEXT
        and validation.get("expected_records") == causal.EXPECTED_EQUIVALENCE_RECORDS
        and validation.get("observed_records") == causal.EXPECTED_EQUIVALENCE_RECORDS
        and validation.get("all_predictions_identical") is True
        and validation.get("all_budget_checks_passed") is True
        and validation.get("path_orders_both_observed") is True,
        "Extension causal sequential/chunked equivalence drifted.",
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _require(
        raw.get("experiment_id") == EQUIVALENCE_RAW_EXPERIMENT_ID
        and raw.get("scale") == scale
        and raw.get("training_seed") == training_seed
        and raw.get("source", {}).get("dirty") is False
        and raw.get("source", {}).get("implementation_digest") == implementation_digest()
        and raw.get("validation") == validation
        and raw.get("records_digest") == causal.records_digest(raw.get("records", [])),
        "Extension causal equivalence raw audit failed.",
    )
    return cast(dict[str, Any], payload)


def output_path(
    root: Path,
    scale: str,
    seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
) -> Path:
    return (
        root
        / scale
        / f"seed-{seed}"
        / f"budget-{budget}"
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def expected_coordinates() -> set[tuple[str, int, str, str, int, int]]:
    return set(
        product(
            extension.SCALES,
            extension.EXTENSION_TRAINING_SEEDS,
            causal.BUDGET_LABELS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            causal.CONTEXTS,
            causal.REPLICATES,
        )
    )


def verify_shard(
    raw: dict[str, Any], run: dict[str, Any], expected_implementation_digest: str
) -> None:
    _require(
        raw.get("experiment_id") == SHARD_EXPERIMENT_ID,
        "Wrong causal extension shard id.",
    )
    seed, calibration_seed, evaluation_seed = extension.seed_triplet(raw["training_seed"])
    metadata = raw.get("extension", {})
    _require(
        metadata.get("manifest", {}).get("sha256") == extension.sha256(extension.MANIFEST_PATH)
        and metadata.get("outcome_dependent_early_stopping") is False
        and raw.get("source", {}).get("base_contract_digest") == primary_contract_digest()
        and raw.get("seed_registry")
        == {
            "training_seed": seed,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
        },
        "Causal extension provenance drifted.",
    )
    calibration_path = _dependency(raw.get("calibration_artifact", {}), "calibration")
    _dependency(raw.get("checkpoint", {}), "checkpoint")
    memory_path = _dependency(raw.get("memory_match_artifact", {}), "memory match")
    equivalence_path = _dependency(raw.get("equivalence_artifact", {}), "equivalence")
    _dependency(raw.get("design_manifest", {}), "causal design")
    memory_match(
        memory_path,
        scale=raw["scale"],
        training_seed=seed,
        calibration_path=calibration_path,
    )
    equivalence(equivalence_path, scale=raw["scale"], training_seed=seed)
    primary_view = {**raw, "experiment_id": "p2-causal-factorial-shard-v1"}
    with bind_registry():
        _PRIMARY_VERIFY_RAW_METADATA(primary_view, run, expected_implementation_digest)
    records = raw.get("records", [])
    metrics = raw.get("batch_metrics", [])
    physical = raw.get("physical_measurements", [])
    _require(
        isinstance(records, list)
        and len(records) == causal.EXAMPLES_PER_SHARD * len(causal.ALL_ARM_NAMES)
        and causal.records_digest(records) == raw.get("records_digest")
        and isinstance(metrics, list)
        and len(metrics) == causal.BATCHES_PER_SHARD * len(causal.ALL_ARM_NAMES)
        and all(item.get("budget_violations") == 0 for item in metrics)
        and isinstance(physical, list)
        and len(physical) == causal.BATCHES_PER_SHARD * len(causal.PHYSICAL_ARM_NAMES)
        and all(item.get("predictions_identical_to_chunked") is True for item in physical),
        "Causal extension shard evidence drifted.",
    )
