from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, cast

import calibrate_p1_layer_quotas as calibration_runner
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_core_shard as core
import summarize_p2_core_matrix as core_summary

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

PRIMARY_TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
PRIMARY_CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
PRIMARY_EVALUATION_SEEDS = (8071401, 8071402, 8071403, 8071404, 8071405)
EXTENSION_TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409)
EXTENSION_CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409)
EXTENSION_EVALUATION_SEEDS = (8071406, 8071407, 8071408, 8071409)
SCALES = ("s55", "s151")
MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-independent-seed-extension-v1.json"
)
CORE_EXPERIMENT_ID = "p2-seed-extension-core-shard-v1"
CORE_MATRIX_EXPERIMENT_ID = "p2-seed-extension-core-matrix-progress-v1"
CORE_AUDIT_EXPERIMENT_ID = "p2-seed-extension-core-matrix-audit-v1"
COMBINED_CORE_AUDIT_EXPERIMENT_ID = "p2-nine-seed-core-matrix-audit-v1"
PRIMARY_CORE_SHARDS = 4_500
EXPECTED_CORE_SHARDS = (
    len(EXTENSION_TRAINING_SEEDS)
    * len(SCALES)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(core.CONTEXTS)
    * len(core.REPLICATES)
)
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4",
    str(MANIFEST_PATH),
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/benchmark_m5_online_controller.py",
    "research/adaptive_v4_memory/scripts/calibrate_p1_layer_quotas.py",
    "research/adaptive_v4_memory/scripts/evaluate_p1_heldout_policy_pilot.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_core_shard.py",
    "research/adaptive_v4_memory/scripts/p2_seed_extension.py",
    "research/adaptive_v4_memory/scripts/run_p2_core_parallel.py",
    "research/adaptive_v4_memory/scripts/run_p2_seed_extension_prerequisites.py",
    "research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py",
    "research/adaptive_v4_memory/scripts/train_m1_associative_recall.py",
    "research/adaptive_v4_memory/scripts/validate_p1_chunked_equivalence.py",
    "research/adaptive_v4_memory/scripts/validate_p1_full_forward_equivalence.py",
)
_PRIMARY_VERIFY_CORE_PAYLOAD = core_summary.verify_raw_shard
_PRIMARY_IMPLEMENTATION_DIGEST = core._implementation_digest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    extension = payload.get("extension_cohort", {})
    primary = payload.get("primary_cohort", {})
    _require(
        payload.get("experiment_id") == "p2-independent-seed-extension-v1"
        and payload.get("status") == "preregistered_before_primary_outcome_inspection"
        and tuple(primary.get("training_seeds", ())) == PRIMARY_TRAINING_SEEDS
        and tuple(primary.get("calibration_seeds", ())) == PRIMARY_CALIBRATION_SEEDS
        and tuple(primary.get("evaluation_seeds", ())) == PRIMARY_EVALUATION_SEEDS
        and tuple(extension.get("training_seeds", ())) == EXTENSION_TRAINING_SEEDS
        and tuple(extension.get("calibration_seeds", ())) == EXTENSION_CALIBRATION_SEEDS
        and tuple(extension.get("evaluation_seeds", ())) == EXTENSION_EVALUATION_SEEDS
        and tuple(extension.get("scales", ())) == SCALES
        and extension.get("families") == len(PAPER_GRADE_WORKLOAD_FAMILIES)
        and extension.get("contexts") == len(core.CONTEXTS)
        and extension.get("replicates_per_context") == len(core.REPLICATES)
        and extension.get("examples_per_shard") == core.EXAMPLES_PER_SHARD
        and extension.get("examples_per_seed_scale_family")
        == len(core.CONTEXTS) * len(core.REPLICATES) * core.EXAMPLES_PER_SHARD
        and extension.get("outcome_dependent_early_stopping") is False,
        "P2 independent-seed extension manifest drifted.",
    )
    _require(
        set(PRIMARY_TRAINING_SEEDS).isdisjoint(EXTENSION_TRAINING_SEEDS)
        and set(PRIMARY_CALIBRATION_SEEDS).isdisjoint(EXTENSION_CALIBRATION_SEEDS)
        and set(PRIMARY_EVALUATION_SEEDS).isdisjoint(EXTENSION_EVALUATION_SEEDS),
        "Primary and extension seed namespaces overlap.",
    )
    return cast(dict[str, Any], payload)


def seed_triplet(training_seed: int) -> tuple[int, int, int]:
    try:
        index = EXTENSION_TRAINING_SEEDS.index(training_seed)
    except ValueError as error:
        raise ValueError(f"Unregistered extension training seed: {training_seed}") from error
    return (
        training_seed,
        EXTENSION_CALIBRATION_SEEDS[index],
        EXTENSION_EVALUATION_SEEDS[index],
    )


@contextmanager
def bind_seed_registry(
    training_seeds: tuple[int, ...] = EXTENSION_TRAINING_SEEDS,
    calibration_seeds: tuple[int, ...] = EXTENSION_CALIBRATION_SEEDS,
    evaluation_seeds: tuple[int, ...] = EXTENSION_EVALUATION_SEEDS,
) -> Iterator[None]:
    """Temporarily expose one explicitly supplied, index-aligned seed registry."""

    _require(
        len(training_seeds) == len(calibration_seeds) == len(evaluation_seeds)
        and len(set(training_seeds)) == len(training_seeds)
        and len(set(calibration_seeds)) == len(calibration_seeds)
        and len(set(evaluation_seeds)) == len(evaluation_seeds),
        "Seed registries must be unique and index aligned.",
    )
    core_module = cast(Any, core)
    calibration_module = cast(Any, calibration_runner)
    heldout_module = cast(Any, heldout)
    old = (
        core_module.TRAINING_SEEDS,
        core_module.EVALUATION_SEEDS,
        calibration_module.CALIBRATION_SEEDS,
        heldout_module.CALIBRATION_SEEDS,
        heldout_module.EVALUATION_SEEDS,
    )
    core_module.TRAINING_SEEDS = training_seeds
    core_module.EVALUATION_SEEDS = evaluation_seeds
    calibration_module.CALIBRATION_SEEDS = calibration_seeds
    heldout_module.CALIBRATION_SEEDS = calibration_seeds
    heldout_module.EVALUATION_SEEDS = evaluation_seeds
    try:
        yield
    finally:
        (
            core_module.TRAINING_SEEDS,
            core_module.EVALUATION_SEEDS,
            calibration_module.CALIBRATION_SEEDS,
            heldout_module.CALIBRATION_SEEDS,
            heldout_module.EVALUATION_SEEDS,
        ) = old


def implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_paths = {
        line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line
    }
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked P2 seed-extension implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def primary_contract_digest() -> str:
    """Digest the frozen implementation shared by primary and extension shards."""

    return _PRIMARY_IMPLEMENTATION_DIGEST()


def source_state() -> dict[str, str | bool]:
    return {
        "commit": head(),
        "dirty": dirty(),
        "implementation_digest": implementation_digest(),
        "base_contract_digest": primary_contract_digest(),
    }


def require_primary_core_audit(matrix_path: Path, audit_path: Path) -> dict[str, Any]:
    """Require the immutable five-seed cohort before extension execution."""

    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    runs = matrix_payload.get("runs", [])
    expected = set(
        product(
            SCALES,
            PRIMARY_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.REPLICATES,
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
        for run in runs
    }
    _require(
        matrix_payload.get("experiment_id") == "p2-core-quality-matrix-progress-v1"
        and matrix_payload.get("completed_shards") == PRIMARY_CORE_SHARDS
        and len(runs) == PRIMARY_CORE_SHARDS
        and observed == expected
        and matrix_payload.get("implementation_digest") == primary_contract_digest(),
        "The immutable five-seed primary core matrix is incomplete or drifted.",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw = audit.get("raw_matrix", {})
    checks = audit.get("audit", {})
    _require(
        audit.get("experiment_id") == "p2-core-quality-matrix-audit-v1"
        and raw.get("path") == str(matrix_path)
        and raw.get("sha256") == sha256(matrix_path)
        and checks.get("all_raw_shards_verified") is True
        and checks.get("all_dependency_digests_verified") is True
        and checks.get("all_record_digests_verified") is True
        and checks.get("no_budget_violations") is True
        and checks.get("exact_record_schema_verified") is True
        and checks.get("exact_execution_rotation_verified") is True
        and checks.get("exact_statistical_cell_coverage_verified") is True
        and checks.get("paired_units_per_seed_scale_family_context") == 200
        and checks.get("paired_units_per_seed_scale_family") == 1_000
        and checks.get("statistical_cells_per_comparison") == 1_350
        and checks.get("unique_shards") == PRIMARY_CORE_SHARDS,
        "The digest-bound five-seed primary core audit is required.",
    )
    return cast(dict[str, Any], audit)


def equivalence(
    path: Path,
    *,
    scale: str,
    training_seed: int,
    checkpoint: Path,
    calibration: Path,
) -> dict[str, Any]:
    with bind_seed_registry():
        payload = core._equivalence(path, scale)
    _, calibration_seed, evaluation_seed = seed_triplet(training_seed)
    pilot_metadata = payload.get("pilot_artifact", {})
    pilot_path = Path(pilot_metadata.get("path", ""))
    _require(
        pilot_path.is_file() and pilot_metadata.get("sha256") == sha256(pilot_path),
        "P2 seed-extension equivalence pilot artifact drifted.",
    )
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    manifest_sha256 = sha256(MANIFEST_PATH)
    _require(
        payload.get("checkpoint", {}).get("path") == str(checkpoint)
        and payload.get("checkpoint", {}).get("sha256") == sha256(checkpoint)
        and payload.get("calibration_artifact", {}).get("path") == str(calibration)
        and payload.get("calibration_artifact", {}).get("sha256") == sha256(calibration)
        and pilot.get("evaluation_seed") == evaluation_seed
        and calibration_payload.get("seed") == calibration_seed
        and payload.get("extension", {}).get("manifest", {}).get("sha256")
        == manifest_sha256
        and pilot.get("extension", {}).get("manifest", {}).get("sha256")
        == manifest_sha256,
        "P2 seed-extension per-checkpoint equivalence drifted.",
    )
    return payload


def extension_metadata() -> dict[str, Any]:
    payload = manifest()
    return {
        "manifest": {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH)},
        "cohort": "four-seed-confirmatory-extension",
        "outcome_dependent_early_stopping": False,
        "pooling_prerequisite": payload["combined_confirmatory_inference"][
            "pooling_prerequisite"
        ],
    }


def build_core_payload(
    model: Any,
    *,
    checkpoint: Path,
    calibration_path: Path,
    calibration_payload: dict[str, Any],
    equivalence_path: Path,
    scale: str,
    family: str,
    context: int,
    replicate: int,
    training_seed: int,
    checkpoint_sha256: str,
    calibration_sha256: str,
    equivalence_sha256: str,
    evaluation_source_state: dict[str, str | bool] | None = None,
) -> dict[str, Any]:
    _, calibration_seed, evaluation_seed = seed_triplet(training_seed)
    equivalence_payload = equivalence(
        equivalence_path,
        scale=scale,
        training_seed=training_seed,
        checkpoint=checkpoint,
        calibration=calibration_path,
    )
    state = evaluation_source_state or source_state()
    _require(state["dirty"] is False, "P2 seed-extension evaluation requires clean source.")
    with bind_seed_registry():
        payload = core.build_payload(
            model,
            checkpoint=checkpoint,
            calibration_path=calibration_path,
            calibration=calibration_payload,
            equivalence_path=equivalence_path,
            equivalence=equivalence_payload,
            scale=scale,
            family=family,
            context=context,
            replicate=replicate,
            training_seed=training_seed,
            batch_size=core.BATCH_SIZE,
            checkpoint_sha256=checkpoint_sha256,
            calibration_sha256=calibration_sha256,
            equivalence_sha256=equivalence_sha256,
            source_state=state,
        )
    payload["experiment_id"] = CORE_EXPERIMENT_ID
    payload["extension"] = extension_metadata()
    payload["seed_registry"] = {
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
    }
    payload["source"]["base_contract_digest"] = primary_contract_digest()
    return payload


def verify_core_payload(
    raw: dict[str, Any], run: dict[str, Any], expected_implementation_digest: str
) -> None:
    _require(raw.get("experiment_id") == CORE_EXPERIMENT_ID, "Wrong extension shard id.")
    training_seed, calibration_seed, evaluation_seed = seed_triplet(raw["training_seed"])
    extension = raw.get("extension", {})
    manifest_metadata = extension.get("manifest", {})
    _require(
        manifest_metadata.get("path") == str(MANIFEST_PATH)
        and manifest_metadata.get("sha256") == sha256(MANIFEST_PATH)
        and extension.get("cohort") == "four-seed-confirmatory-extension"
        and extension.get("outcome_dependent_early_stopping") is False
        and raw.get("source", {}).get("base_contract_digest") == primary_contract_digest()
        and raw.get("seed_registry")
        == {
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "evaluation_seed": evaluation_seed,
        },
        "P2 seed-extension provenance drifted.",
    )
    equivalence(
        Path(raw["equivalence_artifact"]["path"]),
        scale=raw["scale"],
        training_seed=training_seed,
        checkpoint=Path(raw["checkpoint"]["path"]),
        calibration=Path(raw["calibration_artifact"]["path"]),
    )
    primary_view = {**raw, "experiment_id": "p2-core-quality-shard-v1"}
    with bind_seed_registry():
        _PRIMARY_VERIFY_CORE_PAYLOAD(primary_view, run, expected_implementation_digest)


def core_output_path(
    root: Path, scale: str, training_seed: int, family: str, context: int, replicate: int
) -> Path:
    return (
        root
        / scale
        / f"seed-{training_seed}"
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def expected_core_coordinates() -> set[tuple[str, int, str, int, int]]:
    return set(
        product(
            SCALES,
            EXTENSION_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.REPLICATES,
        )
    )


def write_core_matrix(root: Path, output: Path) -> dict[str, Any]:
    digest = implementation_digest()
    runs: list[dict[str, Any]] = []
    for scale, seed, family, context, replicate in sorted(expected_core_coordinates()):
        path = core_output_path(root, scale, seed, family, context, replicate)
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        run = {
            "scale": scale,
            "training_seed": seed,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        verify_core_payload(raw, run, digest)
        runs.append(
            {
                **run,
                "evaluation_seed": raw["evaluation_seed"],
                "raw_artifact": {"path": str(path), "sha256": sha256(path)},
                "records_digest": raw["records_digest"],
                "wall_seconds": raw["wall_seconds"],
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": CORE_MATRIX_EXPERIMENT_ID,
        "source": {"commit": head(), "dirty": False},
        "implementation_digest": digest,
        "base_contract_digest": primary_contract_digest(),
        "extension_manifest": {"path": str(MANIFEST_PATH), "sha256": sha256(MANIFEST_PATH)},
        "frozen_design": {
            "scales": SCALES,
            "training_seeds": EXTENSION_TRAINING_SEEDS,
            "calibration_seeds": EXTENSION_CALIBRATION_SEEDS,
            "evaluation_seeds": EXTENSION_EVALUATION_SEEDS,
            "families": PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": core.CONTEXTS,
            "replicates": core.REPLICATES,
            "examples_per_shard": core.EXAMPLES_PER_SHARD,
            "batch_size": core.BATCH_SIZE,
            "examples_per_family_checkpoint": (
                len(core.CONTEXTS) * len(core.REPLICATES) * core.EXAMPLES_PER_SHARD
            ),
            "core_policies": core.CORE_POLICIES,
            "chunk_size_by_scale": core.CHUNK_SIZE_BY_SCALE,
            "total_expected_shards": EXPECTED_CORE_SHARDS,
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    atomic_json(output, payload)
    return payload
