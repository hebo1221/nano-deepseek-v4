from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p2-707-continuous-layer-rank-enrichment-v1"
FULL_FORWARD_EXPERIMENT_ID = "p2-707-full-forward-continuous-rank-cell-v1"
EXACT_PATH_EXPERIMENT_ID = "p2-707-exact-path-continuous-rank-cell-v1"
FULL_FORWARD_AUDIT_ID = "p2-707-full-forward-continuous-rank-audit-v1"
TERMINAL_AUDIT_ID = "p2-707-continuous-layer-rank-terminal-audit-v1"

MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-707-continuous-layer-rank-enrichment-v1.json"
)
PARENT_MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-localization-stage-a-execution-v1.json"
)
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_707_continuous_layer_rank")
TRAINING_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/training")
CALIBRATION_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix")

SCALES = ("s55", "s151")
TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
BUDGETS = ("2x", "4x")
CONTEXTS = (80, 128, 256, 512, 1024)
FAMILIES = (
    "single-remote-retrieval",
    "multiple-independent-needles",
    "associative-recall",
    "multi-turn-query-shift",
    "dense-global-aggregation",
    "irrelevant-context-local-only",
    "instruction-persistence",
    "adversarial-lexical-distractors",
    "long-generation-changing-evidence",
)
EXAMPLES_PER_FAMILY = 256
BATCH_SIZE = 4
FULL_FORWARD_CONVERSATIONS_PER_CELL = len(FAMILIES) * EXAMPLES_PER_FAMILY
FULL_FORWARD_BATCH_RUNS_PER_CELL = FULL_FORWARD_CONVERSATIONS_PER_CELL // BATCH_SIZE
FULL_FORWARD_QUERIES_PER_LAYER = 7_168
EXACT_PATH_BATCHES_PER_FAMILY_CONTEXT = 1
EXTRACTION_ORDERS = ("forward", "reverse")
BOOTSTRAP_SEED_BASE = 7_171_901
EXACT_PATH_UNIQUE_CONVERSATIONS_PER_CELL = len(FAMILIES) * len(CONTEXTS) * BATCH_SIZE
EXACT_PATH_QUERIES_PER_LAYER = 560
EXACT_PATH_BATCH_RUNS_PER_CELL = (
    len(FAMILIES) * len(CONTEXTS) * len(BUDGETS) * len(EXTRACTION_ORDERS)
)
EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL = EXACT_PATH_BATCH_RUNS_PER_CELL * BATCH_SIZE

IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/benchmark_m5_online_controller.py",
    "research/adaptive_v4_memory/scripts/calibrate_p1_layer_quotas.py",
    "research/adaptive_v4_memory/scripts/evaluate_p1_heldout_policy_pilot.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_targeted_stage_a_shard.py",
    "research/adaptive_v4_memory/scripts/freeze_p2_causal_factorial_arms.py",
    "research/adaptive_v4_memory/scripts/p2_continuous_rank_contract.py",
    "research/adaptive_v4_memory/scripts/p2_continuous_layer_rank.py",
    "research/adaptive_v4_memory/scripts/collect_p2_full_forward_continuous_rank.py",
    "research/adaptive_v4_memory/scripts/collect_p2_exact_path_layer_signals.py",
    "research/adaptive_v4_memory/scripts/run_p2_continuous_rank_matrix.py",
    "research/adaptive_v4_memory/scripts/audit_p2_continuous_rank.py",
)

_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "target",
        "targets",
        "label",
        "labels",
        "answer",
        "answers",
        "prediction",
        "predictions",
        "correct",
        "correctness",
        "accuracy",
        "logit",
        "logits",
        "input_ids",
        "token_ids",
        "raw_text",
    }
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_oid(value: object) -> bool:
    """Accept full object IDs from SHA-1 or SHA-256 Git repositories."""

    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def payload_digest(payload: dict[str, Any]) -> str:
    return json_digest({key: value for key, value in payload.items() if key != "payload_sha256"})


def validate_payload_digest(payload: dict[str, Any]) -> None:
    if payload.get("payload_sha256") != payload_digest(payload):
        raise ValueError("Payload SHA-256 is missing or invalid.")


def reject_supervision_fields(value: Any, *, path: str = "$") -> None:
    """Fail closed if a calibration artifact contains supervision or raw-token fields."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if any(
                normalized == forbidden or normalized.startswith(f"{forbidden}_")
                for forbidden in _FORBIDDEN_OUTPUT_KEYS
            ):
                raise ValueError(f"Forbidden calibration output field at {path}.{key}.")
            reject_supervision_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_supervision_fields(item, path=f"{path}[{index}]")


def source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def implementation_file_digests() -> dict[str, str]:
    missing = [path for path in IMPLEMENTATION_PATHS if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Continuous-rank implementation files are missing: {missing}")
    tracked_output = subprocess.run(
        ["git", "ls-files", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_set = set(tracked_output)
    untracked = [
        path
        for path in IMPLEMENTATION_PATHS
        if (
            (Path(path).is_file() and path not in tracked_set)
            or (
                Path(path).is_dir()
                and not any(
                    candidate.startswith(f"{path.rstrip('/')}/") for candidate in tracked_set
                )
            )
        )
    ]
    if untracked:
        raise RuntimeError(
            f"Continuous-rank execution requires tracked implementation inputs: {untracked}"
        )
    result: dict[str, str] = {}
    for path in IMPLEMENTATION_PATHS:
        candidate = Path(path)
        if candidate.is_file():
            result[path] = sha256(candidate)
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "--", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if not tracked:
            raise RuntimeError(f"Tracked implementation tree is empty: {path}")
        result[path] = json_digest(
            {tracked_path: sha256(Path(tracked_path)) for tracked_path in sorted(tracked)}
        )
    return result


def implementation_digest(files: dict[str, str] | None = None) -> str:
    return json_digest(files if files is not None else implementation_file_digests())


def calibration_path(scale: str, training_seed: int) -> Path:
    return CALIBRATION_ROOT / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"


def checkpoint_path(scale: str, training_seed: int) -> Path:
    return TRAINING_ROOT / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"


def cell_key(scale: str, training_seed: int) -> str:
    if scale not in SCALES or training_seed not in TRAINING_SEEDS:
        raise ValueError("Unknown continuous-rank seed-scale cell.")
    return f"{scale}/seed-{training_seed}"


def expected_cell_keys() -> tuple[str, ...]:
    return tuple(cell_key(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS)


def bootstrap_seed(scale: str, training_seed: int, budget: str) -> int:
    """Return the frozen cell-budget seed for calibration-only resampling."""

    if scale not in SCALES or training_seed not in TRAINING_SEEDS or budget not in BUDGETS:
        raise ValueError("Unknown continuous-rank bootstrap cell.")
    return (
        BOOTSTRAP_SEED_BASE
        + 100 * SCALES.index(scale)
        + 10 * TRAINING_SEEDS.index(training_seed)
        + BUDGETS.index(budget)
    )


def full_forward_output_path(scale: str, training_seed: int) -> Path:
    return OUTPUT_ROOT / "full_forward" / scale / f"seed-{training_seed}" / "cell.json"


def exact_path_output_path(scale: str, training_seed: int) -> Path:
    return OUTPUT_ROOT / "exact_path" / scale / f"seed-{training_seed}" / "cell.json"


def full_forward_audit_path() -> Path:
    return OUTPUT_ROOT / "audits" / "full-forward.audit.json"


def terminal_audit_path() -> Path:
    return OUTPUT_ROOT / "audits" / "terminal.audit.json"


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    resolved_root = OUTPUT_ROOT.resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise ValueError("Continuous-rank output must remain under its frozen output root.")
    reject_supervision_fields(payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Continuous-rank artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_manifest(path: Path = MANIFEST_PATH, *, verify_files: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(payload.get("schema_version") == 1, "Continuous-rank manifest schema drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong continuous-rank manifest.")
    _require(
        payload.get("status")
        == "frozen_before_any_new_fp32_score_extraction_or_continuous_rank_result",
        "Continuous-rank manifest is not in its frozen pre-result state.",
    )
    _require(payload.get("quality_execution_permitted") is False, "Quality must remain blocked.")
    parent = payload.get("parent_contract", {})
    _require(parent.get("path") == str(PARENT_MANIFEST_PATH), "Parent contract path drifted.")
    _require(PARENT_MANIFEST_PATH.is_file(), "Parent contract is missing.")
    _require(parent.get("sha256") == sha256(PARENT_MANIFEST_PATH), "Parent contract drifted.")

    grid = payload.get("grid", {})
    _require(tuple(grid.get("scales", ())) == SCALES, "Scale grid drifted.")
    _require(tuple(grid.get("training_seeds", ())) == TRAINING_SEEDS, "Training seeds drifted.")
    _require(
        tuple(grid.get("calibration_seeds", ())) == CALIBRATION_SEEDS,
        "Calibration seeds drifted.",
    )
    _require(tuple(grid.get("budgets", ())) == BUDGETS, "Budget grid drifted.")
    _require(tuple(grid.get("contexts", ())) == CONTEXTS, "Context grid drifted.")
    _require(tuple(grid.get("families", ())) == FAMILIES, "Workload-family grid drifted.")
    _require(
        grid.get("examples_per_family") == EXAMPLES_PER_FAMILY,
        "Full-forward example count drifted.",
    )
    _require(grid.get("batch_size") == BATCH_SIZE, "Calibration batch size drifted.")
    _require(
        grid.get("full_forward_conversations_per_cell") == FULL_FORWARD_CONVERSATIONS_PER_CELL,
        "Full-forward conversation count drifted.",
    )
    _require(
        grid.get("full_forward_batch_runs_per_cell") == FULL_FORWARD_BATCH_RUNS_PER_CELL,
        "Full-forward batch-run count drifted.",
    )
    _require(
        grid.get("full_forward_queries_per_layer") == FULL_FORWARD_QUERIES_PER_LAYER,
        "Full-forward query count drifted.",
    )
    _require(
        grid.get("exact_path_batches_per_family_context") == EXACT_PATH_BATCHES_PER_FAMILY_CONTEXT,
        "Exact-path batch count drifted.",
    )
    _require(
        tuple(grid.get("exact_path_extraction_orders", ())) == EXTRACTION_ORDERS,
        "Exact-path extraction orders drifted.",
    )
    _require(
        grid.get("exact_path_unique_conversations_per_cell")
        == EXACT_PATH_UNIQUE_CONVERSATIONS_PER_CELL,
        "Exact-path unique-conversation count drifted.",
    )
    _require(
        grid.get("exact_path_batch_runs_per_cell") == EXACT_PATH_BATCH_RUNS_PER_CELL,
        "Exact-path batch-run count drifted.",
    )
    _require(
        grid.get("exact_path_executed_conversations_per_cell")
        == EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL,
        "Exact-path executed-conversation count drifted.",
    )
    _require(
        grid.get("exact_path_queries_per_layer_per_budget_order") == EXACT_PATH_QUERIES_PER_LAYER,
        "Exact-path query count drifted.",
    )

    expected_inputs = set(expected_cell_keys())
    inputs = payload.get("input_inventory", {})
    _require(set(inputs) == expected_inputs, "Input inventory is incomplete.")
    for scale in SCALES:
        for index, training_seed in enumerate(TRAINING_SEEDS):
            item = inputs[cell_key(scale, training_seed)]
            _require(
                item.get("calibration_seed") == CALIBRATION_SEEDS[index],
                "Training-to-calibration seed mapping drifted.",
            )
            for label, expected_path in (
                ("checkpoint", checkpoint_path(scale, training_seed)),
                ("prior_calibration", calibration_path(scale, training_seed)),
            ):
                metadata = item.get(label, {})
                _require(metadata.get("path") == str(expected_path), f"{label} path drifted.")
                _require(expected_path.is_file(), f"{label} input is missing.")
                _require(
                    metadata.get("bytes") == expected_path.stat().st_size, f"{label} size drifted."
                )
                _require(metadata.get("sha256") == sha256(expected_path), f"{label} SHA drifted.")

    implementation = payload.get("implementation", {})
    frozen_files = implementation.get("files", {})
    _require(set(frozen_files) == set(IMPLEMENTATION_PATHS), "Implementation inventory drifted.")
    frozen_commit = implementation.get("source_commit")
    _require(is_git_oid(frozen_commit), "Frozen implementation commit is invalid.")
    _require(
        implementation.get("digest") == implementation_digest(frozen_files),
        "Frozen implementation digest is invalid.",
    )
    if verify_files:
        _require(
            frozen_files == implementation_file_digests(),
            "Checked-out implementation differs from the frozen manifest.",
        )
        state = source_state()
        _require(state["dirty"] is False, "Continuous-rank execution requires clean source.")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(frozen_commit), str(state["commit"])],
            capture_output=True,
        )
        _require(
            ancestry.returncode == 0,
            "Checked-out source does not descend from the frozen implementation commit.",
        )
    return payload


def manifest_binding(path: Path = MANIFEST_PATH) -> dict[str, str]:
    manifest = load_manifest(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "experiment_id": str(manifest["experiment_id"]),
        "implementation_digest": str(manifest["implementation"]["digest"]),
    }


def input_binding(manifest: dict[str, Any], scale: str, training_seed: int) -> dict[str, Any]:
    return dict(manifest["input_inventory"][cell_key(scale, training_seed)])
