from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verify_p3_natural_model import sha256

EXPECTED_P2_SHARDS = 4_500
EXPECTED_SCALES = ("s55", "s151")
EXPECTED_TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
EXPECTED_BUDGETS = ("2x", "4x")
EXPECTED_CAUSAL_SHARDS = 9_000
EXPECTED_CONFIRMATORY_CAUSAL_SHARDS = 16_200
EXPECTED_CONFIRMATORY_SEEDS = 9
DEFAULT_PRIMARY_CORE_AUDIT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.strict.summary.json"
)
DEFAULT_NINE_SEED_CORE = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-core.summary.json"
)
DEFAULT_NINE_SEED_CAUSAL = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"
)
REQUIRED_CAUSAL_TRUE_AUDITS = (
    "no_budget_violations",
    "exact_config_reuse_verified",
    "exact_seed_randomization_verified",
    "physical_controller_budget_verified",
    "exact_statistical_cell_coverage_verified",
    "family_holm_bonferroni_verified",
    "contrast_holm_bonferroni_verified",
    "primary_four_cell_bonferroni_verified",
    "required_scale_seed_completion_verified",
)
REQUIRED_CORE_TRUE_AUDITS = (
    "all_raw_shards_verified",
    "all_dependency_digests_verified",
    "all_record_digests_verified",
    "no_budget_violations",
    "held_out_seed_contract_verified",
    "paired_conversation_coverage_verified",
    "execution_order_coverage_verified",
    "exact_record_schema_verified",
    "exact_execution_rotation_verified",
    "exact_statistical_cell_coverage_verified",
    "aggregate_recomputed",
    "batch_coverage_verified",
    "exact_seed_randomization_verified",
    "family_holm_bonferroni_verified",
)


def _load(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{name} is not available yet: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return payload


def _causal_cell_is_consistent(cell: Any) -> bool:
    if not isinstance(cell, dict):
        return False
    checks = (
        cell.get("pooled_effect_positive"),
        cell.get("four_cell_corrected_lower_bound_positive"),
        cell.get("all_seed_effects_positive"),
        cell.get("all_seed_memory_cells_within_one_percent"),
    )
    return all(type(value) is bool for value in checks) and cell.get("passed") is all(
        checks
    )


def _bound_raw_matrix(summary: dict[str, Any], label: str) -> Path:
    metadata = summary.get("raw_matrix", {})
    path = Path(metadata.get("path", ""))
    if not path.is_file() or metadata.get("sha256") != sha256(path):
        raise RuntimeError(f"{label} raw-matrix binding is missing or drifted.")
    return path


def require_core_audits(
    p2_matrix: Path,
    primary_core_audit: Path,
    nine_seed_core: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = _load(primary_core_audit, "primary P2 core audit")
    primary_audit = primary.get("audit", {})
    primary_matrix = _bound_raw_matrix(primary, "Primary P2 core audit")
    if (
        primary.get("experiment_id") != "p2-core-quality-matrix-audit-v1"
        or primary.get("source", {}).get("dirty") is not False
        or primary_matrix.resolve() != p2_matrix.resolve()
        or primary_audit.get("unique_shards") != EXPECTED_P2_SHARDS
        or primary_audit.get("independent_seed_clusters_per_cell")
        != len(EXPECTED_TRAINING_SEEDS)
        or primary_audit.get("paired_units_per_seed_scale_family") != 1_000
        or primary_audit.get("seed_p_values_used_as_success_gate") is not False
        or primary_audit.get("family_holm_p_values_used_as_success_gate") is not False
        or any(primary_audit.get(name) is not True for name in REQUIRED_CORE_TRUE_AUDITS)
    ):
        raise RuntimeError(
            "P3 is deferred until the digest-bound five-seed P2 core audit is terminal."
        )

    confirmatory = _load(nine_seed_core, "nine-seed P2 core audit")
    confirmatory_audit = confirmatory.get("audit", {})
    _bound_raw_matrix(confirmatory, "Nine-seed P2 core audit")
    pooling = confirmatory.get("pooling_audit", {})
    inference = confirmatory.get("confirmatory_inference", {})
    if (
        confirmatory.get("experiment_id") != "p2-nine-seed-core-matrix-audit-v1"
        or confirmatory.get("source", {}).get("dirty") is not False
        or confirmatory_audit.get("unique_shards") != 8_100
        or confirmatory_audit.get("independent_seed_clusters_per_cell")
        != EXPECTED_CONFIRMATORY_SEEDS
        or confirmatory_audit.get("paired_units_per_seed_scale_family") != 1_000
        or confirmatory_audit.get("seed_p_values_used_as_success_gate") is not False
        or confirmatory_audit.get("family_holm_p_values_used_as_success_gate") is not False
        or any(
            confirmatory_audit.get(name) is not True
            for name in REQUIRED_CORE_TRUE_AUDITS
        )
        or pooling.get("identical_frozen_contracts") is not True
        or pooling.get("disjoint_training_seeds") is not True
        or pooling.get("outcome_dependent_early_stopping") is not False
        or inference.get("independent_training_seeds_per_scale")
        != EXPECTED_CONFIRMATORY_SEEDS
        or inference.get("exact_sign_assignments") != 512
        or inference.get("outcome_dependent_early_stopping") is not False
    ):
        raise RuntimeError(
            "P3 is deferred until the digest-bound nine-seed P2 core audit is terminal."
        )
    return primary, confirmatory


def require_p3_sequence_gate(
    p2_matrix: Path,
    causal_gate: Path,
    nine_seed_causal: Path = DEFAULT_NINE_SEED_CAUSAL,
    primary_core_audit: Path = DEFAULT_PRIMARY_CORE_AUDIT,
    nine_seed_core: Path = DEFAULT_NINE_SEED_CORE,
) -> dict[str, Any]:
    """Require terminal primary and nine-seed P2 evidence before any P3 execution."""
    matrix = _load(p2_matrix, "P2 matrix summary")
    design = matrix.get("frozen_design", {})
    if (
        matrix.get("experiment_id") != "p2-core-quality-matrix-progress-v1"
        or matrix.get("completed_shards") != EXPECTED_P2_SHARDS
        or design.get("total_expected_shards") != EXPECTED_P2_SHARDS
        or tuple(design.get("scales", ())) != EXPECTED_SCALES
        or tuple(design.get("training_seeds", ())) != EXPECTED_TRAINING_SEEDS
    ):
        raise RuntimeError(
            "P3 is deferred until the frozen P2 synthetic core is complete "
            f"({matrix.get('completed_shards', 0)}/{EXPECTED_P2_SHARDS} shards)."
        )

    require_core_audits(p2_matrix, primary_core_audit, nine_seed_core)

    causal = _load(causal_gate, "P2 causal-gate audit")
    gate = causal.get("primary_causal_gate", {})
    audit = causal.get("audit", {})
    cells = gate.get("cells", [])
    if (
        causal.get("experiment_id") != "p2-causal-ablation-audit-v1"
        or audit.get("unique_shards") != EXPECTED_CAUSAL_SHARDS
        or audit.get("all_raw_shards_verified") is not True
        or audit.get("all_dependency_digests_verified") is not True
        or audit.get("all_record_digests_verified") is not True
        or audit.get("all_physical_predictions_identical") is not True
        or audit.get("outcome_dependent_early_stopping") is not False
        or audit.get("seed_p_values_used_as_success_gate") is not False
        or any(audit.get(name) is not True for name in REQUIRED_CAUSAL_TRUE_AUDITS)
        or gate.get("candidate") != "calibrated+pins"
        or gate.get("comparator") != "fixed+pins"
        or tuple(gate.get("scales", ())) != EXPECTED_SCALES
        or tuple(gate.get("budgets", ())) != EXPECTED_BUDGETS
        or gate.get("seeds_per_scale") != len(EXPECTED_TRAINING_SEEDS)
        or gate.get("required_cells") != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or not isinstance(cells, list)
        or len(cells) != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or {
            (cell.get("scale"), cell.get("budget")) for cell in cells
        }
        != {
            (scale, budget) for scale in EXPECTED_SCALES for budget in EXPECTED_BUDGETS
        }
        or any(not _causal_cell_is_consistent(cell) for cell in cells)
        or gate.get("passed") is not all(cell["passed"] for cell in cells)
    ):
        raise RuntimeError(
            "P3 is deferred until the complete preregistered 5-seed, 2-scale "
            "causal audit is available, including failed or bounded cells."
        )
    confirmatory = _load(nine_seed_causal, "nine-seed P2 causal-gate audit")
    confirmatory_gate = confirmatory.get("primary_causal_gate", {})
    confirmatory_audit = confirmatory.get("audit", {})
    confirmatory_cells = confirmatory_gate.get("cells", [])
    if (
        confirmatory.get("experiment_id")
        != "p2-nine-seed-causal-ablation-audit-v1"
        or confirmatory_audit.get("unique_shards")
        != EXPECTED_CONFIRMATORY_CAUSAL_SHARDS
        or confirmatory_audit.get("independent_seed_clusters_per_cell")
        != EXPECTED_CONFIRMATORY_SEEDS
        or confirmatory_audit.get("all_raw_shards_verified") is not True
        or confirmatory_audit.get("all_dependency_digests_verified") is not True
        or confirmatory_audit.get("all_record_digests_verified") is not True
        or confirmatory_audit.get("all_physical_predictions_identical") is not True
        or confirmatory_audit.get("outcome_dependent_early_stopping") is not False
        or confirmatory_audit.get("seed_p_values_used_as_success_gate") is not False
        or any(
            confirmatory_audit.get(name) is not True
            for name in REQUIRED_CAUSAL_TRUE_AUDITS
        )
        or confirmatory.get("pooling_audit", {}).get("identical_frozen_contracts")
        is not True
        or confirmatory.get("pooling_audit", {}).get("disjoint_training_seeds")
        is not True
        or confirmatory.get("confirmatory_inference", {}).get("exact_sign_assignments")
        != 512
        or confirmatory_gate.get("candidate") != "calibrated+pins"
        or confirmatory_gate.get("comparator") != "fixed+pins"
        or tuple(confirmatory_gate.get("scales", ())) != EXPECTED_SCALES
        or tuple(confirmatory_gate.get("budgets", ())) != EXPECTED_BUDGETS
        or confirmatory_gate.get("seeds_per_scale") != EXPECTED_CONFIRMATORY_SEEDS
        or confirmatory_gate.get("required_cells")
        != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or not isinstance(confirmatory_cells, list)
        or len(confirmatory_cells) != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or {
            (cell.get("scale"), cell.get("budget")) for cell in confirmatory_cells
        }
        != {
            (scale, budget) for scale in EXPECTED_SCALES for budget in EXPECTED_BUDGETS
        }
        or any(not _causal_cell_is_consistent(cell) for cell in confirmatory_cells)
        or confirmatory_gate.get("passed")
        is not all(cell["passed"] for cell in confirmatory_cells)
    ):
        raise RuntimeError(
            "P3 is deferred until the complete preregistered nine-seed, two-scale "
            "confirmatory causal audit is available, including failed or bounded cells."
        )

    qualified = (
        gate.get("passed") is True
        and all(cell["passed"] for cell in cells)
        and confirmatory_gate.get("passed") is True
        and all(cell["passed"] for cell in confirmatory_cells)
    )
    return {
        "causal_candidate": "calibrated+pins",
        "causal_candidate_qualified": qualified,
        "baseline_evaluation_required": True,
        "outcome_dependent_early_stopping": False,
        "required_scale_seed_completion_verified": True,
        "causal_statistical_audit_verified": True,
        "confirmatory_seed_clusters_per_cell": EXPECTED_CONFIRMATORY_SEEDS,
        "dependencies": {
            "p2_matrix": {"path": str(p2_matrix), "sha256": sha256(p2_matrix)},
            "primary_core": {
                "path": str(primary_core_audit),
                "sha256": sha256(primary_core_audit),
            },
            "nine_seed_core": {
                "path": str(nine_seed_core),
                "sha256": sha256(nine_seed_core),
            },
            "primary_causal": {
                "path": str(causal_gate),
                "sha256": sha256(causal_gate),
            },
            "nine_seed_causal": {
                "path": str(nine_seed_causal),
                "sha256": sha256(nine_seed_causal),
            },
        },
        "interpretation": (
            "qualified for transfer"
            if qualified
            else "causal arm withheld; native and fixed external baselines still proceed"
        ),
    }
