from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

from p3_sequence_gate import DEFAULT_NINE_SEED_CORE, require_core_audits
from verify_p3_natural_model import sha256

SCALES = ("s55", "s151")
BUDGETS = ("2x", "4x")
ELIGIBLE_PRESSES = {
    "streaming_llm",
    "snapkv",
    "pyramidkv",
    "adakv_snapkv",
    "expected_attention",
    "critical_expected_attention",
}
REQUIRED_CAUSAL_TRUE_AUDITS = (
    "exact_seed_randomization_verified",
    "physical_controller_budget_verified",
    "exact_statistical_cell_coverage_verified",
    "family_holm_bonferroni_verified",
    "contrast_holm_bonferroni_verified",
    "primary_four_cell_bonferroni_verified",
    "required_scale_seed_completion_verified",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is not available yet: {path}")
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"{name} is not a JSON object: {path}")
    return payload


def _audited(payload: dict[str, Any], *, shards: int, seeds: int) -> bool:
    audit = payload.get("audit", {})
    return bool(
        audit.get("unique_shards") == shards
        and audit.get("independent_seed_clusters_per_cell") == seeds
        and audit.get("all_raw_shards_verified") is True
        and audit.get("all_dependency_digests_verified") is True
        and audit.get("all_record_digests_verified") is True
        and audit.get("no_budget_violations") is True
    )


def _causal_audited(payload: dict[str, Any], *, shards: int, seeds: int) -> bool:
    audit = payload.get("audit", {})
    return bool(
        _audited(payload, shards=shards, seeds=seeds)
        and audit.get("all_physical_predictions_identical") is True
        and audit.get("exact_config_reuse_verified") is True
        and audit.get("outcome_dependent_early_stopping") is False
        and audit.get("seed_p_values_used_as_success_gate") is False
        and all(audit.get(name) is True for name in REQUIRED_CAUSAL_TRUE_AUDITS)
    )


def require_cross_family_sequence_gate(
    *,
    primary_core: Path,
    primary_causal: Path,
    nine_seed_causal: Path,
    fixed_selection: Path,
    nine_seed_core: Path = DEFAULT_NINE_SEED_CORE,
) -> dict[str, Any]:
    """Require all pre-outcome evidence frozen for the Phi transfer cohort."""
    core = _load(primary_core, "primary P2 core audit")
    primary_matrix = Path(core.get("raw_matrix", {}).get("path", ""))
    core, _confirmatory_core = require_core_audits(
        primary_matrix,
        primary_core,
        nine_seed_core,
    )
    _require(
        core.get("experiment_id") == "p2-core-quality-matrix-audit-v1"
        and _audited(core, shards=4_500, seeds=5)
        and core.get("audit", {}).get("exact_seed_randomization_verified") is True
        and core.get("audit", {}).get("exact_statistical_cell_coverage_verified") is True,
        "Cross-family P3 is deferred until the strict five-seed P2 core audit is terminal.",
    )

    causal = _load(primary_causal, "primary P2 causal audit")
    primary_gate = causal.get("primary_causal_gate", {})
    _require(
        causal.get("experiment_id") == "p2-causal-ablation-audit-v1"
        and _causal_audited(causal, shards=9_000, seeds=5)
        and primary_gate.get("candidate") == "calibrated+pins"
        and primary_gate.get("comparator") == "fixed+pins",
        "Cross-family P3 is deferred until the terminal five-seed P2 causal audit exists.",
    )

    combined = _load(nine_seed_causal, "nine-seed P2 causal audit")
    combined_gate = combined.get("primary_causal_gate", {})
    cells = combined_gate.get("cells", [])
    identities = {
        (cell.get("scale"), cell.get("budget")) for cell in cells if isinstance(cell, dict)
    }
    _require(
        combined.get("experiment_id") == "p2-nine-seed-causal-ablation-audit-v1"
        and _causal_audited(combined, shards=16_200, seeds=9)
        and combined.get("pooling_audit", {}).get("identical_frozen_contracts") is True
        and combined.get("pooling_audit", {}).get("disjoint_training_seeds") is True
        and combined_gate.get("candidate") == "calibrated+pins"
        and combined_gate.get("comparator") == "fixed+pins"
        and combined_gate.get("required_cells") == 4
        and identities == set(product(SCALES, BUDGETS)),
        "Cross-family P3 is deferred until the terminal nine-seed P2 causal audit exists.",
    )

    selection = _load(fixed_selection, "Qwen fixed-baseline selection")
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False
        and selection.get("selected_arm") in ELIGIBLE_PRESSES
        and selection.get("selected_compression_ratio") == 0.5,
        "Cross-family P3 requires the terminal Qwen-selected 50% fixed baseline.",
    )

    return {
        "outcome_dependent_execution": False,
        "causal_statistical_audits_verified": True,
        "phi_specific_tuning": False,
        "primary_causal_gate_passed": primary_gate.get("passed") is True,
        "nine_seed_causal_gate_passed": combined_gate.get("passed") is True,
        "selected_press": selection["selected_arm"],
        "selected_compression_ratio": selection["selected_compression_ratio"],
        "dependencies": {
            "primary_core": {"path": str(primary_core), "sha256": sha256(primary_core)},
            "nine_seed_core": {
                "path": str(nine_seed_core),
                "sha256": sha256(nine_seed_core),
            },
            "primary_causal": {
                "path": str(primary_causal),
                "sha256": sha256(primary_causal),
            },
            "nine_seed_causal": {
                "path": str(nine_seed_causal),
                "sha256": sha256(nine_seed_causal),
            },
            "fixed_selection": {
                "path": str(fixed_selection),
                "sha256": sha256(fixed_selection),
            },
        },
    }
