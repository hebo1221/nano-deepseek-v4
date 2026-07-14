from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

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


def require_cross_family_sequence_gate(
    *,
    primary_core: Path,
    primary_causal: Path,
    nine_seed_causal: Path,
    fixed_selection: Path,
) -> dict[str, Any]:
    """Require all pre-outcome evidence frozen for the Phi transfer cohort."""
    core = _load(primary_core, "primary P2 core audit")
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
        and _audited(causal, shards=9_000, seeds=5)
        and causal.get("audit", {}).get("all_physical_predictions_identical") is True
        and causal.get("audit", {}).get("exact_config_reuse_verified") is True
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
        and _audited(combined, shards=16_200, seeds=9)
        and combined.get("audit", {}).get("all_physical_predictions_identical") is True
        and combined.get("audit", {}).get("exact_config_reuse_verified") is True
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
        "phi_specific_tuning": False,
        "primary_causal_gate_passed": primary_gate.get("passed") is True,
        "nine_seed_causal_gate_passed": combined_gate.get("passed") is True,
        "selected_press": selection["selected_arm"],
        "selected_compression_ratio": selection["selected_compression_ratio"],
        "dependencies": {
            "primary_core": {"path": str(primary_core), "sha256": sha256(primary_core)},
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
