from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import p2_continuous_rank_contract as contract


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _validate_manifest_binding(
    payload: dict[str, Any], *, manifest_path: Path, manifest: dict[str, Any]
) -> None:
    binding = payload.get("manifest", {})
    _require(binding.get("path") == str(manifest_path), "Cell manifest path drifted.")
    _require(binding.get("sha256") == contract.sha256(manifest_path), "Cell manifest SHA drifted.")
    _require(
        binding.get("experiment_id") == contract.EXPERIMENT_ID,
        "Cell manifest experiment drifted.",
    )
    _require(
        binding.get("implementation_digest") == manifest["implementation"]["digest"],
        "Cell implementation digest drifted.",
    )


def _validate_source(payload: dict[str, Any]) -> None:
    source = payload.get("source", {})
    start = source.get("start", {})
    end = source.get("end", {})
    _require(start.get("dirty") is False, "Cell began from dirty source.")
    _require(end.get("dirty") is False, "Cell ended with dirty source.")
    _require(start.get("commit") == end.get("commit"), "Source commit changed during a cell.")


def _validate_cell(
    path: Path,
    *,
    experiment_id: str,
    scale: str,
    training_seed: int,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path)
    contract.reject_supervision_fields(payload)
    contract.validate_payload_digest(payload)
    _require(payload.get("schema_version") == 1, "Cell schema drifted.")
    _require(payload.get("experiment_id") == experiment_id, "Cell experiment drifted.")
    _require(payload.get("status") == "terminal", "Cell is not terminal.")
    _require(payload.get("scale") == scale, "Cell scale drifted.")
    _require(payload.get("training_seed") == training_seed, "Cell training seed drifted.")
    expected_index = contract.TRAINING_SEEDS.index(training_seed)
    _require(
        payload.get("calibration_seed") == contract.CALIBRATION_SEEDS[expected_index],
        "Cell calibration seed drifted.",
    )
    if experiment_id == contract.FULL_FORWARD_EXPERIMENT_ID:
        worker = importlib.import_module("collect_p2_full_forward_continuous_rank")
    elif experiment_id == contract.EXACT_PATH_EXPERIMENT_ID:
        worker = importlib.import_module("collect_p2_exact_path_layer_signals")
    else:
        raise ValueError(f"No deep cell validator exists for {experiment_id!r}.")
    validator = cast(
        Callable[[dict[str, Any]], None],
        getattr(worker, "validate_payload", None),
    )
    _require(callable(validator), "Cell worker lacks a deep payload validator.")
    validator(payload)
    _validate_manifest_binding(payload, manifest_path=manifest_path, manifest=manifest)
    _validate_source(payload)
    expected_input = contract.input_binding(manifest, scale, training_seed)
    _require(payload.get("input_binding") == expected_input, "Cell input binding drifted.")
    return payload


def _extreme_layers(boundary: dict[str, Any]) -> dict[str, Any]:
    top = boundary.get("top", {})
    bottom = boundary.get("bottom", {})
    top_candidates = top.get("point_boundary_candidates")
    bottom_candidates = bottom.get("point_boundary_candidates")
    _require(
        isinstance(top_candidates, list)
        and bool(top_candidates)
        and all(isinstance(layer, int) for layer in top_candidates),
        "Top boundary candidates are invalid.",
    )
    _require(
        isinstance(bottom_candidates, list)
        and bool(bottom_candidates)
        and all(isinstance(layer, int) for layer in bottom_candidates),
        "Bottom boundary candidates are invalid.",
    )
    unique_top = top_candidates[0] if len(top_candidates) == 1 else None
    unique_bottom = bottom_candidates[0] if len(bottom_candidates) == 1 else None
    eligible = (
        top.get("identified") is True
        and bottom.get("identified") is True
        and unique_top is not None
        and unique_bottom is not None
        and unique_top != unique_bottom
    )
    return {
        "eligible": eligible,
        "top_layer": unique_top,
        "bottom_layer": unique_bottom,
        "top_candidates": top_candidates,
        "bottom_candidates": bottom_candidates,
    }


def _full_forward_cell_result(payload: dict[str, Any]) -> dict[str, Any]:
    reproduction = payload.get("reproduction", {})
    _require(reproduction.get("all_exact") is True, "Frozen P1 reproduction failed.")
    per_budget = reproduction.get("budgets", {})
    _require(set(per_budget) == {"1x", *contract.BUDGETS}, "Reproduction budget set drifted.")
    _require(
        all(item.get("exact") is True for item in per_budget.values()),
        "A frozen P1 calibration budget did not reproduce exactly.",
    )
    budgets = payload.get("budgets", {})
    _require(set(budgets) == set(contract.BUDGETS), "Continuous budget set drifted.")
    result: dict[str, Any] = {}
    for budget in contract.BUDGETS:
        item = budgets[budget]
        _require(item.get("slice_count") == 45, "Full-forward slice count drifted.")
        _require(
            item.get("bootstrap_seed")
            == contract.bootstrap_seed(
                str(payload["scale"]), int(payload["training_seed"]), budget
            ),
            "Full-forward bootstrap seed drifted.",
        )
        extremes = _extreme_layers(item.get("boundary_evidence", {}))
        result[budget] = {
            **extremes,
            "boundary_evidence_sha256": contract.json_digest(item["boundary_evidence"]),
        }
    return result


def summarize_full_forward(*, manifest_path: Path = contract.MANIFEST_PATH) -> dict[str, Any]:
    manifest = contract.load_manifest(manifest_path)
    cells: list[dict[str, Any]] = []
    for scale in contract.SCALES:
        for training_seed in contract.TRAINING_SEEDS:
            path = contract.full_forward_output_path(scale, training_seed)
            payload = _validate_cell(
                path,
                experiment_id=contract.FULL_FORWARD_EXPERIMENT_ID,
                scale=scale,
                training_seed=training_seed,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            cells.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": payload["calibration_seed"],
                    "path": str(path),
                    "sha256": contract.sha256(path),
                    "payload_sha256": payload["payload_sha256"],
                    "budgets": _full_forward_cell_result(payload),
                }
            )
    eligible = all(item["eligible"] for cell in cells for item in cell["budgets"].values())
    payload = {
        "schema_version": 1,
        "experiment_id": contract.FULL_FORWARD_AUDIT_ID,
        "status": "terminal",
        "manifest": contract.manifest_binding(manifest_path),
        "cells": cells,
        "observed_cells": len(cells),
        "observed_seed_scale_budget_cells": len(cells) * len(contract.BUDGETS),
        "all_prior_calibrations_reproduced_exactly": True,
        "all_continuous_boundaries_identified": eligible,
        "exact_path_phase_permitted": eligible,
        "quality_execution_permitted": False,
        "claim_boundary": (
            "Calibration-only full-forward structural feasibility; no mechanism or quality effect."
        ),
    }
    payload["payload_sha256"] = contract.payload_digest(payload)
    return payload


def _full_audit_lookup(
    full_audit: dict[str, Any],
) -> dict[tuple[str, int, str], tuple[int | None, int | None]]:
    result: dict[tuple[str, int, str], tuple[int | None, int | None]] = {}
    for cell in full_audit["cells"]:
        for budget, item in cell["budgets"].items():
            result[(cell["scale"], cell["training_seed"], budget)] = (
                item["top_layer"],
                item["bottom_layer"],
            )
    return result


def _validate_full_audit(path: Path, manifest_path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    contract.reject_supervision_fields(payload)
    contract.validate_payload_digest(payload)
    _require(payload.get("experiment_id") == contract.FULL_FORWARD_AUDIT_ID, "Wrong phase-1 audit.")
    _require(payload.get("status") == "terminal", "Phase-1 audit is not terminal.")
    _require(payload.get("exact_path_phase_permitted") is True, "Exact path remains blocked.")
    binding = payload.get("manifest", {})
    _require(binding.get("path") == str(manifest_path), "Phase-1 manifest path drifted.")
    _require(binding.get("sha256") == contract.sha256(manifest_path), "Phase-1 manifest drifted.")
    recomputed = summarize_full_forward(manifest_path=manifest_path)
    _require(
        payload == recomputed,
        "Phase-1 audit differs from a deep recomputation of all bound full-forward cells.",
    )
    return payload


def _exact_path_cell_result(
    payload: dict[str, Any],
    *,
    full_lookup: dict[tuple[str, int, str], tuple[int | None, int | None]],
) -> dict[str, Any]:
    _require(
        payload.get("batch_runs") == contract.EXACT_PATH_BATCH_RUNS_PER_CELL,
        "Exact-path batch-run count drifted.",
    )
    _require(
        payload.get("executed_conversations")
        == contract.EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL,
        "Exact-path executed-conversation count drifted.",
    )
    budgets = payload.get("budgets", {})
    _require(set(budgets) == set(contract.BUDGETS), "Exact-path budget set drifted.")
    result: dict[str, Any] = {}
    for budget in contract.BUDGETS:
        item = budgets[budget]
        repeat = item.get("repeat_integrity", {})
        workload_repeat = repeat.get("workload_streams_identical") is True
        semantic_repeat = repeat.get("semantic_query_streams_identical") is True
        score_repeat = repeat.get("fp32_score_streams_identical") is True
        observation_repeat = repeat.get("observation_streams_identical") is True
        _require(
            item.get("bootstrap_seed")
            == contract.bootstrap_seed(
                str(payload["scale"]), int(payload["training_seed"]), budget
            ),
            "Exact-path bootstrap seed drifted.",
        )
        extremes = _extreme_layers(item.get("path_boundary_evidence", {}))
        top = extremes["top_layer"]
        bottom = extremes["bottom_layer"]
        frozen_extremes = full_lookup[(payload["scale"], payload["training_seed"], budget)]
        transfer = (top, bottom) == frozen_extremes
        eligible = bool(
            workload_repeat
            and semantic_repeat
            and score_repeat
            and observation_repeat
            and extremes["eligible"]
            and transfer
        )
        result[budget] = {
            "eligible": eligible,
            "top_layer": top,
            "bottom_layer": bottom,
            "top_candidates": extremes["top_candidates"],
            "bottom_candidates": extremes["bottom_candidates"],
            "path_boundary_identified": extremes["eligible"],
            "workload_streams_identical": workload_repeat,
            "semantic_query_streams_identical": semantic_repeat,
            "fp32_score_streams_identical": score_repeat,
            "observation_streams_identical": observation_repeat,
            "full_forward_extremes": list(frozen_extremes),
            "exact_path_extremes": [top, bottom],
            "path_transfer_passed": transfer,
            "boundary_evidence_sha256": contract.json_digest(item["path_boundary_evidence"]),
        }
    return result


def summarize_terminal(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    full_audit_path: Path | None = None,
) -> dict[str, Any]:
    manifest = contract.load_manifest(manifest_path)
    phase_1_path = full_audit_path or contract.full_forward_audit_path()
    full_audit = _validate_full_audit(phase_1_path, manifest_path)
    full_lookup = _full_audit_lookup(full_audit)
    _require(len(full_lookup) == 20, "Phase-1 extreme-layer inventory is incomplete.")
    cells: list[dict[str, Any]] = []
    for scale in contract.SCALES:
        for training_seed in contract.TRAINING_SEEDS:
            path = contract.exact_path_output_path(scale, training_seed)
            payload = _validate_cell(
                path,
                experiment_id=contract.EXACT_PATH_EXPERIMENT_ID,
                scale=scale,
                training_seed=training_seed,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            _require(
                payload.get("full_forward_audit")
                == {"path": str(phase_1_path), "sha256": contract.sha256(phase_1_path)},
                "Exact-path cell phase-1 binding drifted.",
            )
            cells.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": payload["calibration_seed"],
                    "path": str(path),
                    "sha256": contract.sha256(path),
                    "payload_sha256": payload["payload_sha256"],
                    "budgets": _exact_path_cell_result(payload, full_lookup=full_lookup),
                }
            )
    eligible = all(item["eligible"] for cell in cells for item in cell["budgets"].values())
    repeat_identical = all(
        item["workload_streams_identical"]
        and item["semantic_query_streams_identical"]
        and item["fp32_score_streams_identical"]
        and item["observation_streams_identical"]
        for cell in cells
        for item in cell["budgets"].values()
    )
    boundaries_identified = all(
        item["path_boundary_identified"] for cell in cells for item in cell["budgets"].values()
    )
    path_transfer = all(
        item["path_transfer_passed"] for cell in cells for item in cell["budgets"].values()
    )
    payload = {
        "schema_version": 1,
        "experiment_id": contract.TERMINAL_AUDIT_ID,
        "status": "terminal",
        "manifest": contract.manifest_binding(manifest_path),
        "full_forward_audit": {
            "path": str(phase_1_path),
            "sha256": contract.sha256(phase_1_path),
            "payload_sha256": full_audit["payload_sha256"],
        },
        "cells": cells,
        "observed_cells": len(cells),
        "observed_seed_scale_budget_cells": len(cells) * len(contract.BUDGETS),
        "all_repeat_extractions_identical": repeat_identical,
        "all_exact_path_boundaries_identified": boundaries_identified,
        "all_extremes_transfer_from_full_forward": path_transfer,
        "rank_probe_manifest_may_be_designed": eligible,
        "rank_probe_execution_permitted": False,
        "quality_execution_permitted": False,
        "claim_boundary": (
            "Calibration-only layer-rank feasibility under a uniform-reference exact path; "
            "no adaptive-quota, pin, or quality effect."
        ),
    }
    payload["payload_sha256"] = contract.payload_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen 707 continuous-rank study.")
    parser.add_argument("--phase", choices=("full-forward", "terminal"), required=True)
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--full-forward-audit", type=Path)
    args = parser.parse_args()
    if args.phase == "full-forward":
        payload = summarize_full_forward(manifest_path=args.manifest)
        output = contract.full_forward_audit_path()
    else:
        payload = summarize_terminal(
            manifest_path=args.manifest,
            full_audit_path=args.full_forward_audit,
        )
        output = contract.terminal_audit_path()
    contract.write_json_exclusive(output, payload)
    print(
        json.dumps(
            {
                "event": "continuous_rank_audit_terminal",
                "phase": args.phase,
                "output": str(output),
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
