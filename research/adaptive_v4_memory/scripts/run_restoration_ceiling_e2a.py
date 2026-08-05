from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as model_utils
import run_causal_identifiability_atlas as atlas
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import (
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    CSASelectionPlan,
    CSASelectionProbe,
    generate_adaptive_memory_workload,
)
from nano_deepseek_v4.memory_controller import PlannedCSASelection

EXPERIMENT_ID = "restoration-ceiling-e2a-v1"
FAMILIES = atlas.FAMILIES
ABSOLUTE_INDEX_K = (1, 2, 4, 8)
OBSERVABLE_ARMS = tuple(f"index-k{value}" for value in ABSOLUTE_INDEX_K) + (
    "full-compressed",
)
ARMS = ("native", "identity-replay", *OBSERVABLE_ARMS, "force-evidence")

_canonical = atlas._canonical
_digest = atlas._digest
_file_digest = atlas._file_digest
_tensor_digest = atlas._tensor_digest
_atomic_json = atlas._atomic_json
coordinates = atlas.coordinates
cell_path = atlas.cell_path
workload_identity = atlas.workload_identity
evaluation_seed = atlas.evaluation_seed


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected restoration-ceiling manifest.")
    items = coordinates(payload)
    if payload.get("expected_cells") != len(items) or len(items) != 160:
        raise ValueError("The frozen restoration ceiling must contain exactly 160 cells.")
    if tuple(payload.get("arms", ())) != ARMS:
        raise ValueError("Restoration-ceiling arm contract drifted.")
    if tuple(payload.get("families", ())) != FAMILIES:
        raise ValueError("Restoration-ceiling family contract drifted.")
    if tuple(payload.get("absolute_index_cardinalities", ())) != ABSOLUTE_INDEX_K:
        raise ValueError("Absolute index-cardinality contract drifted.")
    implementation = payload.get("implementation", {})
    bindings = (
        ("runner_sha256", Path(__file__)),
        ("analyzer_sha256", Path(implementation["analyzer_path"])),
        ("preregistration_sha256", Path(implementation["preregistration_path"])),
        ("atlas_dependency_sha256", Path(implementation["atlas_dependency_path"])),
    )
    for key, implementation_path in bindings:
        if implementation.get(key) != _file_digest(implementation_path):
            raise ValueError(f"Restoration-ceiling implementation binding drifted: {key}.")
    if payload.get("quality_values_accessed_before_freeze") is not False:
        raise ValueError("Manifest does not attest a pre-outcome freeze.")
    return payload


def _checkpoint(manifest: dict[str, Any], scale: str, seed: int) -> dict[str, Any]:
    return atlas._checkpoint(manifest, scale, seed)


def _query_index(record: Any, batch_index: int, position: int) -> int:
    return atlas._query_index(record, batch_index, position)


def _eligible_indices(record: Any, batch_index: int, query_index: int) -> tuple[int, ...]:
    return atlas._eligible_indices(record, batch_index, query_index)


def _top_indices(scores: torch.Tensor, eligible: tuple[int, ...], topk: int) -> tuple[int, ...]:
    if len(eligible) < topk:
        raise ValueError("Restoration-ceiling query has fewer than eight causal blocks.")
    indices = tuple(int(value) for value in scores.topk(topk).indices.detach().cpu().tolist())
    if len(set(indices)) != topk or any(index not in eligible for index in indices):
        raise ValueError("Native index route contains a duplicate or non-causal block.")
    return indices


def build_observable_routes(
    probe: CSASelectionProbe,
    *,
    query_positions: torch.Tensor,
    native_topk: int,
    trace_id: str,
) -> tuple[CSASelectionPlan, dict[str, CSASelectionPlan], list[dict[str, Any]], str]:
    """Build every answer-free route before evidence positions are accepted."""
    if native_topk not in ABSOLUTE_INDEX_K or not probe.records:
        raise ValueError("Observable route construction requires a supported native top-k.")
    identity_rows: list[PlannedCSASelection] = []
    policy_rows: dict[str, list[PlannedCSASelection]] = {
        arm: [] for arm in OBSERVABLE_ARMS
    }
    receipts: list[dict[str, Any]] = []
    for record in probe.records:
        for batch_index in range(query_positions.shape[0]):
            for query_column in range(query_positions.shape[1]):
                query_position = int(query_positions[batch_index, query_column])
                query_index = _query_index(record, batch_index, query_position)
                eligible = _eligible_indices(record, batch_index, query_index)
                if len(eligible) < max(ABSOLUTE_INDEX_K):
                    raise ValueError("Restoration-ceiling query has fewer than eight blocks.")
                score_tensor = record.scores[batch_index, query_index].detach()
                ends = record.block_end_positions[batch_index].detach().long().cpu()
                native_indices = _top_indices(score_tensor, eligible, native_topk)
                native_ends = tuple(int(ends[index]) for index in native_indices)
                identity_rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=native_ends,
                    )
                )
                selected: dict[str, list[int]] = {"native": list(native_ends)}
                for topk in ABSOLUTE_INDEX_K:
                    arm = f"index-k{topk}"
                    indices = _top_indices(score_tensor, eligible, topk)
                    selected_ends = tuple(int(ends[index]) for index in indices)
                    selected[arm] = list(selected_ends)
                    policy_rows[arm].append(
                        PlannedCSASelection(
                            layer_index=record.layer_index,
                            batch_index=batch_index,
                            query_position=query_position,
                            block_end_positions=selected_ends,
                        )
                    )
                full_ends = tuple(int(ends[index]) for index in eligible)
                selected["full-compressed"] = list(full_ends)
                policy_rows["full-compressed"].append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=full_ends,
                    )
                )
                receipts.append(
                    {
                        "layer_index": record.layer_index,
                        "batch_index": batch_index,
                        "query_column": query_column,
                        "query_position": query_position,
                        "eligible_blocks": len(eligible),
                        "selected_block_ends": selected,
                        "native_score_sha256": _digest(
                            [float(score_tensor[index]) for index in eligible]
                        ),
                    }
                )
    route_bundle = {
        "trace_id": trace_id,
        "native_topk": native_topk,
        "absolute_index_cardinalities": list(ABSOLUTE_INDEX_K),
        "routes": receipts,
    }
    return (
        CSASelectionPlan(trace_id, "identity-replay", tuple(identity_rows)),
        {
            arm: CSASelectionPlan(trace_id, arm, tuple(rows))
            for arm, rows in policy_rows.items()
        },
        receipts,
        _digest(route_bundle),
    )


def _outcome(output: Any, workload: AdaptiveMemoryWorkloadBatch) -> dict[str, Any]:
    return atlas._outcome(output, workload)


@torch.inference_mode()
def run_cell(
    model: Any,
    coordinate: dict[str, Any],
    *,
    native_topk: int,
    evaluation_seed_value: int,
    manifest_sha256: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    workload = generate_adaptive_memory_workload(
        task,
        family=coordinate["family"],
        batch_size=1,
        sequence_length=coordinate["context"],
        generator=torch.Generator().manual_seed(evaluation_seed_value),
        conversation_offset=coordinate["replicate"],
        device="cuda",
    )
    model_utils._set_topk(model, native_topk)
    probe = CSASelectionProbe()
    native = model(workload.input_ids, use_cache=False, csa_probe=probe)
    trace_id = f"{EXPERIMENT_ID}:{_digest(coordinate)[:16]}"
    identity_plan, policy_plans, observable_receipts, observable_digest = (
        build_observable_routes(
            probe,
            query_positions=workload.query_positions,
            native_topk=native_topk,
            trace_id=trace_id,
        )
    )
    identity = model(workload.input_ids, use_cache=False, selection_plan=identity_plan)
    if not torch.equal(native.logits, identity.logits):
        raise RuntimeError("Identity route replay changed model logits.")
    outcomes = {
        "native": _outcome(native, workload),
        "identity-replay": _outcome(identity, workload),
    }
    native_equivalent_arm = f"index-k{native_topk}"
    for arm, plan in policy_plans.items():
        output = model(workload.input_ids, use_cache=False, selection_plan=plan)
        if arm == native_equivalent_arm and not torch.equal(native.logits, output.logits):
            raise RuntimeError("Native-equivalent absolute-K replay changed model logits.")
        outcomes[arm] = _outcome(output, workload)
        del output

    # Evidence is deliberately opened only after the observable route bundle is hashed.
    oracle_plan, oracle_receipts = atlas.build_evidence_oracle(
        probe, workload, topk=native_topk, trace_id=trace_id
    )
    oracle = model(workload.input_ids, use_cache=False, selection_plan=oracle_plan)
    outcomes["force-evidence"] = _outcome(oracle, workload)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": manifest_sha256,
        "coordinate": coordinate,
        "workload_identity": workload_identity(coordinate),
        "evaluation_seed": evaluation_seed_value,
        "input_ids_sha256": _tensor_digest(workload.input_ids),
        "checkpoint": checkpoint,
        "route_contract": {
            "native_topk": native_topk,
            "native_equivalent_arm": native_equivalent_arm,
            "identity_replay_bitwise_equal": True,
            "native_equivalent_replay_bitwise_equal": True,
            "observable_routes_exclude_evidence_and_targets": True,
            "observable_route_bundle_sha256": observable_digest,
            "observable_receipts": observable_receipts,
            "observable_receipts_sha256": _digest(observable_receipts),
            "oracle_receipts": oracle_receipts,
            "oracle_receipts_sha256": _digest(oracle_receipts),
        },
        "targets": workload.targets.cpu().tolist(),
        "outcomes": outcomes,
    }


def _validate_preflight_payload(payload: dict[str, Any]) -> dict[str, Any]:
    route = payload["route_contract"]
    if not (
        route["identity_replay_bitwise_equal"]
        and route["native_equivalent_replay_bitwise_equal"]
    ):
        raise RuntimeError("Restoration-ceiling replay invariant failed.")
    for receipt in route["observable_receipts"]:
        selected = receipt["selected_block_ends"]
        for topk in ABSOLUTE_INDEX_K:
            ends = selected[f"index-k{topk}"]
            if len(ends) != topk or len(set(ends)) != topk:
                raise RuntimeError("Absolute-K route cardinality failed.")
        full = selected["full-compressed"]
        if len(full) != receipt["eligible_blocks"] or len(set(full)) != len(full):
            raise RuntimeError("Full-compressed route coverage failed.")
    return {
        "status": "preflight_passed",
        "coordinate": payload["coordinate"],
        "input_ids_sha256": payload["input_ids_sha256"],
        "observable_route_bundle_sha256": route["observable_route_bundle_sha256"],
        "route_receipts": len(route["observable_receipts"]),
        "quality_values_disclosed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen E2-A restoration ceiling.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-coordinate-index", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The restoration-ceiling audit requires CUDA.")
    manifest = _load_manifest(args.manifest)
    manifest_sha256 = _file_digest(args.manifest)
    all_coordinates = coordinates(manifest)
    if args.preflight_coordinate_index is not None:
        if args.output_root is not None:
            raise ValueError("Preflight does not publish an output cell.")
        if not 0 <= args.preflight_coordinate_index < len(all_coordinates):
            raise ValueError("Preflight coordinate index is out of range.")
        coordinate = all_coordinates[args.preflight_coordinate_index]
        with acquire_gpu_lock(f"{EXPERIMENT_ID}-preflight"):
            checkpoint = _checkpoint(
                manifest, coordinate["scale"], coordinate["training_seed"]
            )
            model = model_utils._load_model(Path(checkpoint["path"]))
            payload = run_cell(
                model,
                coordinate,
                native_topk=int(manifest["fixed_topk"][coordinate["scale"]])
                * int(coordinate["budget_multiplier"]),
                evaluation_seed_value=evaluation_seed(
                    coordinate, manifest["evaluation_seed_namespace"]
                ),
                manifest_sha256=manifest_sha256,
                checkpoint=checkpoint,
            )
            print(json.dumps(_validate_preflight_payload(payload), sort_keys=True), flush=True)
        return
    if args.output_root is None:
        raise ValueError("Normal execution requires --output-root.")
    if args.output_root.resolve() != Path(manifest["output_root"]).resolve():
        raise ValueError("Output root differs from the frozen manifest.")
    expected_paths = {cell_path(args.output_root, item) for item in all_coordinates}
    observed_paths = set(args.output_root.rglob("replicate-*.json"))
    if observed_paths - expected_paths:
        raise RuntimeError("Restoration-ceiling namespace contains an unexpected cell.")
    with acquire_gpu_lock(EXPERIMENT_ID):
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for coordinate in all_coordinates:
            grouped.setdefault((coordinate["scale"], coordinate["training_seed"]), []).append(
                coordinate
            )
        committed = 0
        for (scale, seed), items in grouped.items():
            checkpoint = _checkpoint(manifest, scale, seed)
            model = model_utils._load_model(Path(checkpoint["path"]))
            for coordinate in items:
                path = cell_path(args.output_root, coordinate)
                if path.is_file():
                    prior = json.loads(path.read_text())
                    if (
                        prior.get("experiment_id") != EXPERIMENT_ID
                        or prior.get("manifest_sha256") != manifest_sha256
                        or prior.get("coordinate") != coordinate
                    ):
                        raise RuntimeError(f"Invalid resume cell: {path}")
                else:
                    payload = run_cell(
                        model,
                        coordinate,
                        native_topk=int(manifest["fixed_topk"][scale])
                        * int(coordinate["budget_multiplier"]),
                        evaluation_seed_value=evaluation_seed(
                            coordinate, manifest["evaluation_seed_namespace"]
                        ),
                        manifest_sha256=manifest_sha256,
                        checkpoint=checkpoint,
                    )
                    _atomic_json(path, payload)
                    print(
                        json.dumps({"status": "cell_committed", "coordinate": coordinate}),
                        flush=True,
                    )
                committed += 1
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if committed != manifest["expected_cells"]:
            raise RuntimeError("Restoration-ceiling completion cardinality drifted.")
        print(json.dumps({"status": "complete", "cells": committed}), flush=True)


if __name__ == "__main__":
    main()
