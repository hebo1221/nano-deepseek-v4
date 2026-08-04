from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as model_utils
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

EXPERIMENT_ID = "causal-evidence-route-h0-pilot-v1"
ARMS = ("native", "identity-replay", "force-evidence", "matched-control")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected causal-evidence pilot manifest.")
    grid = payload.get("grid", {})
    expected = math.prod(
        len(grid[key])
        for key in (
            "scales",
            "training_seeds",
            "budget_multipliers",
            "contexts",
            "families",
            "replicates",
        )
    )
    if payload.get("expected_cells") != expected or not 100 <= expected <= 300:
        raise ValueError("The frozen pilot must contain 100--300 cells.")
    implementation = payload.get("implementation", {})
    if implementation.get("script_sha256") != _file_digest(Path(__file__)):
        raise ValueError("Pilot implementation digest drifted.")
    if tuple(payload.get("arms", ())) != ARMS:
        raise ValueError("Pilot arm contract drifted.")
    return payload


def coordinates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grid = manifest["grid"]
    return [
        {
            "scale": scale,
            "training_seed": seed,
            "budget_multiplier": budget_multiplier,
            "context": context,
            "family": family,
            "replicate": replicate,
        }
        for scale, seed, budget_multiplier, context, family, replicate in product(
            grid["scales"],
            grid["training_seeds"],
            grid["budget_multipliers"],
            grid["contexts"],
            grid["families"],
            grid["replicates"],
        )
    ]


def cell_path(root: Path, coordinate: dict[str, Any]) -> Path:
    return (
        root
        / coordinate["scale"]
        / f"seed-{coordinate['training_seed']}"
        / f"budget-{coordinate['budget_multiplier']}x"
        / coordinate["family"]
        / f"context-{coordinate['context']}"
        / f"replicate-{coordinate['replicate']}.json"
    )


def _evaluation_seed(coordinate: dict[str, Any], namespace: str) -> int:
    value = _digest({"namespace": namespace, "coordinate": coordinate})
    return int(value[:16], 16) % (2**63 - 1)


def _query_index(record: Any, batch_index: int, position: int) -> int:
    matches = record.query_positions[batch_index].eq(position).nonzero(as_tuple=False).flatten()
    if matches.numel() != 1:
        raise ValueError("Workload query is absent or duplicated in the CSA probe.")
    return int(matches.item())


def _evidence_index(ends: torch.Tensor, evidence_position: int, query_position: int) -> int:
    index = int(torch.searchsorted(ends.contiguous(), evidence_position).item())
    if index >= ends.numel() or int(ends[index]) > query_position:
        raise ValueError("Evidence does not map to a causal compressed block.")
    return index


def build_counterfactual_plans(
    probe: CSASelectionProbe,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    topk: int,
    trace_id: str,
) -> tuple[CSASelectionPlan, CSASelectionPlan, CSASelectionPlan, list[dict[str, Any]]]:
    """Build equal-cardinality routes that differ by one evidence/control anchor."""
    if topk <= 0 or not probe.records:
        raise ValueError("Counterfactual routes require positive top-k and CSA records.")
    identity_rows: list[PlannedCSASelection] = []
    evidence_rows: list[PlannedCSASelection] = []
    control_rows: list[PlannedCSASelection] = []
    receipts: list[dict[str, Any]] = []
    for record in probe.records:
        for batch_index in range(workload.input_ids.shape[0]):
            for column in range(workload.query_positions.shape[1]):
                query_position = int(workload.query_positions[batch_index, column])
                evidence_position = int(workload.evidence_positions[batch_index, column])
                query_index = _query_index(record, batch_index, query_position)
                native_scores = record.scores[batch_index, query_index].detach()
                native = [int(index) for index in native_scores.topk(topk).indices.cpu().tolist()]
                scores = native_scores.float().cpu()
                ends = record.block_end_positions[batch_index].detach().long().cpu()
                evidence_index = _evidence_index(ends, evidence_position, query_position)
                finite = [index for index, score in enumerate(scores.tolist()) if math.isfinite(score)]
                if evidence_index not in finite:
                    raise ValueError("Evidence block has a non-finite selector score.")
                if len(finite) < topk:
                    raise ValueError("Query lacks enough causal blocks for a matched route.")
                remaining = sorted(
                    (index for index in finite if index not in native),
                    key=lambda index: (-float(scores[index]), index),
                )
                non_evidence = [
                    index for index in (*native, *remaining) if index != evidence_index
                ]
                if len(non_evidence) < topk:
                    raise ValueError("Query lacks a non-evidence matched-control anchor.")
                common = non_evidence[: topk - 1]
                control_index = non_evidence[topk - 1]
                evidence_ends = tuple(int(ends[index]) for index in (*common, evidence_index))
                control_ends = tuple(int(ends[index]) for index in (*common, control_index))
                identity_rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=tuple(int(ends[index]) for index in native),
                    )
                )
                evidence_rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=evidence_ends,
                    )
                )
                control_rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=control_ends,
                    )
                )
                receipts.append(
                    {
                        "layer_index": record.layer_index,
                        "batch_index": batch_index,
                        "query_column": column,
                        "query_position": query_position,
                        "evidence_position": evidence_position,
                        "evidence_block_end": int(ends[evidence_index]),
                        "control_block_end": int(ends[control_index]),
                        "native_contains_evidence": evidence_index in native,
                        "identity_arm": (
                            "force-evidence" if evidence_index in native else "matched-control"
                        ),
                        "selected_blocks": topk,
                    }
                )
    return (
        CSASelectionPlan(trace_id, "identity-replay", tuple(identity_rows)),
        CSASelectionPlan(trace_id, "force-evidence", tuple(evidence_rows)),
        CSASelectionPlan(trace_id, "matched-control", tuple(control_rows)),
        receipts,
    )


def _outcome(output: Any, workload: AdaptiveMemoryWorkloadBatch) -> dict[str, Any]:
    batch = torch.arange(workload.input_ids.shape[0], device=workload.input_ids.device)
    logits = torch.stack(
        [
            output.logits[batch, workload.query_positions[:, column]].float()
            for column in range(workload.query_positions.shape[1])
        ],
        dim=1,
    )
    log_probabilities = logits.log_softmax(dim=-1)
    target_log_prob = log_probabilities.gather(
        -1, workload.targets.unsqueeze(-1)
    ).squeeze(-1)
    predictions = logits.argmax(dim=-1)
    poison_margin: list[list[float]] | None = None
    if workload.family == "adversarial-lexical-distractors":
        length = workload.input_ids.shape[1]
        poison_positions = (length // 2, length // 2 + 6, length // 2 + 12)
        poison_targets = workload.input_ids[:, poison_positions]
        poison_log_prob = log_probabilities.gather(
            -1,
            poison_targets.unsqueeze(1).expand(-1, workload.targets.shape[1], -1),
        ).max(dim=-1).values
        poison_margin = (poison_log_prob - target_log_prob).cpu().tolist()
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": predictions.eq(workload.targets).cpu().tolist(),
        "target_log_prob": target_log_prob.cpu().tolist(),
        "poison_wrong_minus_gold_log_prob_margin": poison_margin,
    }


@torch.inference_mode()
def _run_cell(
    model: Any,
    coordinate: dict[str, Any],
    *,
    topk: int,
    evaluation_seed: int,
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
        generator=torch.Generator().manual_seed(evaluation_seed),
        conversation_offset=coordinate["replicate"],
        device="cuda",
    )
    model_utils._set_topk(model, topk)
    probe = CSASelectionProbe()
    native = model(workload.input_ids, use_cache=False, csa_probe=probe)
    trace_id = f"{EXPERIMENT_ID}:{_digest(coordinate)[:16]}"
    identity_plan, evidence_plan, control_plan, receipts = build_counterfactual_plans(
        probe, workload, topk=topk, trace_id=trace_id
    )
    del probe
    identity = model(workload.input_ids, use_cache=False, selection_plan=identity_plan)
    if not torch.equal(native.logits, identity.logits):
        raise RuntimeError("Identity route replay changed model logits.")
    evidence = model(workload.input_ids, use_cache=False, selection_plan=evidence_plan)
    control = model(workload.input_ids, use_cache=False, selection_plan=control_plan)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": manifest_sha256,
        "coordinate": coordinate,
        "evaluation_seed": evaluation_seed,
        "checkpoint": checkpoint,
        "route_contract": {
            "equal_cardinality": True,
            "one_anchor_difference_per_layer_query": True,
            "identity_replay_bitwise_equal": True,
            "topk": topk,
            "receipts": receipts,
            "receipts_sha256": _digest(receipts),
        },
        "targets": workload.targets.cpu().tolist(),
        "outcomes": {
            "native": _outcome(native, workload),
            "identity-replay": _outcome(identity, workload),
            "force-evidence": _outcome(evidence, workload),
            "matched-control": _outcome(control, workload),
        },
    }


def _checkpoint(manifest: dict[str, Any], scale: str, seed: int) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["checkpoints"]
        if item["scale"] == scale and item["training_seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("Manifest checkpoint coverage drifted.")
    item = matches[0]
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or _file_digest(path) != item["sha256"]:
        raise ValueError(f"Checkpoint binding drifted: {path}")
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded causal-evidence route pilot.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The causal-evidence route pilot requires CUDA.")
    manifest = _load_manifest(args.manifest)
    manifest_sha256 = _file_digest(args.manifest)
    all_coordinates = coordinates(manifest)
    expected_paths = {cell_path(args.output_root, item) for item in all_coordinates}
    observed_paths = set(args.output_root.rglob("replicate-*.json"))
    if observed_paths - expected_paths:
        raise RuntimeError("Pilot output namespace contains an unexpected cell.")
    with acquire_gpu_lock(EXPERIMENT_ID):
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for coordinate in all_coordinates:
            grouped.setdefault(
                (coordinate["scale"], coordinate["training_seed"]), []
            ).append(coordinate)
        committed = 0
        for (scale, seed), items in grouped.items():
            checkpoint = _checkpoint(manifest, scale, seed)
            model = model_utils._load_model(Path(checkpoint["path"]))
            for coordinate in items:
                topk = int(manifest["fixed_topk"][scale]) * int(
                    coordinate["budget_multiplier"]
                )
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
                    payload = _run_cell(
                        model,
                        coordinate,
                        topk=topk,
                        evaluation_seed=_evaluation_seed(
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
            raise RuntimeError("Pilot completion cardinality drifted.")
        print(json.dumps({"status": "complete", "cells": committed}), flush=True)


if __name__ == "__main__":
    main()
