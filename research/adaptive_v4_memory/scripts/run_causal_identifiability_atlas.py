from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Sequence
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
from nano_deepseek_v4.modeling import apply_partial_rope, rope_cos_sin

EXPERIMENT_ID = "causal-identifiability-atlas-e1-v2-1"
OBSERVABLE_POLICIES = (
    "full-read",
    "perturbation-proxy",
    "recency",
    "value-norm",
    "random",
)
ARMS = ("native", "identity-replay", "force-evidence", *OBSERVABLE_POLICIES)
FAMILIES = (
    "single-remote-retrieval",
    "adversarial-lexical-distractors",
    "long-generation-changing-evidence",
    "multiple-independent-needles",
)
EXHAUSTIVE_FAMILIES = (
    "single-remote-retrieval",
    "adversarial-lexical-distractors",
)


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


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(_canonical(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes())
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


def coordinates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cohort in manifest["design"]:
        for seed in cohort["training_seeds"]:
            for context in cohort["contexts"]:
                for family in cohort["families"]:
                    for replicate in cohort["replicates"]:
                        result.append(
                            {
                                "scale": cohort["scale"],
                                "training_seed": seed,
                                "budget_multiplier": cohort["budget_multiplier"],
                                "context": context,
                                "family": family,
                                "replicate": replicate,
                            }
                        )
    if len({_digest(item) for item in result}) != len(result):
        raise ValueError("Atlas design contains duplicate coordinates.")
    return result


def is_exhaustive_coordinate(coordinate: dict[str, Any]) -> bool:
    return coordinate["scale"] == "s151" and coordinate["family"] in EXHAUSTIVE_FAMILIES


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


def workload_identity(coordinate: dict[str, Any]) -> dict[str, Any]:
    """Return the task identity shared across checkpoint seeds and model scales."""
    return {
        "context": coordinate["context"],
        "family": coordinate["family"],
        "replicate": coordinate["replicate"],
    }


def evaluation_seed(coordinate: dict[str, Any], namespace: str) -> int:
    value = _digest({"namespace": namespace, "workload": workload_identity(coordinate)})
    return int(value[:16], 16) % (2**63 - 1)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected causal-identifiability manifest.")
    items = coordinates(payload)
    exhaustive = sum(is_exhaustive_coordinate(item) for item in items)
    if payload.get("expected_cells") != len(items) or len(items) != 160:
        raise ValueError("The frozen atlas must contain exactly 160 cells.")
    if payload.get("expected_exhaustive_cells") != exhaustive or exhaustive != 60:
        raise ValueError("The frozen atlas must contain exactly 60 exhaustive cells.")
    if tuple(payload.get("arms", ())) != ARMS:
        raise ValueError("Atlas arm contract drifted.")
    if tuple(payload.get("families", ())) != FAMILIES:
        raise ValueError("Atlas family contract drifted.")
    implementation = payload.get("implementation", {})
    bindings = (
        ("runner_sha256", Path(__file__)),
        ("analyzer_sha256", Path(implementation["analyzer_path"])),
        ("preregistration_sha256", Path(implementation["preregistration_path"])),
        ("runtime_amendment_sha256", Path(implementation["runtime_amendment_path"])),
        ("preflight_amendment_sha256", Path(implementation["preflight_amendment_path"])),
    )
    for key, implementation_path in bindings:
        if implementation.get(key) != _file_digest(implementation_path):
            raise ValueError(f"Atlas implementation binding drifted: {key}.")
    if payload.get("quality_values_accessed_before_freeze") is not False:
        raise ValueError("Atlas manifest does not attest a pre-outcome freeze.")
    if payload.get("counterfactual_batch_size") != 8:
        raise ValueError("Atlas counterfactual batch size drifted.")
    return payload


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


def _eligible_indices(record: Any, batch_index: int, query_index: int) -> tuple[int, ...]:
    scores = record.scores[batch_index, query_index]
    return tuple(
        index for index, score in enumerate(scores.detach().float().cpu().tolist())
        if math.isfinite(score)
    )


def _stable_random_score(
    *, workload: dict[str, Any], layer_index: int, query_position: int, block_end: int
) -> float:
    identity = {
        "namespace": "causal-identifiability-atlas-random-v1",
        "workload": workload,
        "layer_index": layer_index,
        "query_position": query_position,
        "block_end": block_end,
    }
    return int(_digest(identity)[:16], 16) / float(2**64 - 1)


def _top_indices(scores: Sequence[float], eligible: Sequence[int], topk: int) -> tuple[int, ...]:
    if len(eligible) < topk:
        raise ValueError("Query has fewer causal candidates than the fixed route cardinality.")
    if any(not math.isfinite(float(scores[index])) for index in eligible):
        raise ValueError("Observable score is non-finite on a causal candidate.")
    return tuple(
        sorted(eligible, key=lambda index: (-float(scores[index]), index))[:topk]
    )


def _perturbation_scores(
    model: Any,
    record: Any,
    *,
    batch_index: int,
    query_index: int,
    query_position: int,
    eligible: tuple[int, ...],
) -> list[float]:
    if record.read_scores is None:
        raise ValueError("Perturbation proxy requires full compressed read scores.")
    attention = model.model.layers[record.layer_index].self_attn
    if attention.csa is None:
        raise ValueError("Probe record does not identify a CSA layer.")
    read = record.read_scores[batch_index, query_index].detach().float()
    indices = torch.tensor(eligible, dtype=torch.long, device=read.device)
    probabilities = read.index_select(0, indices).softmax(dim=0)

    values = record.value_blocks[batch_index : batch_index + 1].detach()
    ends = record.block_end_positions[batch_index : batch_index + 1].detach()
    starts = ends - attention.csa.rate + 1
    cos, sin = rope_cos_sin(
        starts,
        attention.csa.rope_dim,
        attention.csa.compress_rope_theta,
    )
    rotated = apply_partial_rope(values.unsqueeze(1), cos, sin).squeeze(0).squeeze(0)
    selected_values = rotated.index_select(0, indices)
    center = (probabilities.unsqueeze(-1) * selected_values.float()).sum(dim=0)
    odds = probabilities / (1.0 - probabilities).clamp_min(1e-6)
    deltas = odds.unsqueeze(-1) * (selected_values.float() - center)

    expanded = deltas.unsqueeze(1).expand(-1, attention.num_heads, -1).unsqueeze(2)
    query_positions = torch.full(
        (len(eligible), 1), query_position, dtype=torch.long, device=read.device
    )
    query_cos, query_sin = rope_cos_sin(
        query_positions,
        attention.rope_dim,
        attention.config.rope_theta,
    )
    expanded = apply_partial_rope(expanded, query_cos, -query_sin)
    flattened = expanded.squeeze(2).reshape(len(eligible), -1)
    projected = attention.o_b_proj(
        attention.o_a_proj(flattened.to(attention.o_a_proj.weight.dtype))
    )
    selected_scores = projected.float().norm(dim=-1).cpu().tolist()
    result = [float("-inf")] * record.scores.shape[-1]
    for index, score in zip(eligible, selected_scores, strict=True):
        result[index] = float(score)
    return result


def _score_table(
    model: Any,
    record: Any,
    *,
    batch_index: int,
    query_index: int,
    query_position: int,
    workload: dict[str, Any],
) -> dict[str, list[float]]:
    eligible = _eligible_indices(record, batch_index, query_index)
    if not eligible:
        raise ValueError("Atlas query has no causal compressed candidate.")
    if record.read_scores is None:
        raise ValueError("Atlas probe is missing full read scores.")
    native = record.scores[batch_index, query_index].detach().float().cpu().tolist()
    read = record.read_scores[batch_index, query_index].detach().float().cpu().tolist()
    ends = record.block_end_positions[batch_index].detach().long().cpu().tolist()
    norms = record.value_blocks[batch_index].detach().float().norm(dim=-1).cpu().tolist()
    table = {
        "native": [float(value) for value in native],
        "full-read": [float(value) for value in read],
        "perturbation-proxy": _perturbation_scores(
            model,
            record,
            batch_index=batch_index,
            query_index=query_index,
            query_position=query_position,
            eligible=eligible,
        ),
        "recency": [float(value) for value in ends],
        "value-norm": [float(value) for value in norms],
        "random": [
            _stable_random_score(
                workload=workload,
                layer_index=record.layer_index,
                query_position=query_position,
                block_end=int(block_end),
            )
            for block_end in ends
        ],
    }
    for scores in table.values():
        for index in range(len(scores)):
            if index not in eligible:
                scores[index] = float("-inf")
    return table


def build_observable_routes(
    model: Any,
    probe: CSASelectionProbe,
    *,
    query_positions: torch.Tensor,
    topk: int,
    trace_id: str,
    workload: dict[str, Any],
    capture_candidates: bool,
) -> tuple[CSASelectionPlan, dict[str, CSASelectionPlan], list[dict[str, Any]], str]:
    """Build routes without accepting targets or evidence positions."""
    if topk <= 0 or not probe.records or query_positions.ndim != 2:
        raise ValueError("Observable route construction requires a probe and positive top-k.")
    identity_rows: list[PlannedCSASelection] = []
    policy_rows: dict[str, list[PlannedCSASelection]] = {
        policy: [] for policy in OBSERVABLE_POLICIES
    }
    receipts: list[dict[str, Any]] = []
    for record in probe.records:
        for batch_index in range(query_positions.shape[0]):
            for query_column in range(query_positions.shape[1]):
                query_position = int(query_positions[batch_index, query_column])
                query_index = _query_index(record, batch_index, query_position)
                eligible = _eligible_indices(record, batch_index, query_index)
                scores = _score_table(
                    model,
                    record,
                    batch_index=batch_index,
                    query_index=query_index,
                    query_position=query_position,
                    workload=workload,
                )
                native_tensor = record.scores[batch_index, query_index].detach()
                native_indices = tuple(
                    int(index) for index in native_tensor.topk(topk).indices.cpu().tolist()
                )
                if any(index not in eligible for index in native_indices):
                    raise ValueError("Native route selected a non-causal candidate.")
                ends = record.block_end_positions[batch_index].detach().long().cpu()
                identity_rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=tuple(int(ends[index]) for index in native_indices),
                    )
                )
                selected: dict[str, list[int]] = {
                    "native": [int(ends[index]) for index in native_indices]
                }
                for policy in OBSERVABLE_POLICIES:
                    indices = _top_indices(scores[policy], eligible, topk)
                    selected[policy] = [int(ends[index]) for index in indices]
                    policy_rows[policy].append(
                        PlannedCSASelection(
                            layer_index=record.layer_index,
                            batch_index=batch_index,
                            query_position=query_position,
                            block_end_positions=tuple(selected[policy]),
                        )
                    )
                candidate_rows = None
                if capture_candidates:
                    candidate_rows = [
                        {
                            "block_index": index,
                            "block_end": int(ends[index]),
                            "scores": {
                                name: float(score_values[index])
                                for name, score_values in scores.items()
                            },
                        }
                        for index in eligible
                    ]
                receipts.append(
                    {
                        "layer_index": record.layer_index,
                        "batch_index": batch_index,
                        "query_column": query_column,
                        "query_position": query_position,
                        "eligible_blocks": len(eligible),
                        "selected_block_ends": selected,
                        "score_table_sha256": _digest(
                            {
                                name: [float(score_values[index]) for index in eligible]
                                for name, score_values in scores.items()
                            }
                        ),
                        "candidates": candidate_rows,
                    }
                )
    route_bundle = {
        "trace_id": trace_id,
        "topk": topk,
        "routes": [
            {
                "layer_index": row["layer_index"],
                "batch_index": row["batch_index"],
                "query_position": row["query_position"],
                "eligible_blocks": row["eligible_blocks"],
                "selected_block_ends": row["selected_block_ends"],
                "score_table_sha256": row["score_table_sha256"],
            }
            for row in receipts
        ],
    }
    return (
        CSASelectionPlan(trace_id, "identity-replay", tuple(identity_rows)),
        {
            policy: CSASelectionPlan(trace_id, policy, tuple(rows))
            for policy, rows in policy_rows.items()
        },
        receipts,
        _digest(route_bundle),
    )


def build_evidence_oracle(
    probe: CSASelectionProbe,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    topk: int,
    trace_id: str,
) -> tuple[CSASelectionPlan, list[dict[str, Any]]]:
    rows: list[PlannedCSASelection] = []
    receipts: list[dict[str, Any]] = []
    for record in probe.records:
        for batch_index in range(workload.query_positions.shape[0]):
            for query_column in range(workload.query_positions.shape[1]):
                query_position = int(workload.query_positions[batch_index, query_column])
                evidence_position = int(workload.evidence_positions[batch_index, query_column])
                query_index = _query_index(record, batch_index, query_position)
                eligible = _eligible_indices(record, batch_index, query_index)
                ends = record.block_end_positions[batch_index].detach().long().cpu()
                evidence_index = _evidence_index(ends, evidence_position, query_position)
                if evidence_index not in eligible:
                    raise ValueError("Task evidence is not a causal candidate.")
                native_scores = record.scores[batch_index, query_index].detach()
                native = tuple(
                    int(index) for index in native_scores.topk(topk).indices.cpu().tolist()
                )
                common = tuple(index for index in native if index != evidence_index)[: topk - 1]
                if len(common) < topk - 1:
                    remainder = sorted(
                        (index for index in eligible if index not in common and index != evidence_index),
                        key=lambda index: (-float(native_scores[index]), index),
                    )
                    common = (*common, *remainder[: topk - 1 - len(common)])
                selected = (*common, evidence_index)
                if len(set(selected)) != topk:
                    raise ValueError("Evidence oracle failed exact route cardinality.")
                selected_ends = tuple(int(ends[index]) for index in selected)
                rows.append(
                    PlannedCSASelection(
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        block_end_positions=selected_ends,
                    )
                )
                receipts.append(
                    {
                        "layer_index": record.layer_index,
                        "batch_index": batch_index,
                        "query_column": query_column,
                        "query_position": query_position,
                        "evidence_position": evidence_position,
                        "evidence_block_end": int(ends[evidence_index]),
                        "native_contains_evidence": evidence_index in native,
                        "selected_block_ends": list(selected_ends),
                    }
                )
    return CSASelectionPlan(trace_id, "force-evidence", tuple(rows)), receipts


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
    target_log_prob = log_probabilities.gather(-1, workload.targets.unsqueeze(-1)).squeeze(-1)
    predictions = logits.argmax(dim=-1)
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": predictions.eq(workload.targets).cpu().tolist(),
        "target_log_prob": target_log_prob.cpu().tolist(),
    }


def _batch_plan(
    *,
    trace_id: str,
    request_id: str,
    layer_index: int,
    query_position: int,
    selections: Sequence[Sequence[int]],
) -> CSASelectionPlan:
    return CSASelectionPlan(
        trace_id,
        request_id,
        tuple(
            PlannedCSASelection(
                layer_index=layer_index,
                batch_index=batch_index,
                query_position=query_position,
                block_end_positions=tuple(int(value) for value in ends),
            )
            for batch_index, ends in enumerate(selections)
        ),
    )


def _pad_candidate_chunk(chunk: Sequence[int], batch_size: int) -> tuple[int, ...]:
    if not chunk or len(chunk) > batch_size:
        raise ValueError("Candidate chunk must contain between one and batch-size items.")
    values = tuple(int(value) for value in chunk)
    return (*values, *((values[-1],) * (batch_size - len(values))))


def _counterfactual_chunk_selections(
    all_ends: Sequence[int], real_chunk: Sequence[int], batch_size: int
) -> dict[str, tuple[tuple[int, ...], ...]]:
    full = tuple(int(value) for value in all_ends)
    padded = _pad_candidate_chunk(real_chunk, batch_size)
    return {
        "candidate": tuple((end,) for end in padded),
        "core": ((),) * batch_size,
        "full": (full,) * batch_size,
        "deletion": tuple(tuple(value for value in full if value != end) for end in padded),
    }


def _selected_query_logits(output: Any, query_position: int) -> torch.Tensor:
    return output.logits[:, query_position].float()


def _token_log_probabilities(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return logits.log_softmax(dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def _exhaustive_atlas(
    model: Any,
    workload: AdaptiveMemoryWorkloadBatch,
    receipts: list[dict[str, Any]],
    oracle_receipts: list[dict[str, Any]],
    *,
    trace_id: str,
    batch_size: int,
) -> dict[str, Any]:
    if workload.query_positions.shape != (1, 1):
        raise ValueError("Exhaustive atlas requires exactly one query.")
    query_position = int(workload.query_positions[0, 0])
    gold_token = int(workload.targets[0, 0])
    oracle_by_layer = {row["layer_index"]: row for row in oracle_receipts}
    layer_results: list[dict[str, Any]] = []
    for receipt in receipts:
        if receipt["query_column"] != 0 or receipt["batch_index"] != 0:
            raise ValueError("Exhaustive receipt axis drifted.")
        candidates = receipt["candidates"]
        if not candidates:
            raise ValueError("Exhaustive observable candidate table is absent.")
        layer_index = int(receipt["layer_index"])
        all_ends = tuple(int(item["block_end"]) for item in candidates)
        evidence_end = int(oracle_by_layer[layer_index]["evidence_block_end"])
        candidate_results: list[dict[str, Any]] = []
        teacher_token: int | None = None
        for start in range(0, len(all_ends), batch_size):
            real_chunk = all_ends[start : start + batch_size]
            route_selections = _counterfactual_chunk_selections(
                all_ends, real_chunk, batch_size
            )
            repeated = workload.input_ids.repeat(batch_size, 1)
            candidate_plan = _batch_plan(
                trace_id=trace_id,
                request_id=f"candidate:{layer_index}:{start}",
                layer_index=layer_index,
                query_position=query_position,
                selections=route_selections["candidate"],
            )
            core_plan = _batch_plan(
                trace_id=trace_id,
                request_id=f"core:{layer_index}:{start}",
                layer_index=layer_index,
                query_position=query_position,
                selections=route_selections["core"],
            )
            full_plan = _batch_plan(
                trace_id=trace_id,
                request_id=f"full:{layer_index}:{start}",
                layer_index=layer_index,
                query_position=query_position,
                selections=route_selections["full"],
            )
            deletion_plan = _batch_plan(
                trace_id=trace_id,
                request_id=f"deletion:{layer_index}:{start}",
                layer_index=layer_index,
                query_position=query_position,
                selections=route_selections["deletion"],
            )
            candidate_logits = _selected_query_logits(
                model(repeated, use_cache=False, selection_plan=candidate_plan), query_position
            )
            core_logits = _selected_query_logits(
                model(repeated, use_cache=False, selection_plan=core_plan), query_position
            )
            full_logits = _selected_query_logits(
                model(repeated, use_cache=False, selection_plan=full_plan), query_position
            )
            deletion_logits = _selected_query_logits(
                model(repeated, use_cache=False, selection_plan=deletion_plan), query_position
            )
            if not all(torch.equal(full_logits[0], row) for row in full_logits[1:]):
                raise RuntimeError("Repeated full-memory rows diverged inside a batch.")
            if not all(torch.equal(core_logits[0], row) for row in core_logits[1:]):
                raise RuntimeError("Repeated core rows diverged inside a batch.")
            chunk_teacher = int(full_logits[0].argmax())
            if teacher_token is None:
                teacher_token = chunk_teacher
            elif teacher_token != chunk_teacher:
                raise RuntimeError("Full-memory teacher token changed across candidate chunks.")
            gold = torch.full(
                (batch_size,), gold_token, dtype=torch.long, device=full_logits.device
            )
            teacher = torch.full(
                (batch_size,), teacher_token, dtype=torch.long, device=full_logits.device
            )
            candidate_gold = _token_log_probabilities(candidate_logits, gold)
            core_gold = _token_log_probabilities(core_logits, gold)
            full_gold = _token_log_probabilities(full_logits, gold)
            deletion_gold = _token_log_probabilities(deletion_logits, gold)
            candidate_teacher = _token_log_probabilities(candidate_logits, teacher)
            core_teacher = _token_log_probabilities(core_logits, teacher)
            full_teacher = _token_log_probabilities(full_logits, teacher)
            deletion_teacher = _token_log_probabilities(deletion_logits, teacher)
            for offset, block_end in enumerate(real_chunk):
                source = candidates[start + offset]
                candidate_results.append(
                    {
                        **source,
                        "is_task_evidence": block_end == evidence_end,
                        "equal_budget_gold_log_prob_delta": float(
                            candidate_gold[offset] - core_gold[offset]
                        ),
                        "equal_budget_teacher_log_prob_delta": float(
                            candidate_teacher[offset] - core_teacher[offset]
                        ),
                        "necessity_gold_log_prob_loss": float(
                            full_gold[offset] - deletion_gold[offset]
                        ),
                        "necessity_teacher_log_prob_loss": float(
                            full_teacher[offset] - deletion_teacher[offset]
                        ),
                        "candidate_gold_correct": bool(
                            int(candidate_logits[offset].argmax()) == gold_token
                        ),
                    }
                )
            del candidate_logits, core_logits, full_logits, deletion_logits, repeated
        if teacher_token is None or len(candidate_results) != len(all_ends):
            raise RuntimeError("Exhaustive atlas failed candidate cardinality.")
        if sum(bool(item["is_task_evidence"]) for item in candidate_results) != 1:
            raise RuntimeError("Exhaustive atlas did not identify exactly one evidence block.")
        layer_results.append(
            {
                "layer_index": layer_index,
                "query_position": query_position,
                "gold_token": gold_token,
                "full_memory_teacher_token": teacher_token,
                "candidate_count": len(candidate_results),
                "core_cardinality": 0,
                "candidate_cardinality": 1,
                "full_cardinality": len(candidate_results),
                "deletion_cardinality": len(candidate_results) - 1,
                "observable_candidates_sha256": _digest(candidates),
                "candidates": candidate_results,
            }
        )
    return {
        "schema_version": 1,
        "counterfactual_batch_size": batch_size,
        "fixed_batch_shape_with_discarded_repeat_last_padding": True,
        "layers": layer_results,
        "layers_sha256": _digest(layer_results),
    }


@torch.inference_mode()
def run_cell(
    model: Any,
    coordinate: dict[str, Any],
    *,
    topk: int,
    evaluation_seed_value: int,
    manifest_sha256: str,
    checkpoint: dict[str, Any],
    counterfactual_batch_size: int,
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
    model_utils._set_topk(model, topk)
    probe = CSASelectionProbe()
    native = model(workload.input_ids, use_cache=False, csa_probe=probe)
    trace_id = f"{EXPERIMENT_ID}:{_digest(coordinate)[:16]}"
    identity_plan, policy_plans, observable_receipts, observable_digest = (
        build_observable_routes(
            model,
            probe,
            query_positions=workload.query_positions,
            topk=topk,
            trace_id=trace_id,
            workload=workload_identity(coordinate),
            capture_candidates=is_exhaustive_coordinate(coordinate),
        )
    )
    identity = model(workload.input_ids, use_cache=False, selection_plan=identity_plan)
    if not torch.equal(native.logits, identity.logits):
        raise RuntimeError("Identity route replay changed model logits.")
    outcomes = {
        "native": _outcome(native, workload),
        "identity-replay": _outcome(identity, workload),
    }
    for policy, plan in policy_plans.items():
        output = model(workload.input_ids, use_cache=False, selection_plan=plan)
        outcomes[policy] = _outcome(output, workload)
        del output

    # Task evidence is opened only after the complete observable route bundle is hashed.
    oracle_plan, oracle_receipts = build_evidence_oracle(
        probe, workload, topk=topk, trace_id=trace_id
    )
    oracle = model(workload.input_ids, use_cache=False, selection_plan=oracle_plan)
    outcomes["force-evidence"] = _outcome(oracle, workload)
    atlas = (
        _exhaustive_atlas(
            model,
            workload,
            observable_receipts,
            oracle_receipts,
            trace_id=trace_id,
            batch_size=counterfactual_batch_size,
        )
        if is_exhaustive_coordinate(coordinate)
        else None
    )
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
            "topk": topk,
            "identity_replay_bitwise_equal": True,
            "observable_routes_exclude_evidence_and_targets": True,
            "observable_route_bundle_sha256": observable_digest,
            "observable_receipts": observable_receipts,
            "observable_receipts_sha256": _digest(observable_receipts),
            "oracle_receipts": oracle_receipts,
            "oracle_receipts_sha256": _digest(oracle_receipts),
        },
        "targets": workload.targets.cpu().tolist(),
        "outcomes": outcomes,
        "exhaustive_atlas": atlas,
    }


def _validate_preflight_payload(payload: dict[str, Any], *, exhaustive: bool) -> dict[str, Any]:
    route = payload["route_contract"]
    receipts = route["observable_receipts"]
    topk = int(route["topk"])
    if not route["identity_replay_bitwise_equal"]:
        raise RuntimeError("Preflight identity invariant failed.")
    for receipt in receipts:
        if any(
            len(receipt["selected_block_ends"][arm]) != topk
            for arm in ("native", *OBSERVABLE_POLICIES)
        ):
            raise RuntimeError("Preflight route cardinality failed.")
    atlas = payload["exhaustive_atlas"]
    if exhaustive != (atlas is not None):
        raise RuntimeError("Preflight exhaustive-atlas status drifted.")
    return {
        "status": "preflight_passed",
        "coordinate": payload["coordinate"],
        "input_ids_sha256": payload["input_ids_sha256"],
        "observable_route_bundle_sha256": route["observable_route_bundle_sha256"],
        "route_receipts": len(receipts),
        "exhaustive_layers": len(atlas["layers"]) if atlas is not None else 0,
        "quality_values_disclosed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen causal-identifiability atlas.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-coordinate-index", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The causal-identifiability atlas requires CUDA.")
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
                topk=int(manifest["fixed_topk"][coordinate["scale"]])
                * int(coordinate["budget_multiplier"]),
                evaluation_seed_value=evaluation_seed(
                    coordinate, manifest["evaluation_seed_namespace"]
                ),
                manifest_sha256=manifest_sha256,
                checkpoint=checkpoint,
                counterfactual_batch_size=manifest["counterfactual_batch_size"],
            )
            print(
                json.dumps(
                    _validate_preflight_payload(
                        payload, exhaustive=is_exhaustive_coordinate(coordinate)
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        return
    if args.output_root is None:
        raise ValueError("Normal execution requires --output-root.")
    if args.output_root.resolve() != Path(manifest["output_root"]).resolve():
        raise ValueError("Output root differs from the frozen manifest.")
    expected_paths = {cell_path(args.output_root, item) for item in all_coordinates}
    observed_paths = set(args.output_root.rglob("replicate-*.json"))
    if observed_paths - expected_paths:
        raise RuntimeError("Atlas output namespace contains an unexpected cell.")
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
                        topk=int(manifest["fixed_topk"][scale])
                        * int(coordinate["budget_multiplier"]),
                        evaluation_seed_value=evaluation_seed(
                            coordinate, manifest["evaluation_seed_namespace"]
                        ),
                        manifest_sha256=manifest_sha256,
                        checkpoint=checkpoint,
                        counterfactual_batch_size=manifest["counterfactual_batch_size"],
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
            raise RuntimeError("Atlas completion cardinality drifted.")
        print(json.dumps({"status": "complete", "cells": committed}), flush=True)


if __name__ == "__main__":
    main()
