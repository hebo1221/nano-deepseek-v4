from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    CSASelectionProbe,
    DeepSeekV4ForCausalLM,
    ReplayQuery,
    RiskExample,
    build_probe_replay_queries,
    extract_request_risk_features,
    generate_adaptive_memory_workload,
)

SPLIT_NAMESPACES = {"train": 110_714_000, "calibration": 120_714_000}
TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
CONTEXTS = (80, 128, 256, 512, 1024)
NORMAL_TOPK_SWEEP = (1, 2, 4, 8)
EXAMPLES_PER_SHARD = 20
BATCH_SIZE = 4
REPLICATES = {"train": tuple(range(10)), "calibration": tuple(range(5))}
DESIGN = Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json")
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4/learned_lookahead.py",
    "nano_deepseek_v4/learned_memory_controller.py",
    "nano_deepseek_v4/memory_probe.py",
    "research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json",
    "research/adaptive_v4_memory/scripts/collect_p1_online_lookahead_labels.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    if set(IMPLEMENTATION_PATHS) != paths:
        raise RuntimeError("Online-lookahead label implementation is not fully tracked.")
    return hashlib.sha256(tree.encode()).hexdigest()


def source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty, "implementation_digest": implementation_digest()}


def generation_seed(
    *, split: str, training_seed: int, family: str, context: int, replicate: int
) -> int:
    if split not in SPLIT_NAMESPACES:
        raise ValueError("Unregistered learned-lookahead split.")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Unregistered checkpoint seed.")
    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError("Unregistered workload family.")
    if context not in CONTEXTS or replicate not in REPLICATES[split]:
        raise ValueError("Unregistered context or split replicate.")
    return (
        SPLIT_NAMESPACES[split]
        + TRAINING_SEEDS.index(training_seed) * 10_000_000
        + PAPER_GRADE_WORKLOAD_FAMILIES.index(family) * 1_000_000
        + context * 100
        + replicate
    )


def _input_digest(input_ids: torch.Tensor) -> str:
    values = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def build_label_rows(
    *,
    queries: tuple[ReplayQuery, ...],
    predictions: dict[int | str, torch.Tensor],
    query_columns: dict[int, int],
    conversation_ids: tuple[str, ...],
    input_ids: torch.Tensor,
    split: str,
    scale: str,
    training_seed: int,
    family: str,
    context: int,
    replicate: int,
    dense_topk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_control: dict[tuple[int, int], list[ReplayQuery]] = {}
    for query in queries:
        by_control.setdefault((query.batch_index, query.query_position), []).append(query)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch_index, conversation_id in enumerate(conversation_ids):
        sequence_digest = _input_digest(input_ids[batch_index])
        for query_position, column in sorted(query_columns.items()):
            feature_position = query_position - 1
            group = tuple(
                sorted(
                    by_control.get((batch_index, feature_position), ()),
                    key=lambda query: query.layer_index,
                )
            )
            if not group:
                failures.append(
                    {
                        "conversation_id": conversation_id,
                        "input_sha256": sequence_digest,
                        "query_position": query_position,
                        "reason": "no_prior_token_csa_probe",
                    }
                )
                continue
            dense_prediction = int(predictions["dense"][batch_index, column])
            matching = tuple(
                topk
                for topk in NORMAL_TOPK_SWEEP
                if int(predictions[topk][batch_index, column]) == dense_prediction
            )
            sufficient = min(matching) if matching else dense_topk
            example = RiskExample(
                group_id=(
                    f"{split}:{scale}:{training_seed}:{family}:{context}:{replicate}:"
                    f"{conversation_id}:q{query_position}"
                ),
                features=extract_request_risk_features(group),
                sufficient_topk=sufficient,
                dense_required=not matching,
            )
            rows.append(
                {
                    **asdict(example),
                    "conversation_id": conversation_id,
                    "input_sha256": sequence_digest,
                    "feature_token_position": feature_position,
                    "label_token_position": query_position,
                    "dense_prediction": dense_prediction,
                    "matching_registered_topk": matching,
                    "causal_offset": 1,
                }
            )
    return rows, failures


@torch.inference_mode()
def collect_batch(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    split: str,
    scale: str,
    training_seed: int,
    context: int,
    replicate: int,
    batch_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    dense_topk = max(
        context // model.config.compress_rates["compressed_sparse_attention"],
        max(NORMAL_TOPK_SWEEP),
    )
    query_columns = heldout._query_columns(workload)
    predicted: dict[int | str, torch.Tensor] = {}
    for topk in (*NORMAL_TOPK_SWEEP, dense_topk):
        pilot._set_topk(model, topk)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _, _ = model.model(workload.input_ids)
            logits = model.lm_head(hidden)
        values = torch.empty_like(workload.targets)
        for position, column in query_columns.items():
            values[:, column] = logits[:, position].argmax(dim=-1)
        predicted["dense" if topk == dense_topk else topk] = values.cpu()

    pilot._set_topk(model, model.config.index_topk)
    probe = CSASelectionProbe()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model.model(workload.input_ids, csa_probe=probe)
    block_bytes = (model.config.head_dim + model.config.index_head_dim) * 2 + 16
    queries = build_probe_replay_queries(
        probe,
        trace_id=(
            f"p1-online-label:{split}:{scale}:{training_seed}:{workload.family}:"
            f"{context}:{replicate}:{batch_offset}"
        ),
        request_id=workload.conversation_ids[0],
        native_topk=model.config.index_topk,
        block_bytes=block_bytes,
    )
    rows, failures = build_label_rows(
        queries=queries,
        predictions=predicted,
        query_columns=query_columns,
        conversation_ids=workload.conversation_ids,
        input_ids=workload.input_ids,
        split=split,
        scale=scale,
        training_seed=training_seed,
        family=workload.family,
        context=context,
        replicate=replicate,
        dense_topk=dense_topk,
    )
    return rows, failures, dense_topk


def collect(args: argparse.Namespace, model: DeepSeekV4ForCausalLM) -> dict[str, Any]:
    seed = generation_seed(
        split=args.split,
        training_seed=args.training_seed,
        family=args.family,
        context=args.context,
        replicate=args.replicate,
    )
    generator = torch.Generator().manual_seed(seed)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dense_topks: set[int] = set()
    for offset in range(0, EXAMPLES_PER_SHARD, BATCH_SIZE):
        workload = generate_adaptive_memory_workload(
            task,
            family=args.family,
            batch_size=BATCH_SIZE,
            sequence_length=args.context,
            generator=generator,
            conversation_offset=args.replicate * EXAMPLES_PER_SHARD + offset,
            device="cuda",
        )
        batch_rows, batch_failures, dense_topk = collect_batch(
            model,
            workload,
            split=args.split,
            scale=args.scale,
            training_seed=args.training_seed,
            context=args.context,
            replicate=args.replicate,
            batch_offset=offset,
        )
        rows.extend(batch_rows)
        failures.extend(batch_failures)
        dense_topks.add(dense_topk)
    rows.sort(key=lambda row: row["group_id"])
    failures.sort(key=lambda row: (row["conversation_id"], row["query_position"]))
    source = source_state()
    if source["dirty"] is not False:
        raise RuntimeError("Online-lookahead label collection requires a clean source tree.")
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-label-shard-v1",
        "split": args.split,
        "scale": args.scale,
        "training_seed": args.training_seed,
        "family": args.family,
        "context": args.context,
        "replicate": args.replicate,
        "generation_seed": seed,
        "conversations": EXAMPLES_PER_SHARD,
        "risk_examples": len(rows),
        "failures": failures,
        "failure_count": len(failures),
        "registered_topk_sweep": NORMAL_TOPK_SWEEP,
        "dense_topk": tuple(sorted(dense_topks)),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "design": {"path": str(DESIGN), "sha256": sha256(DESIGN)},
        "rows_digest": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
        "source": source,
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "features_use_prior_token_only": True,
            "causal_offset": 1,
            "benchmark_targets_used_for_labels": False,
            "dense_model_predictions_used_for_labels": True,
            "split_namespace": SPLIT_NAMESPACES[args.split],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one digest-bound online learned-lookahead label shard."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_NAMESPACES), required=True)
    parser.add_argument("--family", choices=PAPER_GRADE_WORKLOAD_FAMILIES, required=True)
    parser.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicate not in REPLICATES[args.split]:
        raise ValueError("Replicate is outside the frozen split.")
    if not torch.cuda.is_available():
        raise RuntimeError("Online-lookahead label collection requires CUDA.")
    design = json.loads(DESIGN.read_text())
    if (
        design.get("experiment_id") != "p1-online-learned-lookahead-v1"
        or design.get("status") != "frozen_before_execution"
    ):
        raise RuntimeError("The frozen online-lookahead protocol is required.")
    payload = collect(args, pilot._load_model(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "risk_examples": payload["risk_examples"],
                "failures": payload["failure_count"],
                "rows_digest": payload["rows_digest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
