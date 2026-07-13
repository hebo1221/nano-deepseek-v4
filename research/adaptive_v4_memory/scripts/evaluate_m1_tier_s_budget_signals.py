from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    CSASelectionProbe,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    ReplayFeatureRow,
    analyze_budget_signals,
    generate_associative_recall_batch,
)

SEQUENCE_LENGTHS = (48, 64, 80)
TOPK_BUDGETS = (0, 1, 2, 4, 8, 16, 64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint_path: Path, device: torch.device) -> DeepSeekV4ForCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = DeepSeekV4Config(**checkpoint["config"])
    model = DeepSeekV4ForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def _set_index_topk(model: DeepSeekV4ForCausalLM, topk: int) -> None:
    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = topk


def _top_p_cardinality(probabilities: torch.Tensor, threshold: float) -> int:
    cumulative = probabilities.cumsum(dim=0)
    return int(torch.searchsorted(cumulative, threshold).clamp_max(len(probabilities) - 1)) + 1


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _probe_feature_rows(
    probe: CSASelectionProbe,
    *,
    scale: str,
    sequence_length: int,
    example_offset: int,
    native_budget: int,
) -> list[ReplayFeatureRow]:
    rows: list[ReplayFeatureRow] = []
    batch_size = probe.records[0].scores.shape[0]
    prior_by_example: list[set[int]] = [set() for _ in range(batch_size)]
    for record in probe.records:
        for batch_index in range(batch_size):
            finite_scores = record.scores[batch_index, -1]
            finite_scores = finite_scores[torch.isfinite(finite_scores)].float()
            ordered_scores, ordered_indices = finite_scores.sort(descending=True)
            probabilities = ordered_scores.softmax(dim=0)
            selected_count = min(native_budget, len(ordered_indices))
            selected = set(ordered_indices[:selected_count].tolist())
            previous = prior_by_example[batch_index]
            boundary_margin = 0.0
            if 0 < selected_count < len(ordered_scores):
                boundary_margin = float(
                    ordered_scores[selected_count - 1] - ordered_scores[selected_count]
                )
            rows.append(
                ReplayFeatureRow(
                    trace_id=f"m1-tier-s-{scale}",
                    request_id=f"length-{sequence_length}",
                    layer_index=record.layer_index,
                    batch_index=example_offset + batch_index,
                    query_position=sequence_length - 1,
                    context_blocks=len(ordered_scores),
                    native_budget=selected_count,
                    score_entropy=float(
                        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
                    ),
                    top1_probability=float(probabilities[0]),
                    top50_cardinality=_top_p_cardinality(probabilities, 0.5),
                    top90_cardinality=_top_p_cardinality(probabilities, 0.9),
                    top95_cardinality=_top_p_cardinality(probabilities, 0.95),
                    boundary_margin=boundary_margin,
                    score_mean=float(ordered_scores.mean()),
                    score_std=float(ordered_scores.std(unbiased=False)),
                    temporal_jaccard=0.0,
                    cross_layer_jaccard=_jaccard(previous, selected),
                )
            )
            prior_by_example[batch_index] = selected
    return rows


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Tier-S signal evaluation requires a CUDA device.")
    device = torch.device("cuda")
    model = _load_model(args.checkpoint, device)
    task_config = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    generator = torch.Generator().manual_seed(args.seed)
    rows: list[ReplayFeatureRow] = []
    labels: list[int] = []
    accuracy_counts = {budget: 0 for budget in TOPK_BUDGETS}
    evaluated_examples = 0
    dense_fallback_examples = 0
    started = time.time()

    for sequence_length in SEQUENCE_LENGTHS:
        for _ in range(args.batches_per_length):
            batch = generate_associative_recall_batch(
                task_config,
                batch_size=args.batch_size,
                sequence_length=sequence_length,
                generator=generator,
                device=device,
            )
            correct_by_budget: dict[int, torch.Tensor] = {}
            probe = CSASelectionProbe()
            for budget in TOPK_BUDGETS:
                _set_index_topk(model, budget)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if budget == model.config.index_topk:
                        hidden, _, _ = model.model(batch.input_ids, csa_probe=probe)
                    else:
                        hidden, _, _ = model.model(batch.input_ids)
                    predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
                correct = predictions.eq(batch.targets)
                correct_by_budget[budget] = correct
                accuracy_counts[budget] += int(correct.sum())

            example_labels = []
            maximum_blocks = (
                sequence_length // model.config.compress_rates["compressed_sparse_attention"]
            )
            for batch_index in range(args.batch_size):
                sufficient = next(
                    (
                        min(budget, maximum_blocks)
                        for budget in TOPK_BUDGETS
                        if bool(correct_by_budget[budget][batch_index])
                    ),
                    maximum_blocks + 1,
                )
                example_labels.append(sufficient)
                if sufficient == maximum_blocks + 1:
                    dense_fallback_examples += 1
            feature_rows = _probe_feature_rows(
                probe,
                scale=args.scale,
                sequence_length=sequence_length,
                example_offset=evaluated_examples,
                native_budget=model.config.index_topk,
            )
            layer_count = len(probe.records)
            rows.extend(feature_rows)
            labels.extend(label for _ in range(layer_count) for label in example_labels)
            evaluated_examples += args.batch_size

    report = analyze_budget_signals(
        rows,
        labels,
        minimum_relative_improvement=args.minimum_relative_improvement,
        ridge=args.ridge,
    )
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    return {
        "schema_version": 1,
        "experiment_id": "m1-tier-s-budget-signal-v1",
        "scale": args.scale,
        "seed": args.seed,
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "bytes": args.checkpoint.stat().st_size,
        },
        "protocol": {
            "sequence_lengths": SEQUENCE_LENGTHS,
            "topk_budgets": TOPK_BUDGETS,
            "batches_per_length": args.batches_per_length,
            "batch_size": args.batch_size,
            "quality": "exact-match associative recall",
            "label": "minimum shared per-CSA-layer top-k with an exact match; max blocks + 1 if none",
            "validation": "leave-one-generated-example-group-out ridge regression",
            "baseline_features": ["context_blocks", "layer_identity"],
            "native_features": [
                "score_entropy",
                "top1_probability",
                "top-p_cardinality",
                "boundary_margin",
                "score_mean_std",
                "cross_layer_jaccard",
            ],
        },
        "examples": evaluated_examples,
        "feature_rows": len(rows),
        "dense_fallback_examples": dense_fallback_examples,
        "accuracy_by_topk": {
            str(budget): accuracy_counts[budget] / evaluated_examples for budget in TOPK_BUDGETS
        },
        "signal_report": asdict(report),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "predictive_gate_passed": report.passes_predictive_gate
        and math.isfinite(report.relative_mae_improvement),
        "claim_eligible": False,
        "claim_limitations": [
            "one training seed per model scale",
            "synthetic associative-recall task only",
            "both scales missed the preregistered 0.85 native-accuracy training target",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches-per-length", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20270714)
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
