from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    CSASelectionPlan,
    CSASelectionProbe,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    LearnedRiskController,
    RiskExample,
    TrainingFreeControllerConfig,
    build_csa_selection_plan,
    build_probe_replay_queries,
    calibrate_learned_risk_controller,
    evaluate_learned_risk_controller,
    extract_request_risk_features,
    generate_associative_recall_batch,
    run_training_free_controller,
    train_learned_risk_controller,
)

SEQUENCE_LENGTHS = (48, 64, 80)
TOPK_BUDGETS = (0, 1, 2, 4, 8, 64)
SPLIT_SEEDS = {"train": 50560714, "calibration": 60660714, "test": 70760714}
M2_CONFIGS = {
    "s55": {"global_block_budget": 6, "top_p": 0.8},
    "s151": {"global_block_budget": 5, "top_p": 0.8},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path, device: torch.device) -> DeepSeekV4ForCausalLM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = DeepSeekV4Config(**checkpoint["config"])
    model = DeepSeekV4ForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def _set_topk(model: DeepSeekV4ForCausalLM, topk: int) -> None:
    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = topk


@torch.no_grad()
def collect_split(
    model: DeepSeekV4ForCausalLM,
    task_config: AssociativeRecallConfig,
    *,
    split: str,
    batches_per_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[RiskExample, ...], tuple[tuple[int, torch.Tensor, torch.Tensor, tuple], ...]]:
    generator = torch.Generator().manual_seed(SPLIT_SEEDS[split])
    examples = []
    records = []
    block_bytes = (model.config.head_dim + model.config.index_head_dim) * 2 + 16
    example_index = 0
    for sequence_length in SEQUENCE_LENGTHS:
        maximum_blocks = (
            sequence_length // model.config.compress_rates["compressed_sparse_attention"]
        )
        for batch_number in range(batches_per_length):
            batch = generate_associative_recall_batch(
                task_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                generator=generator,
                device=device,
            )
            correct_by_budget = {}
            probe = CSASelectionProbe()
            for budget in TOPK_BUDGETS:
                _set_topk(model, budget)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if budget == model.config.index_topk:
                        hidden, _, _ = model.model(batch.input_ids, csa_probe=probe)
                    else:
                        hidden, _, _ = model.model(batch.input_ids)
                    prediction = model.lm_head(hidden[:, -1]).argmax(dim=-1)
                correct_by_budget[budget] = prediction.eq(batch.targets)
            queries = build_probe_replay_queries(
                probe,
                trace_id=f"m3-{split}-{sequence_length}-{batch_number}",
                request_id="associative-recall",
                native_topk=model.config.index_topk,
                block_bytes=block_bytes,
            )
            final_queries = tuple(
                query for query in queries if query.query_position == sequence_length - 1
            )
            for batch_index in range(batch_size):
                group = tuple(query for query in final_queries if query.batch_index == batch_index)
                sufficient = next(
                    (
                        min(budget, maximum_blocks)
                        for budget in TOPK_BUDGETS
                        if bool(correct_by_budget[budget][batch_index])
                    ),
                    maximum_blocks + 1,
                )
                examples.append(
                    RiskExample(
                        group_id=f"{split}-{example_index}",
                        features=extract_request_risk_features(group),
                        sufficient_topk=sufficient,
                        dense_required=sufficient > 4,
                    )
                )
                example_index += 1
            records.append(
                (
                    sequence_length,
                    batch.input_ids.detach().cpu(),
                    batch.targets.detach().cpu(),
                    final_queries,
                )
            )
    return tuple(examples), tuple(records)


def _masked_examples(
    examples: tuple[RiskExample, ...], native_features: bool
) -> tuple[RiskExample, ...]:
    if native_features:
        return examples
    return tuple(
        RiskExample(
            group_id=example.group_id,
            features=(*example.features[:2], *(0.0 for _ in example.features[2:])),
            sufficient_topk=example.sufficient_topk,
            dense_required=example.dense_required,
        )
        for example in examples
    )


def _train_variant(
    train: tuple[RiskExample, ...],
    calibration: tuple[RiskExample, ...],
    test: tuple[RiskExample, ...],
    *,
    hidden_size: int,
    underallocation_weight: float,
    native_features: bool,
    seed: int,
) -> tuple[LearnedRiskController, object, dict]:
    train_rows = _masked_examples(train, native_features)
    calibration_rows = _masked_examples(calibration, native_features)
    test_rows = _masked_examples(test, native_features)
    torch.manual_seed(seed)
    model = LearnedRiskController(len(train_rows[0].features), hidden_size=hidden_size)
    history = train_learned_risk_controller(
        model,
        train_rows,
        steps=500,
        underallocation_weight=underallocation_weight,
        seed=seed,
    )
    calibrated = calibrate_learned_risk_controller(model, calibration_rows)
    metrics = evaluate_learned_risk_controller(model, test_rows, calibrated)
    return (
        model,
        calibrated,
        {
            "hidden_size": hidden_size,
            "underallocation_weight": underallocation_weight,
            "native_features": native_features,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "calibration": asdict(calibrated),
            "test_metrics": asdict(metrics),
        },
    )


def _merge_plans(plans: list[CSASelectionPlan]) -> CSASelectionPlan:
    return CSASelectionPlan(
        trace_id="m3-learned",
        request_id="associative-recall",
        selections=tuple(selection for plan in plans for selection in plan.selections),
    )


@torch.no_grad()
def evaluate_actual_quality(
    model: DeepSeekV4ForCausalLM,
    learned: LearnedRiskController,
    calibration,
    records,
    *,
    scale: str,
    device: torch.device,
) -> dict:
    learned_correct = 0
    m2_correct = 0
    examples = 0
    learned_blocks = 0
    m2_blocks = 0
    learned_fallbacks = 0
    learned_by_length = {length: [0, 0] for length in SEQUENCE_LENGTHS}
    m2_by_length = {length: [0, 0] for length in SEQUENCE_LENGTHS}
    csa_layers = sum(layer.self_attn.csa is not None for layer in model.model.layers)
    maximum_dense_budget = (
        csa_layers
        * max(SEQUENCE_LENGTHS)
        // model.config.compress_rates["compressed_sparse_attention"]
    )
    for sequence_length, input_ids_cpu, targets_cpu, final_queries in records:
        input_ids = input_ids_cpu.to(device)
        targets = targets_cpu.to(device)
        plans = []
        m2 = run_training_free_controller(
            final_queries,
            TrainingFreeControllerConfig(
                global_block_budget=int(M2_CONFIGS[scale]["global_block_budget"]),
                dense_fallback_block_budget=maximum_dense_budget,
                top_p=M2_CONFIGS[scale]["top_p"],
                uncertainty_threshold=0.8,
                dense_cardinality_threshold=0.9,
            ),
        )
        m2_plan = build_csa_selection_plan(m2)
        m2_blocks += sum(action.selected_blocks for action in m2.actions)
        for batch_index in range(input_ids.shape[0]):
            group = tuple(query for query in final_queries if query.batch_index == batch_index)
            features = torch.tensor([extract_request_risk_features(group)], dtype=torch.float32)
            raw_budget, dense_logit = learned(features)
            predicted = float(raw_budget + calibration.budget_offset)
            fallback = float(dense_logit.sigmoid()) >= calibration.dense_threshold
            maximum_blocks = (
                sequence_length // model.config.compress_rates["compressed_sparse_attention"]
            )
            allowed = (1, 2, 4, 8, maximum_blocks)
            topk = next(
                (value for value in allowed if value >= math.ceil(predicted)), maximum_blocks
            )
            if fallback:
                topk = maximum_blocks
                learned_fallbacks += 1
            result = run_training_free_controller(
                group,
                TrainingFreeControllerConfig(
                    global_block_budget=max(1, csa_layers * topk),
                    dense_fallback_block_budget=maximum_dense_budget,
                    top_p=0.8,
                    min_blocks_per_layer=topk,
                    max_extra_blocks_per_layer=0,
                    enable_dense_fallback=False,
                ),
            )
            learned_blocks += sum(action.selected_blocks for action in result.actions)
            plans.append(build_csa_selection_plan(result))
        learned_plan = _merge_plans(plans)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, _, _ = model.model(input_ids, selection_plan=learned_plan)
            learned_predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
            hidden, _, _ = model.model(input_ids, selection_plan=m2_plan)
            m2_predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
        learned_batch_correct = int(learned_predictions.eq(targets).sum())
        m2_batch_correct = int(m2_predictions.eq(targets).sum())
        learned_correct += learned_batch_correct
        m2_correct += m2_batch_correct
        examples += targets.numel()
        learned_by_length[sequence_length][0] += learned_batch_correct
        learned_by_length[sequence_length][1] += targets.numel()
        m2_by_length[sequence_length][0] += m2_batch_correct
        m2_by_length[sequence_length][1] += targets.numel()
    learned_accuracy = learned_correct / examples
    m2_accuracy = m2_correct / examples
    learned_mean_blocks = learned_blocks / examples
    m2_mean_blocks = m2_blocks / examples
    pareto = (learned_mean_blocks < m2_mean_blocks and learned_accuracy >= m2_accuracy - 0.02) or (
        learned_accuracy >= m2_accuracy + 0.005 and learned_mean_blocks <= m2_mean_blocks
    )
    return {
        "examples": examples,
        "learned": {
            "accuracy": learned_accuracy,
            "mean_selected_blocks": learned_mean_blocks,
            "fallback_rate": learned_fallbacks / examples,
            "accuracy_by_length": {
                str(length): correct / count
                for length, (correct, count) in learned_by_length.items()
            },
        },
        "m2_training_free": {
            "accuracy": m2_accuracy,
            "mean_selected_blocks": m2_mean_blocks,
            "accuracy_by_length": {
                str(length): correct / count for length, (correct, count) in m2_by_length.items()
            },
        },
        "pareto_improved": pareto,
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("M3 Tier-S evaluation requires CUDA.")
    device = torch.device("cuda")
    model = _load_model(args.checkpoint, device)
    task_config = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    train, _ = collect_split(
        model,
        task_config,
        split="train",
        batches_per_length=8,
        batch_size=32,
        device=device,
    )
    calibration_rows, _ = collect_split(
        model,
        task_config,
        split="calibration",
        batches_per_length=4,
        batch_size=32,
        device=device,
    )
    test, test_records = collect_split(
        model,
        task_config,
        split="test",
        batches_per_length=8,
        batch_size=32,
        device=device,
    )
    variants = {}
    learned = None
    learned_calibration = None
    for name, hidden, under_weight, native_features in (
        ("full-asymmetric-h32", 32, 4.0, True),
        ("context-layer-only", 32, 4.0, False),
        ("symmetric-loss", 32, 1.0, True),
        ("small-h8", 8, 4.0, True),
    ):
        trained, calibrated, summary = _train_variant(
            train,
            calibration_rows,
            test,
            hidden_size=hidden,
            underallocation_weight=under_weight,
            native_features=native_features,
            seed=80860714,
        )
        variants[name] = summary
        if name == "full-asymmetric-h32":
            learned = trained
            learned_calibration = calibrated
    if learned is None or learned_calibration is None:
        raise RuntimeError("primary learned controller was not trained.")
    actual = evaluate_actual_quality(
        model,
        learned,
        learned_calibration,
        test_records,
        scale=args.scale,
        device=device,
    )
    primary_metrics = variants["full-asymmetric-h32"]["test_metrics"]
    baseline_metrics = variants["context-layer-only"]["test_metrics"]
    calibration_improved = (
        primary_metrics["budget_mae"] < baseline_metrics["budget_mae"]
        and primary_metrics["underallocation_rate"] <= baseline_metrics["underallocation_rate"]
    )
    checks = {
        "pareto_improved_over_m2": actual["pareto_improved"],
        "calibration_improved_over_context_layer": calibration_improved,
        "refresh_ablation_available": False,
    }
    return {
        "schema_version": 1,
        "experiment_id": "m3-tier-s-learned-risk-controller-v1",
        "scale": args.scale,
        "checkpoint": {
            "sha256": _sha256_file(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "split_examples": {
            "train": len(train),
            "calibration": len(calibration_rows),
            "test": len(test),
        },
        "variants": variants,
        "actual_quality": actual,
        "checks": checks,
        "passes_m3_scale_gate": all(checks.values()),
        "decision": "use learned controller" if all(checks.values()) else "retain M2 controller",
        "claim_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "scale": result["scale"],
                "checks": result["checks"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
