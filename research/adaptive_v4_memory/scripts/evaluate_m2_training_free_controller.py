from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    CSASelectionProbe,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    TrainingFreeControllerConfig,
    build_csa_selection_plan,
    build_probe_replay_queries,
    generate_associative_recall_batch,
    run_training_free_controller,
    validate_controller_replay,
)

SEQUENCE_LENGTHS = (48, 64, 80)
FIXED_TOPK = (1, 2, 4, 8)


@dataclass
class Metrics:
    correct: int = 0
    examples: int = 0
    correct_by_length: dict[int, int] = field(default_factory=dict)
    examples_by_length: dict[int, int] = field(default_factory=dict)
    final_selected_blocks: int = 0
    final_selected_bytes: int = 0
    fallback_examples: int = 0
    total_selected_bytes: int = 0
    movement_bytes: int = 0
    controller_ns: int = 0
    forward_ns: int = 0
    batches: int = 0

    def summary(self) -> dict:
        accuracy_by_length = {
            str(length): self.correct_by_length.get(length, 0) / self.examples_by_length[length]
            for length in sorted(self.examples_by_length)
        }
        return {
            "accuracy": self.correct / self.examples,
            "accuracy_by_length": accuracy_by_length,
            "examples": self.examples,
            "mean_final_selected_blocks": self.final_selected_blocks / self.examples,
            "mean_final_selected_bytes": self.final_selected_bytes / self.examples,
            "fallback_rate": self.fallback_examples / self.examples,
            "movement_to_selection_bytes": self.movement_bytes / max(self.total_selected_bytes, 1),
            "mean_movement_bytes_per_example": self.movement_bytes / self.examples,
            "controller_cpu_us_per_example": self.controller_ns / self.examples / 1000.0,
            "forward_ms_per_batch": self.forward_ns / self.batches / 1_000_000.0,
        }


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


def _controller_grid(model: DeepSeekV4ForCausalLM) -> dict[str, TrainingFreeControllerConfig]:
    csa_layers = sum(layer.self_attn.csa is not None for layer in model.model.layers)
    maximum_blocks = (
        max(SEQUENCE_LENGTHS) // model.config.compress_rates["compressed_sparse_attention"]
    )
    configs = {}
    for blocks_per_layer in (1, 2):
        for top_p in (0.5, 0.8):
            for threshold in (0.65, 0.8):
                name = f"b{blocks_per_layer}-p{top_p:.1f}-u{threshold:.2f}"
                configs[name] = TrainingFreeControllerConfig(
                    global_block_budget=csa_layers * blocks_per_layer,
                    dense_fallback_block_budget=csa_layers * maximum_blocks,
                    top_p=top_p,
                    min_blocks_per_layer=1,
                    max_extra_blocks_per_layer=2,
                    uncertainty_threshold=threshold,
                    dense_cardinality_threshold=0.9,
                    stable_reuse_threshold=0.8,
                    min_refresh_interval=1,
                    max_refresh_interval=4,
                )
    return configs


def _update_accuracy(
    metrics: Metrics,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sequence_length: int,
) -> None:
    correct = int(predictions.eq(targets).sum())
    examples = targets.numel()
    metrics.correct += correct
    metrics.examples += examples
    metrics.correct_by_length[sequence_length] = (
        metrics.correct_by_length.get(sequence_length, 0) + correct
    )
    metrics.examples_by_length[sequence_length] = (
        metrics.examples_by_length.get(sequence_length, 0) + examples
    )


@torch.no_grad()
def evaluate_split(
    model: DeepSeekV4ForCausalLM,
    task_config: AssociativeRecallConfig,
    *,
    controller_configs: dict[str, TrainingFreeControllerConfig],
    seed: int,
    batches_per_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    fixed_metrics = {topk: Metrics() for topk in FIXED_TOPK}
    controller_metrics = {name: Metrics() for name in controller_configs}
    native_metrics = Metrics()
    csa_layers = sum(layer.self_attn.csa is not None for layer in model.model.layers)
    block_bytes = (model.config.head_dim + model.config.index_head_dim) * torch.tensor(
        [], dtype=torch.bfloat16
    ).element_size() + 2 * 8

    for sequence_length in SEQUENCE_LENGTHS:
        block_count = sequence_length // model.config.compress_rates["compressed_sparse_attention"]
        for batch_number in range(batches_per_length):
            batch = generate_associative_recall_batch(
                task_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                generator=generator,
                device=device,
            )
            _set_index_topk(model, model.config.index_topk)
            probe = CSASelectionProbe()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden, _, _ = model.model(batch.input_ids, csa_probe=probe)
                native_predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
            torch.cuda.synchronize()
            native_metrics.forward_ns += time.perf_counter_ns() - started
            native_metrics.batches += 1
            _update_accuracy(native_metrics, native_predictions, batch.targets, sequence_length)
            native_metrics.final_selected_blocks += (
                min(model.config.index_topk, block_count) * csa_layers * batch_size
            )
            native_metrics.final_selected_bytes += (
                min(model.config.index_topk, block_count) * csa_layers * batch_size * block_bytes
            )

            queries = build_probe_replay_queries(
                probe,
                trace_id=f"m2-{seed}-{sequence_length}-{batch_number}",
                request_id="associative-recall",
                native_topk=model.config.index_topk,
                block_bytes=block_bytes,
            )
            control_queries = tuple(
                query for query in queries if query.query_position == sequence_length - 1
            )

            for topk, metrics in fixed_metrics.items():
                _set_index_topk(model, topk)
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden, _, _ = model.model(batch.input_ids)
                    predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
                torch.cuda.synchronize()
                metrics.forward_ns += time.perf_counter_ns() - started
                metrics.batches += 1
                _update_accuracy(metrics, predictions, batch.targets, sequence_length)
                selected = min(topk, block_count) * csa_layers * batch_size
                metrics.final_selected_blocks += selected
                metrics.final_selected_bytes += selected * block_bytes

            for name, config in controller_configs.items():
                metrics = controller_metrics[name]
                started = time.perf_counter_ns()
                controller = run_training_free_controller(control_queries, config)
                plan = build_csa_selection_plan(controller)
                metrics.controller_ns += time.perf_counter_ns() - started
                validate_controller_replay(control_queries, controller)
                _set_index_topk(model, model.config.index_topk)
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden, _, _ = model.model(batch.input_ids, selection_plan=plan)
                    predictions = model.lm_head(hidden[:, -1]).argmax(dim=-1)
                torch.cuda.synchronize()
                metrics.forward_ns += time.perf_counter_ns() - started
                metrics.batches += 1
                _update_accuracy(metrics, predictions, batch.targets, sequence_length)
                final_actions = [
                    action
                    for action in controller.actions
                    if action.query_position == sequence_length - 1
                ]
                metrics.final_selected_blocks += sum(
                    action.selected_blocks for action in final_actions
                )
                metrics.final_selected_bytes += sum(
                    action.selected_bytes for action in final_actions
                )
                metrics.fallback_examples += sum(
                    action.fallback_reason is not None for action in final_actions
                )
                metrics.total_selected_bytes += controller.total_selected_bytes
                metrics.movement_bytes += controller.total_movement_bytes

    return {
        "native": native_metrics.summary(),
        "fixed": {str(topk): metrics.summary() for topk, metrics in fixed_metrics.items()},
        "controllers": {
            name: {
                "config": asdict(controller_configs[name]),
                "metrics": metrics.summary(),
            }
            for name, metrics in controller_metrics.items()
        },
    }


def _select_calibrated(calibration: dict) -> tuple[str, str]:
    fixed_two = calibration["fixed"]["2"]
    eligible = []
    for name, payload in calibration["controllers"].items():
        metrics = payload["metrics"]
        if (
            metrics["accuracy"] >= fixed_two["accuracy"] - 0.02
            and metrics["mean_final_selected_blocks"]
            <= fixed_two["mean_final_selected_blocks"] * 1.05
        ):
            eligible.append(name)
    pool = eligible or list(calibration["controllers"])
    selected = max(
        pool,
        key=lambda name: (
            calibration["controllers"][name]["metrics"]["accuracy"]
            - 0.002 * calibration["controllers"][name]["metrics"]["mean_final_selected_blocks"],
            -calibration["controllers"][name]["metrics"]["movement_to_selection_bytes"],
            name,
        ),
    )
    reason = "met fixed-top-k-2 calibration constraints" if eligible else "fallback utility rule"
    return selected, reason


def _test_gate(
    test: dict,
    selected_name: str,
    no_fallback_name: str,
    forced_fallback_name: str,
) -> dict:
    selected = test["controllers"][selected_name]["metrics"]
    no_fallback = test["controllers"][no_fallback_name]["metrics"]
    forced_fallback = test["controllers"][forced_fallback_name]["metrics"]
    fixed = test["fixed"]
    matched_key = min(
        fixed,
        key=lambda key: abs(
            fixed[key]["mean_final_selected_blocks"] - selected["mean_final_selected_blocks"]
        ),
    )
    matched = fixed[matched_key]
    regression = matched["accuracy"] - selected["accuracy"]
    length_regressions = {
        length: matched["accuracy_by_length"][length] - selected["accuracy_by_length"][length]
        for length in selected["accuracy_by_length"]
    }
    fewer_blocks = (
        selected["mean_final_selected_blocks"] < matched["mean_final_selected_blocks"] - 1e-9
        and regression <= 0.02
    )
    better_quality = (
        selected["accuracy"] >= matched["accuracy"] + 0.005
        and selected["mean_final_selected_blocks"] <= matched["mean_final_selected_blocks"] + 1e-9
    )
    checks = {
        "memory_matched_accuracy_regression": regression <= 0.02,
        "worst_length_accuracy_regression": max(length_regressions.values()) <= 0.05,
        "movement_cost": selected["movement_to_selection_bytes"] <= 2.0,
        "controller_overhead": selected["controller_cpu_us_per_example"] <= 1000.0,
        "fallback_recovery": forced_fallback["fallback_rate"] > 0.0
        and forced_fallback["accuracy"] >= no_fallback["accuracy"],
        "pareto_improvement": fewer_blocks or better_quality,
    }
    return {
        "matched_fixed_topk": int(matched_key),
        "accuracy_regression": regression,
        "length_accuracy_regressions": length_regressions,
        "natural_fallback_accuracy_recovery": selected["accuracy"] - no_fallback["accuracy"],
        "forced_fallback_accuracy_recovery": forced_fallback["accuracy"] - no_fallback["accuracy"],
        "checks": checks,
        "passes_m2_gate": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("M2 Tier-S evaluation requires CUDA.")
    device = torch.device("cuda")
    model = _load_model(args.checkpoint, device)
    task_config = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    grid = _controller_grid(model)
    calibration = evaluate_split(
        model,
        task_config,
        controller_configs=grid,
        seed=args.calibration_seed,
        batches_per_length=args.calibration_batches,
        batch_size=args.batch_size,
        device=device,
    )
    selected_name, selection_reason = _select_calibrated(calibration)
    selected_config = grid[selected_name]
    no_fallback_name = f"{selected_name}-no-fallback"
    forced_fallback_name = f"{selected_name}-forced-fallback"
    no_fallback = replace(
        selected_config,
        enable_dense_fallback=False,
        dense_fallback_block_budget=selected_config.global_block_budget,
    )
    forced_fallback = replace(selected_config, uncertainty_threshold=0.0)
    test = evaluate_split(
        model,
        task_config,
        controller_configs={
            selected_name: selected_config,
            no_fallback_name: no_fallback,
            forced_fallback_name: forced_fallback,
        },
        seed=args.test_seed,
        batches_per_length=args.test_batches,
        batch_size=args.batch_size,
        device=device,
    )
    gate = _test_gate(
        test,
        selected_name,
        no_fallback_name,
        forced_fallback_name,
    )
    return {
        "schema_version": 1,
        "experiment_id": "m2-tier-s-training-free-controller-v1",
        "scale": args.scale,
        "checkpoint": {
            "sha256": _sha256_file(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "calibration_seed": args.calibration_seed,
        "test_seed": args.test_seed,
        "selected_controller": selected_name,
        "selection_reason": selection_reason,
        "calibration": calibration,
        "test": test,
        "gate": gate,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "claim_eligible": False,
        "interpretation": "offline two-pass replay evidence; no online serving claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-seed", type=int, default=30360714)
    parser.add_argument("--test-seed", type=int, default=40460714)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--test-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "scale": result["scale"],
                "selected_controller": result["selected_controller"],
                "gate": result["gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
