from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_natural_workloads import (
    encode_rendered_segments_exact,
    render_chat_split_user_content,
)
from p3_sequence_gate import require_p3_sequence_gate
from prepare_p3_natural_ruler_dataset import (
    EXPECTED_ROWS_PER_LENGTH,
    LENGTHS,
    MODEL_REVISION,
    SAMPLES_PER_TASK,
)
from prepare_p3_ruler_dataset import RULER_REVISION, TASKS
from run_p3_mrcr import (
    arm_config,
    atomic_json,
    failure_record,
    infer_one,
    runtime_environment,
)
from run_p3_ruler_matrix import (
    KVPRESS_REVISION,
    example_score,
    git_dirty,
    git_head,
    load_dataset,
    load_evaluator,
)
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "RULER"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
EXPECTED_EXAMPLES = len(LENGTHS) * EXPECTED_ROWS_PER_LENGTH


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def token_digest(context_ids: torch.Tensor, question_ids: torch.Tensor) -> str:
    payload = {
        "context_ids": context_ids.squeeze(0).tolist(),
        "question_ids": question_ids.squeeze(0).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def rendered_input(
    pipeline: Any,
    *,
    context: str,
    question: str,
    answer_prefix: str,
    maximum_context: int,
) -> dict[str, Any]:
    _require(maximum_context > 0, "RULER maximum context must be positive.")
    rendered_context, rendered_question = render_chat_split_user_content(
        pipeline.tokenizer,
        context,
        question,
        enable_thinking=False,
    )
    rendered_question += answer_prefix
    context_ids, question_ids, boundary_retreat = encode_rendered_segments_exact(
        pipeline.tokenizer,
        rendered_context,
        rendered_question,
    )
    rendered_prompt = rendered_context + rendered_question
    return {
        "context_ids": context_ids,
        "question_ids": question_ids,
        "exact_input_tokens": int(context_ids.shape[1] + question_ids.shape[1]),
        "raw_prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
        "input_token_ids_sha256": token_digest(context_ids, question_ids),
        "token_boundary_retreat": boundary_retreat,
    }


def expected_example_ids() -> list[str]:
    return [
        f"{length}:{task}:{row_number}"
        for length in LENGTHS
        for task in TASKS
        for row_number in range(SAMPLES_PER_TASK)
    ]


def load_dataset_contracts(
    root: Path, *, natural_manifest_digest: str, generator_digest: str
) -> tuple[dict[int, dict[str, Any]], str]:
    manifests: dict[int, dict[str, Any]] = {}
    digests: list[str] = []
    for length in LENGTHS:
        path = root / str(length) / "dataset-manifest.json"
        _require(path.is_file(), f"Missing natural RULER dataset manifest: {path}.")
        payload = json.loads(path.read_text())
        _require(
            payload.get("experiment_id") == "p3-natural-ruler-qwen3-4b-dataset-v1",
            f"Wrong natural RULER dataset id at {length}.",
        )
        _require(
            payload.get("source", {}).get("dirty") is False
            and payload.get("source", {}).get("implementation_sha256") == generator_digest,
            f"Natural RULER generator provenance drifted at {length}.",
        )
        _require(
            payload.get("natural_suite_manifest", {}).get("sha256") == natural_manifest_digest,
            f"Natural RULER suite manifest drifted at {length}.",
        )
        _require(
            payload.get("ruler", {}).get("revision") == RULER_REVISION
            and payload.get("ruler", {}).get("clean_tracked_tree") is True
            and payload.get("tokenizer", {}).get("model_revision") == MODEL_REVISION,
            f"Natural RULER upstream revision drifted at {length}.",
        )
        generation = payload.get("generation", {})
        _require(
            generation.get("length_tokens") == length
            and generation.get("samples_per_task") == SAMPLES_PER_TASK
            and generation.get("tasks") == list(TASKS)
            and generation.get("total_rows") == EXPECTED_ROWS_PER_LENGTH,
            f"Natural RULER generation contract drifted at {length}.",
        )
        manifests[length] = payload
        digests.append(f"{length}:{sha256(path)}")
    digest_set = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    return manifests, digest_set


def load_dependencies(
    manifest_path: Path, inventory_path: Path, selection_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["model"]["revision"] == MODEL_REVISION, "RULER model revision drifted.")
    ruler = manifest["benchmarks"][BENCHMARK]
    _require(
        tuple(ruler["lengths_tokens"]) == LENGTHS
        and ruler["samples_per_task"] == SAMPLES_PER_TASK
        and ruler["task_count"] == len(TASKS)
        and ruler["expected_predictions_per_arm"] == EXPECTED_EXAMPLES,
        "Natural RULER manifest contract drifted.",
    )
    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Common natural dataset inventory is missing or dirty.",
    )
    selection = json.loads(selection_path.read_text())
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False,
        "Fixed baseline selection is missing or dirty.",
    )
    _require(
        selection.get("selected_arm")
        in {
            "streaming_llm",
            "snapkv",
            "pyramidkv",
            "adakv_snapkv",
            "expected_attention",
            "critical_expected_attention",
        }
        and selection.get("selected_compression_ratio") == 0.5,
        "Fixed baseline selection is outside the preregistered candidate set.",
    )
    return manifest, selection


def _existing_records(
    progress: Path, partial: Path, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    if not progress.exists() and not partial.exists():
        atomic_json(progress, identity)
        partial.touch()
        return []
    if progress.is_file() and not partial.exists():
        partial.touch()
    if partial.is_file() and partial.stat().st_size == 0 and not progress.exists():
        atomic_json(progress, identity)
    _require(progress.is_file() and partial.is_file(), "Partial natural RULER state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial RULER provenance drifted.")
    records = [json.loads(line) for line in partial.read_text().splitlines() if line]
    _require(len(records) <= EXPECTED_EXAMPLES, "Partial RULER arm has too many records.")
    return records


def _completed(cell_path: Path, arm: str, identity: dict[str, Any]) -> bool:
    if not cell_path.is_file():
        return False
    cell = json.loads(cell_path.read_text())
    _require(
        cell.get("experiment_id") == "p3-natural-benchmark-arm-cell-v1"
        and cell.get("benchmark") == BENCHMARK
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("run_identity") == identity,
        f"Completed natural RULER provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed natural RULER records drifted: {records}.",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-4B RULER natural baseline arms.")
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/ruler-qwen3-4b/data"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--dataset-inventory",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-data/dataset-inventory.json"
        ),
    )
    parser.add_argument(
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/ruler-qwen3-4b"),
    )
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--max-new-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _require(
        args.max_new_examples is None or args.max_new_examples > 0,
        "max-new-examples must be positive.",
    )
    sequence_decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    if git_dirty(Path.cwd()):
        raise RuntimeError("Natural RULER evaluation requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, selection = load_dependencies(
        args.manifest, args.dataset_inventory, args.fixed_selection
    )
    _require(
        args.seed == manifest["benchmarks"][BENCHMARK]["generation_seed"],
        "Natural RULER generation seed drifted from the frozen manifest.",
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])
    manifest_digest = sha256(args.manifest)
    generator_digest = sha256(Path(__file__).with_name("prepare_p3_natural_ruler_dataset.py"))
    dataset_manifests, dataset_digest_set = load_dataset_contracts(
        args.dataset_root,
        natural_manifest_digest=manifest_digest,
        generator_digest=generator_digest,
    )
    inventory_digest = sha256(args.dataset_inventory)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    runner_digest = sha256(Path(__file__))
    scorer_digest = sha256(Path(__file__).with_name("run_p3_ruler_matrix.py"))
    official_scorer = kvpress_root / manifest["benchmarks"][BENCHMARK]["scorer"]["path"]
    official_scorer_digest = manifest["benchmarks"][BENCHMARK]["scorer"]["sha256"]
    _require(
        official_scorer.is_file() and sha256(official_scorer) == official_scorer_digest,
        "Pinned official RULER scorer drifted.",
    )
    selected_arms = tuple(args.arm or ARMS)
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "dataset_manifest_digest_set_sha256": dataset_digest_set,
            "inventory_sha256": inventory_digest,
            "causal_gate_sha256": causal_digest,
            "fixed_selection_sha256": selection_digest,
            "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
            "arm_config": arm_config(arm, selection, selection_digest),
            "seed": args.seed,
        }
        for arm in selected_arms
    }
    pending = [
        arm
        for arm in selected_arms
        if not _completed(args.output_root / arm / "cell.json", arm, identities[arm])
    ]
    if not pending:
        print(json.dumps({"status": "complete", "arms": list(selected_arms)}))
        return
    _require(torch.cuda.is_available(), "Natural RULER evaluation requires CUDA.")

    lock = acquire_gpu_lock("p3-natural-ruler")
    try:
        EvaluationConfig, EvaluationRunner, _scorer = load_evaluator(kvpress_root)
        base_config = EvaluationConfig(
            dataset="ruler",
            data_dir=str(LENGTHS[0]),
            model=str(model_snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=args.seed,
            max_context_length=manifest["model"]["maximum_supported_context_tokens"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(base_config)
        runner._setup_press()
        runner._setup_model_pipeline()
        environment = runtime_environment()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        maximum_context = manifest["model"]["maximum_supported_context_tokens"]
        expected_ids = expected_example_ids()
        _require(len(expected_ids) == EXPECTED_EXAMPLES, "Natural RULER identity grid drifted.")

        for arm in pending:
            settings = identities[arm]["arm_config"]
            runner.config.press_name = settings["press_name"]
            runner.config.compression_ratio = settings["compression_ratio"]
            runner._setup_press()
            root = args.output_root / arm
            progress = root / "progress.json"
            partial = root / "records.partial.jsonl"
            existing = _existing_records(progress, partial, identities[arm])
            _require(
                all(
                    existing[index]["example_id"] == expected_ids[index]
                    for index in range(len(existing))
                ),
                "Partial natural RULER example order drifted.",
            )
            limit = EXPECTED_EXAMPLES
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            global_index = 0
            with partial.open("a") as handle:
                for length in LENGTHS:
                    frame = load_dataset(dataset_manifests[length])
                    _require(len(frame) == EXPECTED_ROWS_PER_LENGTH, "RULER length row drifted.")
                    for row in frame.to_dict(orient="records"):
                        if global_index < len(existing):
                            global_index += 1
                            continue
                        if global_index >= limit:
                            break
                        example_id = f"{length}:{row['row_id']}"
                        _require(
                            example_id == expected_ids[global_index],
                            "Natural RULER generated order drifted.",
                        )
                        rendered = rendered_input(
                            runner.pipeline,
                            context=row["context"],
                            question=row["question"],
                            answer_prefix=row["answer_prefix"],
                            maximum_context=maximum_context,
                        )
                        reserve = int(row["max_new_tokens"])
                        base = {
                            "example_id": example_id,
                            "benchmark": BENCHMARK,
                            "arm": arm,
                            "exact_input_tokens": rendered["exact_input_tokens"],
                            "generation_reserve_tokens": reserve,
                            "raw_prompt_sha256": rendered["raw_prompt_sha256"],
                            "input_token_ids_sha256": rendered["input_token_ids_sha256"],
                            "token_boundary_retreat": rendered["token_boundary_retreat"],
                            "arm_config": settings,
                            "revisions": {
                                "model_revision": MODEL_REVISION,
                                "dataset_revision": RULER_REVISION,
                                "code_revision": RULER_REVISION,
                                "scorer_sha256": scorer_digest,
                                "official_scorer_sha256": official_scorer_digest,
                            },
                            "length_tokens": length,
                            "task": row["task"],
                        }
                        if rendered["exact_input_tokens"] + reserve > maximum_context:
                            record = failure_record(
                                base,
                                failure_type="unsupported-context",
                                latency_ms=0.0,
                                peak_hbm_bytes=0,
                            )
                        else:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                            started = time.perf_counter_ns()
                            try:
                                response, resident_bytes = infer_one(
                                    pipeline=runner.pipeline,
                                    press=runner.press,
                                    rendered=rendered,
                                    max_new_tokens=reserve,
                                )
                                torch.cuda.synchronize()
                                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                                peak_hbm = torch.cuda.max_memory_allocated()
                                if not response.strip():
                                    record = failure_record(
                                        base,
                                        failure_type="empty-generation",
                                        latency_ms=latency_ms,
                                        peak_hbm_bytes=peak_hbm,
                                        hot_resident_bytes=resident_bytes,
                                    )
                                else:
                                    generated = len(
                                        runner.pipeline.tokenizer.encode(
                                            response, add_special_tokens=False
                                        )
                                    )
                                    record = {
                                        **base,
                                        "status": "scored",
                                        "raw_response": response,
                                        "parsed_response": response,
                                        "score": example_score(
                                            row["task"], response, row["answer"]
                                        ),
                                        "failure_type": None,
                                        "stop_reason": (
                                            "max-new-tokens"
                                            if generated >= reserve
                                            else "eos-or-special-token"
                                        ),
                                        "generated_tokens_observed": generated,
                                        "latency_ms": latency_ms,
                                        "peak_hbm_bytes": peak_hbm,
                                        "hot_resident_bytes": resident_bytes,
                                    }
                            except torch.cuda.OutOfMemoryError as error:
                                record = failure_record(
                                    base,
                                    failure_type="oom",
                                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                    peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                    error=error,
                                )
                                torch.cuda.empty_cache()
                            except Exception as error:
                                record = failure_record(
                                    base,
                                    failure_type="runtime-error",
                                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                    peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                    error=error,
                                )
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        global_index += 1
                        print(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "completed_examples": global_index,
                                    "total": EXPECTED_EXAMPLES,
                                }
                            ),
                            flush=True,
                        )
                    if global_index >= limit:
                        break
            if limit < EXPECTED_EXAMPLES:
                continue
            _require(global_index == EXPECTED_EXAMPLES, "Natural RULER run did not close.")
            records = root / "records.jsonl"
            partial.replace(records)
            progress.unlink()
            cell = {
                "schema_version": 1,
                "experiment_id": "p3-natural-benchmark-arm-cell-v1",
                "benchmark": BENCHMARK,
                "arm": arm,
                "status": "terminal",
                "source": {
                    "commit": source_commit,
                    "dirty": False,
                    "implementation_sha256": runner_digest,
                },
                "run_identity": identities[arm],
                "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                "causal_gate": {"path": str(args.causal_gate), "sha256": causal_digest},
                "dataset_inventory": {
                    "path": str(args.dataset_inventory),
                    "sha256": inventory_digest,
                },
                "fixed_baseline_selection": {
                    "path": str(args.fixed_selection),
                    "sha256": selection_digest,
                },
                "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
                "benchmark_dataset_digest_set_sha256": dataset_digest_set,
                "p3_sequence_decision": sequence_decision,
                "environment": environment,
                "raw_records": {"path": str(records), "sha256": sha256(records)},
            }
            atomic_json(root / "cell.json", cell)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
