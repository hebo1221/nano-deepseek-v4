from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_natural_metrics import score_mrcr
from p3_natural_workloads import (
    encode_rendered_segments_exact,
    mrcr_generation_reserve,
    render_chat_split_last_user,
    select_mrcr_primary_rows,
)
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from transformers import DynamicCache
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "MRCR"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
NEEDLE_COUNTS = (2, 4, 8)
EXPECTED_EXAMPLES = 1500
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_rows(
    paths: list[Path], *, contract: dict[str, Any], official_encoder: Any
) -> list[dict[str, Any]]:
    by_needle: dict[int, list[dict[str, Any]]] = {count: [] for count in NEEDLE_COUNTS}
    for path in paths:
        parent = path.parent.name
        _require(parent.endswith("needle"), f"Cannot infer MRCR needle count from {path}.")
        needle_count = int(parent.removesuffix("needle"))
        _require(needle_count in by_needle, f"Unexpected MRCR needle count: {needle_count}.")
        rows = pq.read_table(path).to_pylist()
        _require(all(isinstance(row, dict) for row in rows), f"Invalid MRCR rows in {path}.")
        by_needle[needle_count].extend(rows)

    selected: list[dict[str, Any]] = []
    boundaries = contract["bin_boundaries_tokens"]
    for needle_count in NEEDLE_COUNTS:
        rows = select_mrcr_primary_rows(
            rows=by_needle[needle_count],
            encoder=official_encoder,
            boundaries=boundaries,
            primary_bins=contract["primary_bins_through_128k"],
            samples_per_bin=contract["samples_per_bin_per_needle_count"],
        )
        expected_per_needle = (
            contract["primary_bins_through_128k"] * contract["samples_per_bin_per_needle_count"]
        )
        _require(len(rows) == expected_per_needle, "MRCR per-needle primary total drifted.")
        for row in rows:
            identity_bytes = json.dumps(
                {
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "prefix": row["random_string_to_prepend"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            identifier = hashlib.sha256(identity_bytes).hexdigest()
            selected.append(
                {**row, "example_id": f"{needle_count}n:{row['official_bin_index']}:{identifier}"}
            )
    expected_total = len(NEEDLE_COUNTS) * (
        contract["primary_bins_through_128k"] * contract["samples_per_bin_per_needle_count"]
    )
    _require(len(selected) == expected_total, "MRCR primary example total drifted.")
    identifiers = [row["example_id"] for row in selected]
    _require(len(set(identifiers)) == len(identifiers), "MRCR example identities are duplicated.")
    return selected


def rendered_input(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    context, query = render_chat_split_last_user(tokenizer, messages)
    context_ids, query_ids, boundary_retreat = encode_rendered_segments_exact(
        tokenizer, context, query
    )
    return {
        "context_ids": context_ids,
        "question_ids": query_ids,
        "exact_input_tokens": int(context_ids.shape[1] + query_ids.shape[1]),
        "raw_prompt_sha256": hashlib.sha256((context + query).encode()).hexdigest(),
        "token_boundary_retreat": boundary_retreat,
    }


def cache_bytes(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


@torch.inference_mode()
def infer_one(
    *, pipeline: Any, press: Any, rendered: dict[str, Any], max_new_tokens: int
) -> tuple[str, int]:
    context_ids = rendered["context_ids"].to(pipeline.model.device)
    question_ids = rendered["question_ids"].to(pipeline.model.device)
    original_context_length = int(context_ids.shape[1])
    cache = DynamicCache()
    with press(pipeline.model) if press is not None else contextlib.nullcontext():
        pipeline.model.model(input_ids=context_ids, past_key_values=cache)
    resident_bytes = cache_bytes(cache)
    response = pipeline.generate_answer(
        question_ids=question_ids,
        cache=cache,
        context_length=original_context_length,
        max_new_tokens=max_new_tokens,
    )
    return response, resident_bytes


def arm_config(arm: str, selection: dict[str, Any], selection_digest: str) -> dict[str, Any]:
    if arm == "native-dense":
        return {"press_name": "no_press", "compression_ratio": 0.0}
    _require(arm == "strongest-memory-matched-fixed", f"Unknown MRCR arm: {arm}.")
    return {
        "press_name": selection["selected_arm"],
        "compression_ratio": selection["selected_compression_ratio"],
        "selection_sha256": selection_digest,
    }


def failure_record(
    base: dict[str, Any],
    *,
    failure_type: str,
    latency_ms: float,
    peak_hbm_bytes: int,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        **base,
        "status": "failure",
        "raw_response": "",
        "parsed_response": None,
        "score": None,
        "failure_type": failure_type,
        "stop_reason": failure_type,
        "latency_ms": latency_ms,
        "peak_hbm_bytes": peak_hbm_bytes,
        "hot_resident_bytes": 0,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }


def _dependency_file(metadata: dict[str, Any], expected_digest: str) -> Path:
    path = Path(metadata["path"])
    _require(path.is_file() and sha256(path) == expected_digest, f"Dataset drifted: {path}.")
    _require(metadata["sha256"] == expected_digest, f"Dataset inventory drifted: {path}.")
    return path


def load_dependencies(
    manifest_path: Path, inventory_path: Path, selection_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["model"]["revision"] == MODEL_REVISION, "Model revision drifted.")
    contract = manifest["benchmarks"][BENCHMARK]
    _require(contract["dataset"]["revision"] == DATASET_REVISION, "MRCR revision drifted.")
    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Natural dataset inventory is not a clean frozen artifact.",
    )
    observed = inventory["benchmarks"][BENCHMARK]
    _require(observed["revision"] == DATASET_REVISION, "MRCR inventory revision drifted.")
    expected_files = {row["path"]: row for row in contract["dataset"]["files"]}
    observed_files = observed["files"]
    _require(len(observed_files) == len(expected_files), "MRCR inventory file count drifted.")
    paths: list[Path] = []
    for metadata in observed_files:
        relative = "/".join(Path(metadata["path"]).parts[-2:])
        _require(relative in expected_files, f"Unexpected MRCR inventory path: {relative}.")
        paths.append(_dependency_file(metadata, expected_files[relative]["sha256"]))

    selection = json.loads(selection_path.read_text())
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False,
        "Fixed baseline selection is missing or dirty.",
    )
    _require(
        selection.get("selected_arm")
        in {"streaming_llm", "snapkv", "expected_attention", "critical_expected_attention"}
        and selection.get("selected_compression_ratio") == 0.5,
        "Fixed baseline selection is outside the preregistered candidate set.",
    )
    return manifest, selection, paths


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
    _require(progress.is_file() and partial.is_file(), "Partial MRCR arm state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial MRCR provenance drifted.")
    records = [json.loads(line) for line in partial.read_text().splitlines() if line]
    _require(len(records) <= EXPECTED_EXAMPLES, "Partial MRCR arm has too many records.")
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
        f"Completed MRCR provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed MRCR records drifted: {records}.",
    )
    return True


def runtime_environment() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "transformers": importlib.metadata.version("transformers"),
        "kvpress": importlib.metadata.version("kvpress"),
        "tiktoken": importlib.metadata.version("tiktoken"),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B MRCR primary cells.")
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/mrcr"),
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
        raise RuntimeError("MRCR evaluation requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, selection, dataset_paths = load_dependencies(
        args.manifest, args.dataset_inventory, args.fixed_selection
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])

    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("MRCR requires the frozen tiktoken o200k_base tokenizer.") from error
    official_encoder = tiktoken.get_encoding("o200k_base")
    rows = load_rows(
        dataset_paths,
        contract=manifest["benchmarks"][BENCHMARK],
        official_encoder=official_encoder,
    )
    _require(len(rows) == EXPECTED_EXAMPLES, "Frozen MRCR run must contain 1,500 examples.")
    runner_digest = sha256(Path(__file__))
    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    scorer_digest = sha256(Path(__file__).with_name("p3_natural_metrics.py"))
    maximum_context = manifest["model"]["maximum_supported_context_tokens"]
    selected_arms = tuple(args.arm or ARMS)
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
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
    _require(torch.cuda.is_available(), "MRCR evaluation requires CUDA.")

    lock = acquire_gpu_lock("p3-mrcr")
    try:
        EvaluationConfig, EvaluationRunner, _scorer = load_evaluator(kvpress_root)
        config = EvaluationConfig(
            dataset="mrcr",
            model=str(model_snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=args.seed,
            max_context_length=maximum_context,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(config)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
        environment = runtime_environment()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        for arm in pending:
            arm_settings = identities[arm]["arm_config"]
            runner.config.press_name = arm_settings["press_name"]
            runner.config.compression_ratio = arm_settings["compression_ratio"]
            runner._setup_press()
            root = args.output_root / arm
            progress = root / "progress.json"
            partial = root / "records.partial.jsonl"
            existing = _existing_records(progress, partial, identities[arm])
            _require(
                all(
                    existing[index]["example_id"] == rows[index]["example_id"]
                    for index in range(len(existing))
                ),
                "Partial MRCR example order drifted.",
            )
            limit = len(rows)
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            with partial.open("a") as handle:
                for index in range(len(existing), limit):
                    row = rows[index]
                    rendered = rendered_input(tokenizer, row["messages"])
                    reserve = mrcr_generation_reserve(row, tokenizer)
                    base = {
                        "example_id": row["example_id"],
                        "benchmark": BENCHMARK,
                        "arm": arm,
                        "exact_input_tokens": rendered["exact_input_tokens"],
                        "generation_reserve_tokens": reserve,
                        "raw_prompt_sha256": rendered["raw_prompt_sha256"],
                        "token_boundary_retreat": rendered["token_boundary_retreat"],
                        "arm_config": arm_settings,
                        "revisions": {
                            "model_revision": MODEL_REVISION,
                            "dataset_revision": DATASET_REVISION,
                            "code_revision": f"dataset-readme@{DATASET_REVISION}",
                            "scorer_sha256": scorer_digest,
                        },
                        "needle_count": int(row["example_id"].split("n:", 1)[0]),
                        "official_bin_index": row["official_bin_index"],
                        "official_o200k_prompt_plus_answer_tokens": row[
                            "official_o200k_prompt_plus_answer_tokens"
                        ],
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
                                )
                            else:
                                generated = len(
                                    tokenizer.encode(response, add_special_tokens=False)
                                )
                                record = {
                                    **base,
                                    "status": "scored",
                                    "raw_response": response,
                                    "parsed_response": response,
                                    "score": score_mrcr(
                                        response,
                                        row["answer"],
                                        row["random_string_to_prepend"],
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
                    print(
                        json.dumps(
                            {"arm": arm, "completed_examples": index + 1, "total": len(rows)}
                        ),
                        flush=True,
                    )
            if limit < len(rows):
                continue
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
                "p3_sequence_decision": sequence_decision,
                "environment": environment,
                "raw_records": {"path": str(records), "sha256": sha256(records)},
            }
            atomic_json(root / "cell.json", cell)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
