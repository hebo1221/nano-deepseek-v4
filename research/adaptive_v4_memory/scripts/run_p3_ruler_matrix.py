from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_sequence_gate import require_p3_sequence_gate
from prepare_p3_ruler_dataset import LENGTHS, MODEL_REVISION, RULER_REVISION, TASKS, sha256

KVPRESS_REVISION = "6d965557a5b9f0201a2301b23c454473dd681d0d"
ARMS = {
    "native": ("no_press", (0.0,)),
    "streaming_llm": ("streaming_llm", (0.25, 0.5, 0.75)),
    "snapkv": ("snapkv", (0.25, 0.5, 0.75)),
    "pyramidkv": ("pyramidkv", (0.25, 0.5, 0.75)),
    "adakv_snapkv": ("adakv_snapkv", (0.25, 0.5, 0.75)),
    "expected_attention": ("expected_attention", (0.25, 0.5, 0.75)),
    "critical_expected_attention": ("critical_expected_attention", (0.25, 0.5, 0.75)),
}
QUESTION_PATTERNS = {
    "niah": re.compile(r"What (?:is|are all) the special magic"),
    "vt": re.compile(r"Question: Find all variables that are assigned the value"),
    "cwe": re.compile(r"Question: What are the 10 most common words in the above list\?"),
    "fwe": re.compile(r"Question: Do not provide any explanation\."),
    "qa": re.compile(r"Answer the question based on the given documents\."),
}
ANSWER_PATTERNS = {
    "niah": re.compile(r"The special magic"),
    "vt": re.compile(r"Answer:"),
    "cwe": re.compile(r"Answer:"),
    "fwe": re.compile(r"Answer:"),
    "qa": re.compile(r"Answer:"),
}
MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f]")


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_dirty(path: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def split_prompt(text: str, task: str) -> tuple[str, str, str]:
    category = task.split("_")[0]
    matches = list(QUESTION_PATTERNS[category].finditer(text))
    if not matches:
        raise ValueError(f"Question delimiter missing for task {task}.")
    context, qa = text[: matches[-1].start()], text[matches[-1].start() :]
    answer = ANSWER_PATTERNS[category].search(qa)
    if answer is None:
        raise ValueError(f"Answer delimiter missing for task {task}.")
    return context, qa[: answer.start()], qa[answer.start() :]


def load_dataset(manifest: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for task in TASKS:
        metadata = manifest["task_artifacts"][task]
        path = Path(metadata["path"])
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise ValueError(f"RULER task artifact drifted: {path}")
        with path.open() as handle:
            for row_number, line in enumerate(handle):
                row = json.loads(line)
                context, question, answer_prefix = split_prompt(row["input"], task)
                records.append(
                    {
                        "row_id": f"{task}:{row_number}",
                        "context": context,
                        "question": question,
                        "answer_prefix": answer_prefix,
                        "answer": row["outputs"],
                        "task": task,
                        "max_new_tokens": MAX_NEW_TOKENS[task.split("_")[0]],
                    }
                )
    expected = manifest["generation"]["total_rows"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} RULER rows, found {len(records)}.")
    return pd.DataFrame.from_records(records)


def example_score(task: str, prediction: str, references: list[str]) -> float:
    normalized = CONTROL_CHARACTERS.sub("", prediction.strip()).strip().lower()
    matches = [1.0 if reference.lower() in normalized else 0.0 for reference in references]
    return max(matches) if task.startswith("qa_") else sum(matches) / len(matches)


def cells(
    lengths: tuple[int, ...], arms: tuple[str, ...], ratios: tuple[float, ...] | None
) -> tuple[tuple[int, str, str, float], ...]:
    result: list[tuple[int, str, str, float]] = []
    for length in lengths:
        for arm in arms:
            press, registered = ARMS[arm]
            selected = registered
            if ratios is not None and arm != "native":
                selected = tuple(ratio for ratio in registered if ratio in ratios)
            result.extend((length, arm, press, ratio) for ratio in selected)
    return tuple(result)


def cell_dir(root: Path, length: int, arm: str, ratio: float) -> Path:
    return root / str(length) / arm / f"ratio-{ratio:.2f}"


def completed(
    cell: Path,
    runner_digest: str,
    manifest_digest: str,
    dataset_digest: str,
    causal_gate_digest: str,
) -> bool:
    audit_path = cell / "audit.json"
    if not audit_path.is_file():
        if cell.exists():
            raise ValueError(f"Incomplete P3 cell will not be overwritten: {cell}")
        return False
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "complete"
        or audit.get("source", {}).get("dirty") is not False
        or audit.get("source", {}).get("implementation_sha256") != runner_digest
        or audit.get("experiment_manifest", {}).get("sha256") != manifest_digest
        or audit.get("dataset_manifest", {}).get("sha256") != dataset_digest
        or audit.get("upstreams", {}).get("causal_gate_sha256") != causal_gate_digest
    ):
        raise ValueError(f"Existing P3 cell has stale provenance: {cell}")
    for artifact in audit["outputs"].values():
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"Existing P3 output drifted: {path}")
    return True


def environment() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "kvpress": importlib.metadata.version("kvpress"),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def verify_model_snapshot(path: Path, manifest: dict[str, Any]) -> None:
    expected = manifest["model"]["snapshot_files_sha256"]
    actual = {item.name for item in path.iterdir() if item.is_file()}
    if actual != set(expected):
        raise ValueError("Model snapshot file set drifted.")
    for name, digest in expected.items():
        if sha256(path / name) != digest:
            raise ValueError(f"Model snapshot artifact drifted: {path / name}")


def load_evaluator(kvpress_root: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(kvpress_root / "evaluation"))
    try:
        from evaluate import EvaluationConfig, EvaluationRunner
        from evaluate_registry import SCORER_REGISTRY
    finally:
        sys.path.pop(0)
    return EvaluationConfig, EvaluationRunner, SCORER_REGISTRY["ruler"]


def write_failure(
    cell: Path, source_commit: str, command: list[str], started: float, error: BaseException
) -> None:
    atomic_json(
        cell.with_name(cell.name + ".failure.json"),
        {
            "schema_version": 1,
            "experiment_id": "p3-ruler-qwen3-1.7b-cell-failure-v1",
            "status": "failed",
            "source": {"commit": source_commit, "dirty": False},
            "command": command,
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen resume-safe P3 RULER matrix.")
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/ruler-qwen3-1.7b/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/ruler-qwen3-1.7b/results"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-ruler-qwen3-1.7b-v1.json"),
    )
    parser.add_argument("--length", type=int, action="append", choices=LENGTHS)
    parser.add_argument("--arm", action="append", choices=tuple(ARMS))
    parser.add_argument("--ratio", type=float, action="append", choices=(0.25, 0.5, 0.75))
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"
        ),
    )
    args = parser.parse_args()
    p3_decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    causal_gate_digest = sha256(args.causal_gate)
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        raise ValueError("max-new-cells must be positive.")
    source_commit = git_head(Path.cwd())
    if git_dirty(Path.cwd()):
        raise RuntimeError("P3 RULER evaluation requires a clean experiment source tree.")
    kvpress_root, model_snapshot = args.kvpress_root.resolve(), args.model_snapshot.resolve()
    if git_head(kvpress_root) != KVPRESS_REVISION or git_dirty(kvpress_root):
        raise ValueError("KVPress checkout does not match its clean frozen revision.")
    if model_snapshot.name != MODEL_REVISION:
        raise ValueError("Model snapshot does not match the frozen Qwen revision.")
    manifest = json.loads(args.manifest.read_text())
    if manifest["model"]["revision"] != MODEL_REVISION:
        raise ValueError("Experiment manifest model revision drifted.")
    verify_model_snapshot(model_snapshot, manifest)
    manifest_digest = sha256(args.manifest)
    runner_digest = sha256(Path(__file__).resolve())
    lengths = tuple(args.length or LENGTHS)
    arms = tuple(args.arm or ARMS)
    ratios = tuple(args.ratio) if args.ratio else None
    all_cells = cells(lengths, arms, ratios)
    datasets: dict[int, tuple[dict[str, Any], str, pd.DataFrame]] = {}
    generator_digest = sha256(Path(__file__).with_name("prepare_p3_ruler_dataset.py"))
    for length in lengths:
        path = args.dataset_root / str(length) / "dataset-manifest.json"
        data_manifest = json.loads(path.read_text())
        if (
            data_manifest["ruler"]["revision"] != RULER_REVISION
            or data_manifest["tokenizer"]["model_revision"] != MODEL_REVISION
            or data_manifest["generation"]["length_tokens"] != length
            or data_manifest["source"].get("dirty") is not False
            or data_manifest["source"].get("implementation_sha256") != generator_digest
        ):
            raise ValueError(f"Dataset provenance drifted for length {length}.")
        datasets[length] = (data_manifest, sha256(path), load_dataset(data_manifest))
    pending = [
        item
        for item in all_cells
        if not completed(
            cell_dir(args.output_root, item[0], item[1], item[3]),
            runner_digest,
            manifest_digest,
            datasets[item[0]][1],
            causal_gate_digest,
        )
    ]
    if args.max_new_cells is not None:
        pending = pending[: args.max_new_cells]
    if not pending:
        print(json.dumps({"status": "complete", "cells": len(all_cells)}, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("P3 RULER evaluation requires CUDA.")
    _gpu_lock = acquire_gpu_lock("p3-ruler-matrix")
    EvaluationConfig, EvaluationRunner, scorer = load_evaluator(kvpress_root)
    first_length = pending[0][0]
    base = EvaluationConfig(
        dataset="ruler",
        data_dir=str(first_length),
        model=str(model_snapshot),
        device="cuda:0",
        press_name="no_press",
        compression_ratio=0.0,
        output_dir=str(args.output_root),
        seed=args.seed,
    )
    runner = EvaluationRunner(base)
    runner._setup_press()
    runner._setup_model_pipeline()
    runtime_environment = environment()
    command = [sys.executable, *sys.argv]
    for completed_cells, (length, arm, press, ratio) in enumerate(pending, start=1):
        cell = cell_dir(args.output_root, length, arm, ratio)
        staging = cell.with_name(cell.name + f".tmp-{os.getpid()}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        started = time.monotonic()
        try:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
            config = EvaluationConfig(
                dataset="ruler",
                data_dir=str(length),
                model=str(model_snapshot),
                device="cuda:0",
                press_name=press,
                compression_ratio=ratio,
                output_dir=str(args.output_root),
                seed=args.seed,
            )
            runner.config = config
            runner._setup_press()
            runner.df = datasets[length][2].copy(deep=True)
            runner._run_inference()
            metrics = scorer(runner.df.copy(deep=True))
            runner.df["string_match"] = [
                example_score(task, prediction, references)
                for task, prediction, references in zip(
                    runner.df["task"],
                    runner.df["predicted_answer"],
                    runner.df["answer"],
                    strict=True,
                )
            ]
            torch.cuda.synchronize()
            predictions_path, metrics_path = staging / "predictions.csv", staging / "metrics.json"
            config_path = staging / "config.json"
            columns = [
                "row_id",
                "task",
                "question",
                "answer_prefix",
                "answer",
                "max_new_tokens",
                "predicted_answer",
                "string_match",
                "compression_ratio",
            ]
            runner.df[[name for name in columns if name in runner.df]].to_csv(
                predictions_path, index=False
            )
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
            outputs = {
                name: {"path": str(cell / path.name), "sha256": sha256(path)}
                for name, path in {
                    "predictions": predictions_path,
                    "metrics": metrics_path,
                    "config": config_path,
                }.items()
            }
            dataset_manifest_path = args.dataset_root / str(length) / "dataset-manifest.json"
            audit = {
                "schema_version": 1,
                "experiment_id": "p3-ruler-qwen3-1.7b-cell-v1",
                "status": "complete",
                "source": {
                    "commit": source_commit,
                    "dirty": False,
                    "implementation_sha256": runner_digest,
                },
                "cell": {"length_tokens": length, "arm": arm, "press": press, "ratio": ratio},
                "command": command,
                "seed": args.seed,
                "environment": runtime_environment,
                "upstreams": {
                    "kvpress_revision": KVPRESS_REVISION,
                    "ruler_revision": RULER_REVISION,
                    "model_revision": MODEL_REVISION,
                    "causal_gate": str(args.causal_gate),
                    "causal_gate_sha256": causal_gate_digest,
                    "p3_sequence_decision": p3_decision,
                },
                "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                "dataset_manifest": {
                    "path": str(dataset_manifest_path),
                    "sha256": datasets[length][1],
                },
                "measurements": {
                    "rows": len(runner.df),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
                "outputs": outputs,
            }
            (staging / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
            cell.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(cell)
            failure = cell.with_name(cell.name + ".failure.json")
            if failure.exists():
                failure.unlink()
            print(
                json.dumps({"completed": str(cell), "completed_cells": completed_cells}), flush=True
            )
        except BaseException as error:
            write_failure(cell, source_commit, command, started, error)
            raise


if __name__ == "__main__":
    main()
