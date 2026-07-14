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
from p3_natural_workloads import build_scbench_workload, sha256
from p3_scbench_official import OfficialSCBenchScorer, load_official_components
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from transformers import DynamicCache

BENCHMARK = "SCBench"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
MODES = ("multi-turn", "multi-request")
EXPECTED_CONTEXTS = 922
EXPECTED_TURNS_PER_MODE = 5143
EXPECTED_PREDICTIONS = 10286
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "283310bb8c5ba6909dd9a6b1be087d2937f76f6d"
CODE_REVISION = "a4eb395f949ea39e871f9bc586d683390692c6be"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def arm_config(arm: str, selection: dict[str, Any], selection_digest: str) -> dict[str, Any]:
    if arm == "native-dense":
        return {"press_name": "no_press", "compression_ratio": 0.0}
    _require(arm == "strongest-memory-matched-fixed", f"Unknown SCBench arm: {arm}.")
    return {
        "press_name": selection["selected_arm"],
        "compression_ratio": selection["selected_compression_ratio"],
        "selection_sha256": selection_digest,
    }


def cache_lengths(cache: Any) -> list[int]:
    return [cache.get_seq_length(index) for index in range(len(cache))]


def cache_bytes(cache: Any) -> int:
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def encode_exact(tokenizer: Any, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    _require(
        getattr(ids, "ndim", None) == 2 and ids.shape[0] == 1 and ids.shape[1] > 0,
        "SCBench prompt segment did not produce one non-empty token sequence.",
    )
    return ids


@torch.inference_mode()
def prefill(
    *, pipeline: Any, press: Any, input_ids: torch.Tensor, cache: Any
) -> None:
    with press(pipeline.model) if press is not None else contextlib.nullcontext():
        pipeline.model.model(
            input_ids=input_ids.to(pipeline.model.device),
            past_key_values=cache,
        )


@torch.inference_mode()
def generate_and_restore(
    *,
    pipeline: Any,
    input_ids: torch.Tensor,
    cache: Any,
    logical_position_start: int,
    max_new_tokens: int,
    retain_input: bool,
) -> tuple[str, int, str]:
    """Greedily decode and retain either the input prefix or the prior cache only."""
    _require(max_new_tokens > 0, "SCBench generation reserve must be positive.")
    before = cache_lengths(cache)
    device_ids = input_ids.to(pipeline.model.device)
    position_ids = torch.arange(
        logical_position_start,
        logical_position_start + device_ids.shape[1],
        device=pipeline.model.device,
    ).unsqueeze(0)
    outputs = pipeline.model(
        input_ids=device_ids,
        past_key_values=cache,
        position_ids=position_ids,
        num_logits_to_keep=1,
    )
    after_input = cache_lengths(cache)
    generated = [outputs.logits[0, -1].argmax()]
    stop_ids = pipeline.model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    stopped = generated[-1].item() in stop_ids
    next_position = position_ids[:, -1:] + 1
    for offset in range(max_new_tokens - 1):
        if stopped:
            break
        outputs = pipeline.model(
            input_ids=generated[-1].reshape(1, 1),
            past_key_values=cache,
            position_ids=next_position + offset,
        )
        token = outputs.logits[0, -1].argmax()
        generated.append(token)
        stopped = token.item() in stop_ids
    pipeline._remove_answer_from_cache(cache, after_input if retain_input else before)
    response = str(
        pipeline.tokenizer.decode(torch.stack(generated), skip_special_tokens=True)
    )
    return response, len(generated), "eos-or-special-token" if stopped else "max-new-tokens"


def load_dependencies(
    *,
    manifest_path: Path,
    inventory_path: Path,
    source_inventory_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], Path, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["model"]["revision"] == MODEL_REVISION, "Model revision drifted.")
    contract = manifest["benchmarks"][BENCHMARK]
    _require(contract["dataset"]["revision"] == DATASET_REVISION, "Dataset revision drifted.")
    _require(contract["upstream_code"]["revision"] == CODE_REVISION, "Code revision drifted.")

    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Natural dataset inventory is missing or dirty.",
    )
    observed = inventory["benchmarks"][BENCHMARK]
    _require(observed["revision"] == DATASET_REVISION, "SCBench inventory revision drifted.")
    expected = {row["path"]: row for row in contract["dataset"]["files"]}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    _require(len(observed["files"]) == len(expected), "SCBench file count drifted.")
    for metadata in observed["files"]:
        path = Path(metadata["path"])
        relative = "/".join(path.parts[-2:])
        _require(relative in expected, f"Unexpected SCBench file: {relative}.")
        _require(
            path.is_file()
            and sha256(path) == expected[relative]["sha256"]
            and metadata["sha256"] == expected[relative]["sha256"],
            f"SCBench dataset file drifted: {path}.",
        )
        task = Path(relative).parts[0]
        records = pq.read_table(path).to_pylist()
        _require(
            len(records) == contract["tasks"][task]["rows"]
            and all(isinstance(row, dict) for row in records),
            f"SCBench task row contract drifted: {task}.",
        )
        rows_by_task[task] = records
    _require(set(rows_by_task) == set(contract["tasks"]), "SCBench task set drifted.")
    contexts = sum(len(records) for records in rows_by_task.values())
    turns = sum(
        len(row["multi_turns"])
        for records in rows_by_task.values()
        for row in records
    )
    _require(
        contexts == EXPECTED_CONTEXTS and turns == EXPECTED_TURNS_PER_MODE,
        "SCBench context or turn total drifted.",
    )

    sources = json.loads(source_inventory_path.read_text())
    _require(
        sources.get("experiment_id") == "p3-natural-source-inventory-v1"
        and sources.get("status") == "verified"
        and sources.get("source", {}).get("dirty") is False,
        "Natural source inventory is missing or dirty.",
    )
    source = sources["benchmarks"][BENCHMARK]
    _require(
        source.get("revision") == CODE_REVISION and source.get("clean_tracked_tree") is True,
        "SCBench source checkout drifted.",
    )
    source_root = Path(source["path"])
    file_digests = {row["path"]: row["sha256"] for row in source["files"]}
    _require(
        file_digests == contract["upstream_code"]["files_sha256"],
        "SCBench source digest set drifted.",
    )
    for relative, digest in file_digests.items():
        _require(
            (source_root / relative).is_file() and sha256(source_root / relative) == digest,
            f"SCBench source file drifted: {relative}.",
        )

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
        "Fixed baseline selection is outside the preregistered set.",
    )
    return manifest, selection, rows_by_task, source_root, file_digests


def verify_model_snapshot(path: Path, manifest: dict[str, Any]) -> None:
    expected = manifest["model"]["snapshot_files_sha256"]
    actual = {item.name for item in path.iterdir() if item.is_file()}
    _require(actual == set(expected), "Model snapshot file set drifted.")
    for name, digest in expected.items():
        _require(sha256(path / name) == digest, f"Model snapshot drifted: {name}.")


def conversation_id(task: str, row_index: int, row: dict[str, Any], mode: str) -> str:
    digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{mode}:{task}:{row_index}:{digest}"


def conversation_plan(
    rows_by_task: dict[str, list[dict[str, Any]]], modes: tuple[str, ...] = MODES
) -> list[tuple[str, int, dict[str, Any], str]]:
    return [
        (task, index, row, mode)
        for mode in modes
        for task in sorted(rows_by_task)
        for index, row in enumerate(rows_by_task[task])
    ]


def _existing_conversations(
    progress: Path, partial: Path, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    if not progress.exists() and not partial.exists():
        atomic_json(progress, identity)
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.touch()
        return []
    if progress.is_file() and not partial.exists():
        partial.touch()
    _require(progress.is_file() and partial.is_file(), "Partial SCBench state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial SCBench provenance drifted.")
    payloads = [json.loads(line) for line in partial.read_text().splitlines() if line]
    _require(len(payloads) <= EXPECTED_CONTEXTS * len(MODES), "Too many SCBench conversations.")
    _require(
        all(isinstance(payload.get("records"), list) for payload in payloads),
        "Partial SCBench conversation is malformed.",
    )
    return payloads


def flatten_conversations(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for payload in payloads for record in payload["records"]]


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
        f"Completed SCBench provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed SCBench records drifted: {records}.",
    )
    return True


def failure_record(base: dict[str, Any], failure_type: str, error: BaseException | None = None) -> dict[str, Any]:
    return {
        **base,
        "status": "failure",
        "raw_response": "",
        "parsed_response": None,
        "score": None,
        "score_detail": None,
        "failure_type": failure_type,
        "stop_reason": failure_type,
        "generated_tokens_observed": 0,
        "latency_ms": 0.0,
        "peak_hbm_bytes": 0,
        "hot_resident_bytes": 0,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }


def base_record(
    *,
    identifier: str,
    task: str,
    row_index: int,
    mode: str,
    turn: dict[str, Any],
    turn_index: int,
    arm: str,
    config: dict[str, Any],
    prompt_sha256: str,
    exact_input_tokens: int,
    cumulative_input_tokens: int,
    revisions: dict[str, str],
) -> dict[str, Any]:
    return {
        "example_id": f"{identifier}:turn:{turn_index}",
        "conversation_id": identifier,
        "benchmark": BENCHMARK,
        "task": task,
        "subtask": turn["subtask"],
        "row_index": row_index,
        "mode": mode,
        "turn_index": turn_index,
        "exact_input_tokens": exact_input_tokens,
        "cumulative_input_tokens": cumulative_input_tokens,
        "generation_reserve_tokens": turn["generation_reserve_tokens"],
        "raw_prompt_sha256": prompt_sha256,
        "arm": arm,
        "arm_config": config,
        "revisions": revisions,
    }


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
        "evaluate": importlib.metadata.version("evaluate"),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def run_conversation(
    *,
    pipeline: Any,
    press: Any,
    scorer: OfficialSCBenchScorer,
    official_module: Any,
    tokenizer: Any,
    task: str,
    row_index: int,
    row: dict[str, Any],
    mode: str,
    arm: str,
    config: dict[str, Any],
    maximum_context: int,
    revisions: dict[str, str],
) -> dict[str, Any]:
    workload = build_scbench_workload(
        row=row,
        task=task,
        mode=mode,
        tokenizer=tokenizer,
        official_module=official_module,
    )
    identifier = conversation_id(task, row_index, row, mode)
    turns = workload["turns"]
    cache = DynamicCache()
    records: list[dict[str, Any]] = []
    cumulative = 0
    hot_bytes = 0
    shared = workload["shared_context"]
    logical_prompt = hashlib.sha256()
    try:
        if mode == "multi-request":
            _require(isinstance(shared, str), "SCBench multi-request context is missing.")
            shared_ids = encode_exact(tokenizer, shared)
            cumulative = int(shared_ids.shape[1])
            logical_prompt.update(shared.encode())
            prefill(pipeline=pipeline, press=press, input_ids=shared_ids, cache=cache)
            hot_bytes = cache_bytes(cache)
        for turn_index, turn in enumerate(turns):
            segment = turn["prompt_segment"]
            segment_ids = encode_exact(tokenizer, segment)
            segment_tokens = int(segment_ids.shape[1])
            if mode == "multi-turn" and turn_index == 0:
                if segment_tokens < 2:
                    raise ValueError("First SCBench multi-turn prompt is too short.")
                prefill(
                    pipeline=pipeline,
                    press=press,
                    input_ids=segment_ids[:, :-1],
                    cache=cache,
                )
                hot_bytes = cache_bytes(cache)
                generation_ids = segment_ids[:, -1:]
                logical_start = segment_tokens - 1
            else:
                generation_ids = segment_ids
                logical_start = cumulative
            next_cumulative = cumulative + segment_tokens
            turn_prompt = logical_prompt.copy()
            turn_prompt.update(segment.encode())
            prompt_digest = turn_prompt.hexdigest()
            base = base_record(
                identifier=identifier,
                task=task,
                row_index=row_index,
                mode=mode,
                turn=turn,
                turn_index=turn_index,
                arm=arm,
                config=config,
                prompt_sha256=prompt_digest,
                exact_input_tokens=(cumulative + segment_tokens if mode == "multi-request" else segment_tokens),
                cumulative_input_tokens=next_cumulative,
                revisions=revisions,
            )
            if next_cumulative + turn["generation_reserve_tokens"] > maximum_context:
                records.append(failure_record(base, "unsupported-context"))
                if mode == "multi-turn":
                    logical_prompt.update(segment.encode())
                    cumulative = next_cumulative
                continue
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            try:
                response, generated, stop_reason = generate_and_restore(
                    pipeline=pipeline,
                    input_ids=generation_ids,
                    cache=cache,
                    logical_position_start=logical_start,
                    max_new_tokens=turn["generation_reserve_tokens"],
                    retain_input=mode == "multi-turn",
                )
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                if not response.strip():
                    record = failure_record(base, "empty-generation")
                    record["latency_ms"] = latency_ms
                    record["peak_hbm_bytes"] = torch.cuda.max_memory_allocated()
                    record["hot_resident_bytes"] = hot_bytes
                else:
                    score, detail = scorer.score(
                        task=task,
                        row=row,
                        turn=row["multi_turns"][turn_index],
                        prediction=response,
                        subtask=turn["subtask"],
                    )
                    record = {
                        **base,
                        "status": "scored",
                        "raw_response": response,
                        "parsed_response": None,
                        "score": score,
                        "score_detail": detail,
                        "failure_type": None,
                        "stop_reason": stop_reason,
                        "generated_tokens_observed": generated,
                        "latency_ms": latency_ms,
                        "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
                        "hot_resident_bytes": hot_bytes,
                        "error_type": None,
                        "error": None,
                    }
            except torch.cuda.OutOfMemoryError as error:
                record = failure_record(base, "oom", error)
                record["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
                record["peak_hbm_bytes"] = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
            except Exception as error:
                record = failure_record(base, "runtime-error", error)
                record["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000.0
                record["peak_hbm_bytes"] = torch.cuda.max_memory_allocated()
            records.append(record)
            if mode == "multi-turn":
                logical_prompt.update(segment.encode())
                cumulative = next_cumulative
    except Exception as error:
        for turn_index in range(len(records), len(turns)):
            turn = turns[turn_index]
            base = base_record(
                identifier=identifier,
                task=task,
                row_index=row_index,
                mode=mode,
                turn=turn,
                turn_index=turn_index,
                arm=arm,
                config=config,
                prompt_sha256=hashlib.sha256(turn["prompt_segment"].encode()).hexdigest(),
                exact_input_tokens=0,
                cumulative_input_tokens=cumulative,
                revisions=revisions,
            )
            records.append(failure_record(base, "conversation-runtime-error", error))
    _require(len(records) == len(turns), "SCBench conversation did not retain every turn.")
    return {"conversation_id": identifier, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B SCBench arms.")
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-data/dataset-inventory.json"),
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-sources/source-inventory.json"),
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/scbench"),
    )
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--max-new-conversations", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _require(
        args.max_new_conversations is None or args.max_new_conversations > 0,
        "max-new-conversations must be positive.",
    )
    decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    _require(not git_dirty(Path.cwd()), "SCBench evaluation requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, selection, rows_by_task, source_root, source_digests = load_dependencies(
        manifest_path=args.manifest,
        inventory_path=args.dataset_inventory,
        source_inventory_path=args.source_inventory,
        selection_path=args.fixed_selection,
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_model_snapshot(model_snapshot, manifest)
    plan = conversation_plan(rows_by_task)
    _require(len(plan) == EXPECTED_CONTEXTS * len(MODES), "SCBench conversation total drifted.")
    selection_digest = sha256(args.fixed_selection)
    runner_digest = sha256(Path(__file__).resolve())
    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    source_inventory_digest = sha256(args.source_inventory)
    causal_digest = sha256(args.causal_gate)
    scorer_digest = sha256(Path(__file__).with_name("p3_scbench_official.py"))
    revisions = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "code_revision": CODE_REVISION,
        "scorer_sha256": scorer_digest,
    }
    selected_arms = tuple(args.arm or ARMS)
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "inventory_sha256": inventory_digest,
            "source_inventory_sha256": source_inventory_digest,
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
    _require(torch.cuda.is_available(), "SCBench evaluation requires CUDA.")
    gpu_lock = acquire_gpu_lock("p3-scbench")
    try:
        EvaluationConfig, EvaluationRunner, _ = load_evaluator(kvpress_root)
        base = EvaluationConfig(
            dataset="scbench",
            model=str(model_snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=args.seed,
            max_context_length=manifest["model"]["maximum_supported_context_tokens"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(base)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
        official, repo_module, rouge = load_official_components(source_root, source_digests)
        scorer = OfficialSCBenchScorer(
            rows_by_task=rows_by_task,
            repo_module=repo_module,
            rouge_metric=rouge,
        )
        environment = runtime_environment()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for arm in pending:
            config = identities[arm]["arm_config"]
            runner.config.press_name = config["press_name"]
            runner.config.compression_ratio = config["compression_ratio"]
            runner._setup_press()
            root = args.output_root / arm
            progress = root / "progress.json"
            partial = root / "conversations.jsonl"
            existing = _existing_conversations(progress, partial, identities[arm])
            _require(
                all(
                    payload["conversation_id"]
                    == conversation_id(task, index, row, mode)
                    for payload, (task, index, row, mode) in zip(existing, plan, strict=False)
                ),
                "Partial SCBench conversation order drifted.",
            )
            limit = len(plan)
            if args.max_new_conversations is not None:
                limit = min(limit, len(existing) + args.max_new_conversations)
            root.mkdir(parents=True, exist_ok=True)
            with partial.open("a") as handle:
                for plan_index in range(len(existing), limit):
                    task, row_index, row, mode = plan[plan_index]
                    torch.cuda.empty_cache()
                    payload = run_conversation(
                        pipeline=runner.pipeline,
                        press=runner.press,
                        scorer=scorer,
                        official_module=official,
                        tokenizer=tokenizer,
                        task=task,
                        row_index=row_index,
                        row=row,
                        mode=mode,
                        arm=arm,
                        config=config,
                        maximum_context=manifest["model"]["maximum_supported_context_tokens"],
                        revisions=revisions,
                    )
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "completed_conversations": plan_index + 1,
                                "total_conversations": len(plan),
                            }
                        ),
                        flush=True,
                    )
            if limit < len(plan):
                continue
            payloads = _existing_conversations(progress, partial, identities[arm])
            records = flatten_conversations(payloads)
            _require(len(records) == EXPECTED_PREDICTIONS, "SCBench prediction total drifted.")
            records_path = root / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
            )
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
                "dataset_inventory": {"path": str(args.dataset_inventory), "sha256": inventory_digest},
                "evaluation_source_inventory": {
                    "path": str(args.source_inventory),
                    "sha256": source_inventory_digest,
                },
                "fixed_baseline_selection": {
                    "path": str(args.fixed_selection),
                    "sha256": selection_digest,
                },
                "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
                "p3_sequence_decision": decision,
                "environment": environment,
                "raw_records": {"path": str(records_path), "sha256": sha256(records_path)},
            }
            atomic_json(root / "cell.json", cell)
            progress.unlink()
    finally:
        gpu_lock.close()


if __name__ == "__main__":
    main()
