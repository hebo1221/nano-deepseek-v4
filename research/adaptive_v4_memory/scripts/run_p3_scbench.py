from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_cross_family_sequence_gate import require_cross_family_sequence_gate
from p3_natural_workloads import build_scbench_workload
from p3_scbench_official import (
    ROUGE_REVISION,
    ROUGE_SCRIPT_SHA256,
    OfficialSCBenchScorer,
    load_official_components,
)
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_longbench_v2 import cache_bytes
from run_p3_mrcr import arm_config, atomic_json, failure_record, runtime_environment
from run_p3_natural_ruler import compatibility_arm_config
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from transformers import DynamicCache
from validate_p3_natural_adaptive_quota_scbench_manifest import (
    validate_manifest as validate_adaptive_scbench_manifest,
)
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "SCBench"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
ADAPTIVE_QUOTA_ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
DEFAULT_OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/scbench")
ADAPTIVE_QUOTA_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/scbench-qwen3-4b"
)
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "283310bb8c5ba6909dd9a6b1be087d2937f76f6d"
CODE_REVISION = "a4eb395f949ea39e871f9bc586d683390692c6be"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_adaptive_prerequisite(
    path: Path, *, experiment_id: str, predictions: int, label: str
) -> dict[str, str]:
    _require(path.is_file(), f"Adaptive SCBench prerequisite is unavailable: {label}.")
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    _require(payload.get("experiment_id") == experiment_id, f"Wrong prerequisite: {label}.")
    if experiment_id == "p3-natural-adaptive-quota-ruler-audit-v1":
        _require(
            payload.get("status") == "terminal"
            and audit.get("total_predictions") == predictions
            and audit.get("all_raw_records_verified") is True
            and audit.get("all_dependency_digests_verified") is True
            and audit.get("failure_accounting_complete") is True
            and audit.get("quota_physical_audits_verified") is True
            and audit.get("same_global_token_budget_verified") is True
            and audit.get("causal_layer_order_verified") is True,
            "Adaptive RULER prerequisite lacks verified quota evidence.",
        )
    elif experiment_id == "p3-natural-scbench-audit-v1":
        arms = payload.get("arms", {})
        _require(
            audit.get("all_raw_artifacts_verified") is True
            and audit.get("all_failure_accounting_complete") is True
            and audit.get("all_required_arms_input_paired") is True
            and audit.get("all_reported_scores_recomputed_from_raw_response") is True
            and sum(
                int(row.get("accounted_examples", -predictions))
                for row in arms.values()
                if isinstance(row, dict)
            )
            == predictions,
            "Baseline SCBench prerequisite is not a complete audited result.",
        )
    else:
        raise ValueError(f"Unsupported adaptive SCBench prerequisite: {experiment_id}.")
    return {"path": str(path), "sha256": sha256(path)}


def encode_segment(tokenizer: Any, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    _require(
        isinstance(ids, torch.Tensor) and ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0,
        "SCBench prompt segment tokenization is empty or malformed.",
    )
    return ids


def token_sequence_digest(segments: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        values = segment.detach().cpu().to(torch.int64).contiguous().numpy()
        digest.update(int(values.shape[1]).to_bytes(8, "big"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def prompt_sequence_digest(segments: list[str]) -> str:
    payload = json.dumps(segments, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_lengths(cache: Any) -> list[int]:
    return [cache.get_seq_length(index) for index in range(len(cache))]


@torch.inference_mode()
def prefill_cache(*, pipeline: Any, press: Any, input_ids: torch.Tensor) -> DynamicCache:
    cache = DynamicCache()
    values = input_ids.to(pipeline.model.device)
    with press(pipeline.model) if press is not None else contextlib.nullcontext():
        pipeline.model.model(input_ids=values, past_key_values=cache)
    return cache


@torch.inference_mode()
def append_prompt(
    *, pipeline: Any, cache: DynamicCache, input_ids: torch.Tensor, logical_position: int
) -> int:
    values = input_ids.to(pipeline.model.device)
    positions = torch.arange(
        logical_position,
        logical_position + values.shape[1],
        device=pipeline.model.device,
    ).unsqueeze(0)
    pipeline.model(
        input_ids=values,
        past_key_values=cache,
        position_ids=positions,
        num_logits_to_keep=1,
    )
    return cache_bytes(cache)


@torch.inference_mode()
def generate_turn(
    *,
    pipeline: Any,
    cache: DynamicCache,
    input_ids: torch.Tensor,
    logical_position: int,
    max_new_tokens: int,
) -> tuple[str, int, int, str]:
    values = input_ids.to(pipeline.model.device)
    positions = torch.arange(
        logical_position,
        logical_position + values.shape[1],
        device=pipeline.model.device,
    ).unsqueeze(0)
    outputs = pipeline.model(
        input_ids=values,
        past_key_values=cache,
        position_ids=positions,
        num_logits_to_keep=1,
    )
    prompt_lengths = cache_lengths(cache)
    resident_bytes = cache_bytes(cache)
    generated_ids = [outputs.logits[0, -1].argmax()]
    stop_ids = pipeline.model.generation_config.eos_token_id
    if not isinstance(stop_ids, list):
        stop_ids = [stop_ids]
    stopped = generated_ids[-1].item() in stop_ids
    next_position = positions[:, -1:] + 1
    while len(generated_ids) < max_new_tokens and not stopped:
        outputs = pipeline.model(
            input_ids=generated_ids[-1].view(1, 1),
            past_key_values=cache,
            position_ids=next_position + len(generated_ids) - 1,
        )
        generated_ids.append(outputs.logits[0, -1].argmax())
        stopped = generated_ids[-1].item() in stop_ids
    response = str(pipeline.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))
    pipeline._remove_answer_from_cache(cache, prompt_lengths)
    return (
        response,
        resident_bytes,
        len(generated_ids),
        "eos-or-special-token" if stopped else "max-new-tokens",
    )


def load_dependencies(
    *,
    manifest_path: Path,
    inventory_path: Path,
    source_inventory_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], Path]:
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["benchmarks"][BENCHMARK]
    _require(manifest["model"]["revision"] == MODEL_REVISION, "SCBench model drifted.")
    _require(
        contract["dataset"]["revision"] == DATASET_REVISION
        and contract["upstream_code"]["revision"] == CODE_REVISION,
        "SCBench frozen revisions drifted.",
    )
    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Natural dataset inventory is missing or dirty.",
    )
    observed = inventory["benchmarks"][BENCHMARK]
    _require(observed["revision"] == DATASET_REVISION, "SCBench inventory revision drifted.")
    expected_files = {row["path"]: row for row in contract["dataset"]["files"]}
    observed_files = {Path(row["path"]).parts[-2]: row for row in observed["files"]}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for relative, expected in expected_files.items():
        task = Path(relative).parts[0]
        metadata = observed_files.get(task)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing SCBench dataset task: {task}.")
        path = Path(metadata["path"])
        _require(
            path.is_file()
            and metadata["sha256"] == expected["sha256"]
            and sha256(path) == expected["sha256"],
            f"SCBench dataset artifact drifted: {task}.",
        )
        rows = pq.read_table(path).to_pylist()
        _require(len(rows) == expected["rows"], f"SCBench row count drifted: {task}.")
        rows_by_task[task] = rows

    source_inventory = json.loads(source_inventory_path.read_text())
    _require(
        source_inventory.get("experiment_id") == "p3-natural-source-inventory-v1"
        and source_inventory.get("status") == "verified"
        and source_inventory.get("source", {}).get("dirty") is False,
        "Natural source inventory is missing or dirty.",
    )
    source = source_inventory["benchmarks"][BENCHMARK]
    _require(
        source["revision"] == CODE_REVISION and source.get("clean_tracked_tree") is True,
        "SCBench source checkout drifted.",
    )
    source_root = Path(source["path"])
    expected_sources = contract["upstream_code"]["files_sha256"]
    observed_sources = {row["path"]: row["sha256"] for row in source["files"]}
    _require(observed_sources == expected_sources, "SCBench source inventory drifted.")
    for relative, digest in expected_sources.items():
        path = source_root / relative
        _require(path.is_file() and sha256(path) == digest, f"SCBench source drifted: {relative}.")

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
    return manifest, selection, rows_by_task, source_root


def iter_workloads(
    *,
    manifest: dict[str, Any],
    rows_by_task: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    prompt_module: Any,
) -> Iterator[tuple[str, str, int, dict[str, Any], dict[str, Any]]]:
    contract = manifest["benchmarks"][BENCHMARK]
    for mode in contract["modes"]:
        for task in contract["tasks"]:
            for row_index, row in enumerate(rows_by_task[task]):
                workload = build_scbench_workload(
                    row=row,
                    task=task,
                    mode=mode,
                    tokenizer=tokenizer,
                    official_module=prompt_module,
                )
                yield mode, task, row_index, row, workload


def _existing_records(
    progress: Path, partial: Path, identity: dict[str, Any], expected: int
) -> list[dict[str, Any]]:
    if not progress.exists() and not partial.exists():
        atomic_json(progress, identity)
        partial.mkdir(parents=True)
        return []
    if progress.is_file() and not partial.exists():
        partial.mkdir(parents=True)
    if partial.is_dir() and not any(partial.iterdir()) and not progress.exists():
        atomic_json(progress, identity)
    _require(progress.is_file() and partial.is_dir(), "Partial SCBench state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial SCBench provenance drifted.")
    parts = sorted(partial.glob("*.json"))
    _require(
        [part.name for part in parts] == [f"{index:06d}.json" for index in range(len(parts))],
        "Partial SCBench turn sequence drifted.",
    )
    records = [json.loads(part.read_text()) for part in parts]
    _require(len(records) <= expected, "Partial SCBench has too many records.")
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
        f"Completed SCBench provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed SCBench records drifted: {records}.",
    )
    return True


def base_record(
    *,
    example_id: str,
    arm: str,
    mode: str,
    task: str,
    row_index: int,
    turn: dict[str, Any],
    prompt_segments: list[str],
    token_segments: list[torch.Tensor],
    exact_tokens: int,
    config: dict[str, Any],
    revisions: dict[str, str],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "benchmark": BENCHMARK,
        "arm": arm,
        "mode": mode,
        "task": task,
        "row_index": row_index,
        "turn_index": turn["turn_index"],
        "subtask": turn["subtask"],
        "exact_input_tokens": exact_tokens,
        "generation_reserve_tokens": turn["generation_reserve_tokens"],
        "raw_prompt_sha256": prompt_sequence_digest(prompt_segments),
        "input_token_ids_sha256": token_sequence_digest(token_segments),
        "token_boundary_retreat": 0,
        "arm_config": config,
        "revisions": revisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B SCBench arms.")
    parser.add_argument("--cohort", choices=("baseline", "adaptive-quota"), default="baseline")
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
        "--source-inventory",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-sources/source-inventory.json"
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
        "--primary-core-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.strict.summary.json"
        ),
    )
    parser.add_argument(
        "--nine-seed-causal-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"),
    )
    parser.add_argument(
        "--adaptive-quota-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-scbench-v1.json"
        ),
    )
    parser.add_argument(
        "--adaptive-ruler-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "ruler-qwen3-4b.summary.json"
        ),
    )
    parser.add_argument(
        "--baseline-scbench-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural/scbench.summary.json"
        ),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--arm", action="append", choices=(*ARMS, *ADAPTIVE_QUOTA_ARMS))
    parser.add_argument("--max-new-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _require(
        args.max_new_examples is None or args.max_new_examples > 0,
        "max-new-examples must be positive.",
    )
    adaptive_manifest: dict[str, Any] | None = None
    adaptive_prerequisites: dict[str, dict[str, str]] = {}
    if args.cohort == "adaptive-quota":
        sequence_decision = require_cross_family_sequence_gate(
            primary_core=args.primary_core_summary,
            primary_causal=args.causal_gate,
            nine_seed_causal=args.nine_seed_causal_summary,
            fixed_selection=args.fixed_selection,
        )
        adaptive_manifest = json.loads(args.adaptive_quota_manifest.read_text())
        validate_adaptive_scbench_manifest(adaptive_manifest)
        adaptive_prerequisites = {
            "adaptive_ruler": load_adaptive_prerequisite(
                args.adaptive_ruler_summary,
                experiment_id="p3-natural-adaptive-quota-ruler-audit-v1",
                predictions=65_000,
                label="Qwen3-4B adaptive-quota RULER audit",
            ),
            "baseline_scbench": load_adaptive_prerequisite(
                args.baseline_scbench_summary,
                experiment_id="p3-natural-scbench-audit-v1",
                predictions=20_572,
                label="Qwen3-4B baseline SCBench audit",
            ),
        }
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = ADAPTIVE_QUOTA_OUTPUT_ROOT
    else:
        sequence_decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    if git_dirty(Path.cwd()):
        raise RuntimeError("SCBench requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, selection, rows_by_task, source_root = load_dependencies(
        manifest_path=args.manifest,
        inventory_path=args.dataset_inventory,
        source_inventory_path=args.source_inventory,
        selection_path=args.fixed_selection,
    )
    _require(
        args.seed == manifest["benchmarks"][BENCHMARK]["generation_seed"],
        "SCBench generation seed drifted from the frozen manifest.",
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])
    expected = manifest["benchmarks"][BENCHMARK]["expected_predictions_per_arm"]
    if adaptive_manifest is not None:
        _require(
            adaptive_manifest["benchmark"]["predictions_per_arm"] == expected,
            "Adaptive SCBench prediction count drifted from the base suite.",
        )
    runner_digest = sha256(Path(__file__))
    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    source_inventory_digest = sha256(args.source_inventory)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    adaptive_manifest_digest = (
        sha256(args.adaptive_quota_manifest) if adaptive_manifest is not None else None
    )
    scorer_bundle = hashlib.sha256(
        "\n".join(
            sorted(
                [
                    sha256(Path(__file__).with_name("p3_scbench_metrics.py")),
                    sha256(Path(__file__).with_name("p3_scbench_official.py")),
                    ROUGE_SCRIPT_SHA256,
                    *manifest["benchmarks"][BENCHMARK]["upstream_code"]["files_sha256"].values(),
                ]
            )
        ).encode()
    ).hexdigest()
    revisions = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "code_revision": CODE_REVISION,
        "scorer_sha256": scorer_bundle,
        "rouge_revision": ROUGE_REVISION,
        "rouge_script_sha256": ROUGE_SCRIPT_SHA256,
    }
    cohort_arms = ADAPTIVE_QUOTA_ARMS if args.cohort == "adaptive-quota" else ARMS
    selected_arms = tuple(args.arm or cohort_arms)
    _require(
        len(selected_arms) == len(set(selected_arms))
        and all(arm in cohort_arms for arm in selected_arms),
        f"Selected SCBench arms do not belong to the {args.cohort} cohort.",
    )
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "inventory_sha256": inventory_digest,
            "source_inventory_sha256": source_inventory_digest,
            "causal_gate_sha256": causal_digest,
            "fixed_selection_sha256": selection_digest,
            "cohort": args.cohort,
            "adaptive_quota_manifest_sha256": adaptive_manifest_digest,
            "adaptive_prerequisite_sha256": {
                name: metadata["sha256"] for name, metadata in adaptive_prerequisites.items()
            },
            "sequence_gate_dependency_sha256": {
                name: metadata["sha256"]
                for name, metadata in sequence_decision.get("dependencies", {}).items()
            },
            "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
            "scorer_bundle_sha256": scorer_bundle,
            "arm_config": (
                compatibility_arm_config(arm, selection, selection_digest)
                if args.cohort == "adaptive-quota"
                else arm_config(arm, selection, selection_digest)
            ),
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
    _require(torch.cuda.is_available(), "SCBench requires CUDA.")

    lock = acquire_gpu_lock("p3-scbench")
    try:
        from p3_protected_prefix_press import (
            wrap_same_budget_adaptive_quota_protected_prefix,
            wrap_same_budget_protected_prefix,
        )

        EvaluationConfig, EvaluationRunner, _unused = load_evaluator(kvpress_root)
        config = EvaluationConfig(
            dataset="longbench-v2",
            model=str(model_snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=args.seed,
            max_context_length=manifest["model"]["maximum_supported_context_tokens"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(config)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
        files = manifest["benchmarks"][BENCHMARK]["upstream_code"]["files_sha256"]
        prompt_module, repo_module, rouge_metric = load_official_components(source_root, files)
        scorer = OfficialSCBenchScorer(
            rows_by_task=rows_by_task,
            repo_module=repo_module,
            rouge_metric=rouge_metric,
        )
        environment = runtime_environment()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        maximum_context = manifest["model"]["maximum_supported_context_tokens"]

        for arm in pending:
            settings = identities[arm]["arm_config"]
            runner.config.press_name = settings["press_name"]
            runner.config.compression_ratio = settings["compression_ratio"]
            runner._setup_press()
            active_press: Any = runner.press
            compatibility_press: Any = None
            if args.cohort == "adaptive-quota":
                if arm == "fixed+pins":
                    compatibility_press = wrap_same_budget_protected_prefix(runner.press)
                else:
                    compatibility_press = wrap_same_budget_adaptive_quota_protected_prefix(
                        runner.press,
                        max_adjustment_fraction=settings["max_adjustment_fraction"],
                    )
                active_press = compatibility_press
            root = args.output_root / arm
            progress = root / "progress.json"
            partial = root / "record-parts"
            existing = _existing_records(progress, partial, identities[arm], expected)
            limit = expected
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            global_index = 0
            root.mkdir(parents=True, exist_ok=True)
            with contextlib.nullcontext():
                for mode, task, row_index, row, workload in iter_workloads(
                    manifest=manifest,
                    rows_by_task=rows_by_task,
                    tokenizer=tokenizer,
                    prompt_module=prompt_module,
                ):
                    turns = workload["turns"]
                    ids = [f"{mode}:{task}:{row_index}:{turn['turn_index']}" for turn in turns]
                    for offset, identifier in enumerate(ids):
                        position = global_index + offset
                        if position < len(existing):
                            _require(
                                existing[position]["example_id"] == identifier,
                                "Partial SCBench example order drifted.",
                            )
                    row_end = global_index + len(turns)
                    if row_end <= len(existing):
                        global_index = row_end
                        continue
                    if global_index >= limit:
                        break
                    start_turn = max(0, len(existing) - global_index)
                    stop_turn = min(len(turns), limit - global_index)
                    prompt_strings: list[str] = []
                    token_segments: list[torch.Tensor] = []
                    if mode == "multi-request":
                        shared = workload["shared_context"]
                        _require(
                            isinstance(shared, str) and bool(shared),
                            "SCBench shared context missing.",
                        )
                        prompt_strings.append(shared)
                        token_segments.append(encode_segment(tokenizer, shared))
                    else:
                        prompt_strings.append(turns[0]["prompt_segment"])
                        token_segments.append(encode_segment(tokenizer, turns[0]["prompt_segment"]))
                        for turn in turns[1:]:
                            prompt_strings.append(turn["prompt_segment"])
                            token_segments.append(encode_segment(tokenizer, turn["prompt_segment"]))

                    cache: DynamicCache | None = None
                    logical_position = 0
                    shared_metrics: dict[str, Any] = {
                        "initial_prefill_latency_ms": 0.0,
                        "initial_prefill_peak_hbm_bytes": 0,
                        "initial_prefill_hot_resident_bytes": 0,
                        "quota_physical_audit": None,
                    }
                    row_error: BaseException | None = None
                    row_blocked_by_unsupported = False
                    initial_tokens = int(token_segments[0].shape[1])
                    try:
                        initial_required = initial_tokens
                        if mode == "multi-turn":
                            initial_required += int(turns[0]["generation_reserve_tokens"])
                        if initial_required > maximum_context:
                            raise OverflowError("initial SCBench prompt exceeds model context")
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        prefill_started = time.perf_counter_ns()
                        if compatibility_press is not None:
                            span = settings["protected_prefix_token_span"]
                            compatibility_press.configure(
                                protected_start=span["start"], protected_end=span["end"]
                            )
                        if mode == "multi-request":
                            cache = prefill_cache(
                                pipeline=runner.pipeline,
                                press=active_press,
                                input_ids=token_segments[0],
                            )
                            logical_position = int(token_segments[0].shape[1])
                        else:
                            first = token_segments[0]
                            _require(first.shape[1] >= 2, "SCBench first prompt is too short.")
                            cache = prefill_cache(
                                pipeline=runner.pipeline,
                                press=active_press,
                                input_ids=first[:, :-1],
                            )
                            logical_position = int(first.shape[1]) - 1
                        compatibility_audit = (
                            compatibility_press.audit()
                            if compatibility_press is not None
                            else None
                        )
                        torch.cuda.synchronize()
                        shared_metrics = {
                            "initial_prefill_latency_ms": (time.perf_counter_ns() - prefill_started)
                            / 1_000_000.0,
                            "initial_prefill_peak_hbm_bytes": torch.cuda.max_memory_allocated(),
                            "initial_prefill_hot_resident_bytes": cache_bytes(cache),
                            "quota_physical_audit": compatibility_audit,
                        }
                        if mode == "multi-turn" and start_turn > 0:
                            append_prompt(
                                pipeline=runner.pipeline,
                                cache=cache,
                                input_ids=token_segments[0][:, -1:],
                                logical_position=logical_position,
                            )
                            logical_position += 1
                            for prior in range(1, start_turn):
                                append_prompt(
                                    pipeline=runner.pipeline,
                                    cache=cache,
                                    input_ids=token_segments[prior],
                                    logical_position=logical_position,
                                )
                                logical_position += int(token_segments[prior].shape[1])
                    except OverflowError:
                        row_blocked_by_unsupported = True
                    except BaseException as error:
                        row_error = error
                        torch.cuda.empty_cache()

                    for turn_index in range(start_turn, stop_turn):
                        turn = turns[turn_index]
                        if mode == "multi-request":
                            query = encode_segment(tokenizer, turn["prompt_segment"])
                            record_segments = [token_segments[0], query]
                            record_prompts = [prompt_strings[0], turn["prompt_segment"]]
                            exact_tokens = int(token_segments[0].shape[1] + query.shape[1])
                            turn_ids = query
                            turn_position = int(token_segments[0].shape[1])
                        else:
                            record_segments = token_segments[: turn_index + 1]
                            record_prompts = prompt_strings[: turn_index + 1]
                            exact_tokens = sum(int(item.shape[1]) for item in record_segments)
                            turn_ids = (
                                token_segments[0][:, -1:]
                                if turn_index == 0
                                else token_segments[turn_index]
                            )
                            turn_position = logical_position
                        identifier = ids[turn_index]
                        base = base_record(
                            example_id=identifier,
                            arm=arm,
                            mode=mode,
                            task=task,
                            row_index=row_index,
                            turn=turn,
                            prompt_segments=record_prompts,
                            token_segments=record_segments,
                            exact_tokens=exact_tokens,
                            config=settings,
                            revisions=revisions,
                        )
                        base.update(shared_metrics)
                        reserve = turn["generation_reserve_tokens"]
                        if row_blocked_by_unsupported or exact_tokens + reserve > maximum_context:
                            record = failure_record(
                                base,
                                failure_type="unsupported-context",
                                latency_ms=0.0,
                                peak_hbm_bytes=0,
                            )
                            if mode == "multi-turn":
                                row_blocked_by_unsupported = True
                        elif row_error is not None or cache is None:
                            record = failure_record(
                                base,
                                failure_type=(
                                    "oom"
                                    if isinstance(row_error, torch.cuda.OutOfMemoryError)
                                    else "runtime-error"
                                ),
                                latency_ms=0.0,
                                peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                error=row_error,
                            )
                        else:
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                            started = time.perf_counter_ns()
                            base_lengths = cache_lengths(cache)
                            response: str | None = None
                            resident = 0
                            generated: int | None = None
                            stop_reason: str | None = None
                            try:
                                response, resident, generated, stop_reason = generate_turn(
                                    pipeline=runner.pipeline,
                                    cache=cache,
                                    input_ids=turn_ids,
                                    logical_position=turn_position,
                                    max_new_tokens=reserve,
                                )
                                if mode == "multi-request":
                                    runner.pipeline._remove_answer_from_cache(cache, base_lengths)
                                else:
                                    logical_position += int(turn_ids.shape[1])
                                torch.cuda.synchronize()
                                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                                peak = torch.cuda.max_memory_allocated()
                                if not response.strip():
                                    record = failure_record(
                                        base,
                                        failure_type="empty-generation",
                                        latency_ms=latency_ms,
                                        peak_hbm_bytes=peak,
                                        hot_resident_bytes=resident,
                                    )
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
                                        "parsed_response": response,
                                        "score": score,
                                        "failure_type": None,
                                        "stop_reason": stop_reason,
                                        "generated_tokens_observed": generated,
                                        "latency_ms": latency_ms,
                                        "peak_hbm_bytes": peak,
                                        "hot_resident_bytes": resident,
                                        "scorer_detail": detail,
                                    }
                            except torch.cuda.OutOfMemoryError as error:
                                row_error = error
                                record = failure_record(
                                    base,
                                    failure_type="oom",
                                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                    peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                    error=error,
                                )
                                torch.cuda.empty_cache()
                            except Exception as error:
                                if response is None:
                                    row_error = error
                                record = failure_record(
                                    base,
                                    failure_type="runtime-error",
                                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                    peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                    hot_resident_bytes=resident,
                                    raw_response=response or "",
                                    parsed_response=response,
                                    stop_reason=stop_reason,
                                    generated_tokens_observed=generated,
                                    error=error,
                                )
                        atomic_json(partial / f"{global_index + turn_index:06d}.json", record)
                        print(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "completed_examples": global_index + turn_index + 1,
                                    "total": expected,
                                }
                            ),
                            flush=True,
                        )
                    global_index = row_end
                    if global_index >= limit:
                        break
            if limit < expected:
                continue
            _require(global_index == expected, "SCBench prediction grid did not close.")
            completed = _existing_records(progress, partial, identities[arm], expected)
            _require(len(completed) == expected, "SCBench atomic turn grid did not close.")
            records = root / "records.jsonl"
            temporary = root / ".records.jsonl.tmp"
            temporary.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in completed)
            )
            temporary.replace(records)
            cell = {
                "schema_version": 1,
                "experiment_id": "p3-natural-benchmark-arm-cell-v1",
                "benchmark": BENCHMARK,
                "arm": arm,
                "cohort": args.cohort,
                "status": "terminal",
                "source": {
                    "commit": source_commit,
                    "dirty": False,
                    "implementation_sha256": runner_digest,
                },
                "run_identity": identities[arm],
                "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                "adaptive_quota_manifest": (
                    {
                        "path": str(args.adaptive_quota_manifest),
                        "sha256": adaptive_manifest_digest,
                    }
                    if adaptive_manifest_digest is not None
                    else None
                ),
                "adaptive_prerequisites": adaptive_prerequisites,
                "causal_gate": {"path": str(args.causal_gate), "sha256": causal_digest},
                "dataset_inventory": {
                    "path": str(args.dataset_inventory),
                    "sha256": inventory_digest,
                },
                "evaluation_source_inventory": {
                    "path": str(args.source_inventory),
                    "sha256": source_inventory_digest,
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
            progress.unlink()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
