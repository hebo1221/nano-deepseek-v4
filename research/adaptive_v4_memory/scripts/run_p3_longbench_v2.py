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
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_cross_family_sequence_gate import require_cross_family_sequence_gate
from p3_natural_metrics import extract_longbench_v2_choice, score_longbench_v2
from p3_natural_workloads import (
    build_longbench_v2_segments,
    encode_rendered_segments_exact,
    render_chat_split_user_content,
)
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_natural_ruler import compatibility_arm_config
from run_p3_ruler_matrix import (
    KVPRESS_REVISION,
    git_dirty,
    git_head,
    kvpress_runtime_binding,
    load_evaluator,
)
from summarize_p3_natural_suite import sha256
from transformers import DynamicCache
from validate_p3_natural_adaptive_quota_longbench_v2_manifest import (
    validate_manifest as validate_adaptive_longbench_manifest,
)
from verify_p3_natural_model import verify_snapshot

BENCHMARK = "LongBench-v2"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
ADAPTIVE_QUOTA_ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural/longbench-v2"
)
ADAPTIVE_QUOTA_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
    "longbench-v2-qwen3-4b"
)
GENERATION_RESERVE = 128
EXPECTED_EXAMPLES = 503
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_adaptive_prerequisite(
    path: Path, *, experiment_id: str, predictions: int, label: str
) -> dict[str, str]:
    _require(path.is_file(), f"Adaptive LongBench v2 prerequisite is unavailable: {label}.")
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
    elif experiment_id == "p3-natural-longbench-v2-audit-v1":
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
            "Baseline LongBench v2 prerequisite is not a complete audited result.",
        )
    else:
        raise ValueError(f"Unsupported adaptive LongBench v2 prerequisite: {experiment_id}.")
    return {"path": str(path), "sha256": sha256(path)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prompt_parts(row: dict[str, Any], template: str) -> tuple[str, str]:
    return build_longbench_v2_segments(row, template)


def rendered_input(tokenizer: Any, context: str, question: str) -> dict[str, Any]:
    rendered_context, rendered_question = render_chat_split_user_content(
        tokenizer, context, question
    )
    context_ids, question_ids, boundary_retreat = encode_rendered_segments_exact(
        tokenizer, rendered_context, rendered_question
    )
    exact_tokens = int(context_ids.shape[1] + question_ids.shape[1])
    prompt_digest = hashlib.sha256((rendered_context + rendered_question).encode()).hexdigest()
    token_ids = torch.cat((context_ids, question_ids), dim=1)
    token_ids_digest = hashlib.sha256(
        token_ids.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
    ).hexdigest()
    return {
        "context_ids": context_ids,
        "question_ids": question_ids,
        "exact_input_tokens": exact_tokens,
        "raw_prompt_sha256": prompt_digest,
        "input_token_ids_sha256": token_ids_digest,
        "token_boundary_retreat": boundary_retreat,
    }


def load_rows(path: Path, template: str, expected: int = EXPECTED_EXAMPLES) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    _require(
        isinstance(payload, list) and len(payload) == expected,
        f"LongBench v2 must contain exactly {expected} rows.",
    )
    identifiers: set[str] = set()
    for row in payload:
        _require(isinstance(row, dict), "LongBench v2 row is not an object.")
        identifier = row.get("_id")
        _require(
            isinstance(identifier, str) and bool(identifier) and identifier not in identifiers,
            "LongBench v2 example id is missing or duplicated.",
        )
        identifiers.add(identifier)
        _require(row.get("answer") in {"A", "B", "C", "D"}, "Invalid answer label.")
        prompt_parts(row, template)
    return payload


def load_dependencies(
    *,
    manifest_path: Path,
    inventory_path: Path,
    source_inventory_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, str, Path]:
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["model"]["revision"] == MODEL_REVISION, "Model revision drifted.")
    contract = manifest["benchmarks"][BENCHMARK]
    _require(contract["dataset"]["revision"] == DATASET_REVISION, "Dataset revision drifted.")
    _require(contract["upstream_code"]["revision"] == CODE_REVISION, "Code revision drifted.")
    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Natural dataset inventory is not a clean frozen artifact.",
    )
    inventory_entry = inventory["benchmarks"][BENCHMARK]
    _require(inventory_entry["revision"] == DATASET_REVISION, "Inventory revision drifted.")
    files = inventory_entry["files"]
    _require(len(files) == 1, "LongBench v2 inventory must contain one file.")
    dataset_path = Path(files[0]["path"])
    _require(
        dataset_path.is_file()
        and sha256(dataset_path) == contract["dataset"]["files"][0]["sha256"]
        and sha256(dataset_path) == files[0]["sha256"],
        "LongBench v2 dataset artifact drifted.",
    )
    source_inventory = json.loads(source_inventory_path.read_text())
    _require(
        source_inventory.get("experiment_id") == "p3-natural-source-inventory-v1"
        and source_inventory.get("status") == "verified"
        and source_inventory.get("source", {}).get("dirty") is False,
        "Natural evaluation source inventory is missing or dirty.",
    )
    source_entry = source_inventory["benchmarks"][BENCHMARK]
    _require(
        source_entry.get("revision") == CODE_REVISION
        and source_entry.get("clean_tracked_tree") is True,
        "LongBench v2 source checkout drifted.",
    )
    source_root = Path(source_entry["path"])
    template_path = source_root / "prompts/0shot.txt"
    source_files = {row["path"]: row["sha256"] for row in source_entry["files"]}
    _require(
        template_path.is_file()
        and source_files.get("prompts/0shot.txt") == sha256(template_path)
        and source_files["prompts/0shot.txt"]
        == contract["upstream_code"]["files_sha256"]["prompts/0shot.txt"],
        "Pinned LongBench v2 prompt template drifted.",
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
    return (
        manifest,
        inventory,
        selection,
        dataset_path,
        template_path.read_text(),
        template_path,
    )


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
    cache_lengths = [cache.get_seq_length(index) for index in range(len(cache))]
    response = pipeline.generate_answer(
        question_ids=question_ids,
        cache=cache,
        context_length=original_context_length,
        max_new_tokens=max_new_tokens,
    )
    pipeline._remove_answer_from_cache(cache, cache_lengths)
    return response, resident_bytes


def arm_config(arm: str, selection: dict[str, Any], selection_digest: str) -> dict[str, Any]:
    if arm == "native-dense":
        return {"press_name": "no_press", "compression_ratio": 0.0}
    _require(arm == "strongest-memory-matched-fixed", f"Unknown natural arm: {arm}")
    return {
        "press_name": selection["selected_arm"],
        "compression_ratio": selection["selected_compression_ratio"],
        "selection_sha256": selection_digest,
    }


def base_record(
    *,
    row: dict[str, Any],
    arm: str,
    rendered: dict[str, Any],
    config: dict[str, Any],
    revisions: dict[str, str],
) -> dict[str, Any]:
    return {
        "example_id": row["_id"],
        "benchmark": BENCHMARK,
        "arm": arm,
        "exact_input_tokens": rendered["exact_input_tokens"],
        "generation_reserve_tokens": GENERATION_RESERVE,
        "raw_prompt_sha256": rendered["raw_prompt_sha256"],
        "input_token_ids_sha256": rendered["input_token_ids_sha256"],
        "token_boundary_retreat": rendered["token_boundary_retreat"],
        "arm_config": config,
        "revisions": revisions,
        "domain": row.get("domain"),
        "sub_domain": row.get("sub_domain"),
        "difficulty": row.get("difficulty"),
        "length_stratum": row.get("length"),
    }


def failure_record(
    base: dict[str, Any],
    *,
    failure_type: str,
    latency_ms: float,
    peak_hbm_bytes: int,
    hot_resident_bytes: int = 0,
    raw_response: str = "",
    parsed_response: str | None = None,
    stop_reason: str | None = None,
    generated_tokens_observed: int | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record = {
        **base,
        "status": "failure",
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "score": None,
        "failure_type": failure_type,
        "stop_reason": stop_reason or failure_type,
        "latency_ms": latency_ms,
        "peak_hbm_bytes": peak_hbm_bytes,
        "hot_resident_bytes": hot_resident_bytes,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }
    if generated_tokens_observed is not None:
        record["generated_tokens_observed"] = generated_tokens_observed
    return record


def _existing_progress(
    progress_path: Path,
    records_path: Path,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    if not progress_path.exists() and not records_path.exists():
        atomic_json(progress_path, identity)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.touch()
        return []
    if progress_path.is_file() and not records_path.exists():
        records_path.touch()
    if records_path.is_file() and records_path.stat().st_size == 0 and not progress_path.exists():
        atomic_json(progress_path, identity)
    _require(progress_path.is_file() and records_path.is_file(), "Partial arm state is incomplete.")
    _require(json.loads(progress_path.read_text()) == identity, "Partial arm provenance drifted.")
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    _require(len(records) <= EXPECTED_EXAMPLES, "Partial arm has too many records.")
    return records


def _completed(
    cell_path: Path,
    *,
    arm: str,
    identity: dict[str, Any],
) -> bool:
    if not cell_path.is_file():
        return False
    cell = json.loads(cell_path.read_text())
    if (
        cell.get("experiment_id") != "p3-natural-benchmark-arm-cell-v1"
        or cell.get("benchmark") != BENCHMARK
        or cell.get("arm") != arm
        or cell.get("status") != "terminal"
        or cell.get("run_identity") != identity
    ):
        raise ValueError(f"Completed natural arm provenance drifted: {cell_path}")
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed natural arm records drifted: {records}",
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
        "kvpress_binding": kvpress_runtime_binding(),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B LongBench v2 arms.")
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
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--source-inventory",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-sources/source-inventory.json"
        ),
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
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longbench-v2-v1.json"
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
        "--baseline-longbench-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural/longbench-v2.summary.json"
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
        decision = require_cross_family_sequence_gate(
            primary_core=args.primary_core_summary,
            primary_causal=args.causal_gate,
            nine_seed_causal=args.nine_seed_causal_summary,
            fixed_selection=args.fixed_selection,
        )
        adaptive_manifest = json.loads(args.adaptive_quota_manifest.read_text())
        validate_adaptive_longbench_manifest(adaptive_manifest)
        adaptive_prerequisites = {
            "adaptive_ruler": load_adaptive_prerequisite(
                args.adaptive_ruler_summary,
                experiment_id="p3-natural-adaptive-quota-ruler-audit-v1",
                predictions=65_000,
                label="Qwen3-4B adaptive-quota RULER audit",
            ),
            "baseline_longbench_v2": load_adaptive_prerequisite(
                args.baseline_longbench_summary,
                experiment_id="p3-natural-longbench-v2-audit-v1",
                predictions=1_006,
                label="Qwen3-4B baseline LongBench v2 audit",
            ),
        }
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = ADAPTIVE_QUOTA_OUTPUT_ROOT
    else:
        decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    if git_dirty(Path.cwd()):
        raise RuntimeError("LongBench v2 evaluation requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, _inventory, selection, dataset_path, prompt_template, prompt_path = load_dependencies(
        manifest_path=args.manifest,
        inventory_path=args.dataset_inventory,
        source_inventory_path=args.source_inventory,
        selection_path=args.fixed_selection,
    )
    _require(
        args.seed == manifest["benchmarks"][BENCHMARK]["generation_seed"],
        "LongBench v2 generation seed drifted from the frozen manifest.",
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])
    rows = load_rows(dataset_path, prompt_template)
    if adaptive_manifest is not None:
        _require(
            adaptive_manifest["benchmark"]["predictions_per_arm"] == len(rows)
            and adaptive_manifest["model"]["snapshot_digest_set_sha256"]
            == manifest["model"]["snapshot_digest_set_sha256"]
            and adaptive_manifest["benchmark"]["dataset_sha256"]
            == manifest["benchmarks"][BENCHMARK]["dataset"]["files"][0]["sha256"]
            and adaptive_manifest["benchmark"]["prompt_sha256"] == sha256(prompt_path),
            "Adaptive LongBench v2 example count drifted from the base suite.",
        )
    runner_digest = sha256(Path(__file__).resolve())
    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    causal_digest = sha256(args.causal_gate)
    selection_digest = sha256(args.fixed_selection)
    source_inventory_digest = sha256(args.source_inventory)
    adaptive_manifest_digest = (
        sha256(args.adaptive_quota_manifest) if adaptive_manifest is not None else None
    )
    scorer_digest = sha256(Path(__file__).with_name("p3_natural_metrics.py"))
    maximum_context = manifest["model"]["maximum_supported_context_tokens"]
    revisions = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "code_revision": CODE_REVISION,
        "scorer_sha256": scorer_digest,
        "prompt_sha256": sha256(prompt_path),
    }
    cohort_arms = ADAPTIVE_QUOTA_ARMS if args.cohort == "adaptive-quota" else ARMS
    selected_arms = tuple(args.arm or cohort_arms)
    _require(
        len(selected_arms) == len(set(selected_arms))
        and all(arm in cohort_arms for arm in selected_arms),
        f"Selected LongBench v2 arms do not belong to the {args.cohort} cohort.",
    )
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "inventory_sha256": inventory_digest,
            "causal_gate_sha256": causal_digest,
            "fixed_selection_sha256": selection_digest,
            "source_inventory_sha256": source_inventory_digest,
            "cohort": args.cohort,
            "adaptive_quota_manifest_sha256": adaptive_manifest_digest,
            "adaptive_prerequisite_sha256": {
                name: metadata["sha256"] for name, metadata in adaptive_prerequisites.items()
            },
            "sequence_gate_dependency_sha256": {
                name: metadata["sha256"]
                for name, metadata in decision.get("dependencies", {}).items()
            },
            "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
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
        if not _completed(args.output_root / arm / "cell.json", arm=arm, identity=identities[arm])
    ]
    if not pending:
        print(json.dumps({"status": "complete", "arms": list(selected_arms)}))
        return
    _require(torch.cuda.is_available(), "LongBench v2 evaluation requires CUDA.")
    gpu_lock = acquire_gpu_lock("p3-longbench-v2")
    try:
        from p3_protected_prefix_press import (
            wrap_same_budget_adaptive_quota_protected_prefix,
            wrap_same_budget_protected_prefix,
        )

        EvaluationConfig, EvaluationRunner, _scorer = load_evaluator(kvpress_root)
        base = EvaluationConfig(
            dataset="longbench-v2",
            model=str(model_snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=args.seed,
            max_context_length=maximum_context,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(base)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
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
            active_press: Any = runner.press
            compatibility_press: Any = None
            if args.cohort == "adaptive-quota":
                if arm == "fixed+pins":
                    compatibility_press = wrap_same_budget_protected_prefix(runner.press)
                else:
                    compatibility_press = wrap_same_budget_adaptive_quota_protected_prefix(
                        runner.press,
                        max_adjustment_fraction=config["max_adjustment_fraction"],
                    )
                active_press = compatibility_press
            root = args.output_root / arm
            progress_path = root / "progress.json"
            records_path = root / "records.jsonl"
            existing = _existing_progress(progress_path, records_path, identities[arm])
            _require(
                all(
                    existing[index]["example_id"] == rows[index]["_id"]
                    for index in range(len(existing))
                ),
                "Partial LongBench v2 example order drifted.",
            )
            limit = len(rows)
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            with records_path.open("a") as handle:
                for index in range(len(existing), limit):
                    row = rows[index]
                    context, question = prompt_parts(row, prompt_template)
                    rendered = rendered_input(tokenizer, context, question)
                    base_row = base_record(
                        row=row,
                        arm=arm,
                        rendered=rendered,
                        config=config,
                        revisions=revisions,
                    )
                    if args.cohort == "adaptive-quota":
                        base_row["quota_physical_audit"] = None
                    if rendered["exact_input_tokens"] + GENERATION_RESERVE > maximum_context:
                        record = failure_record(
                            base_row,
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
                            if compatibility_press is not None:
                                span = config["protected_prefix_token_span"]
                                compatibility_press.configure(
                                    protected_start=span["start"], protected_end=span["end"]
                                )
                            response, resident_bytes = infer_one(
                                pipeline=runner.pipeline,
                                press=active_press,
                                rendered=rendered,
                                max_new_tokens=GENERATION_RESERVE,
                            )
                            compatibility_audit = (
                                compatibility_press.audit()
                                if compatibility_press is not None
                                else None
                            )
                            if args.cohort == "adaptive-quota":
                                base_row["quota_physical_audit"] = compatibility_audit
                            torch.cuda.synchronize()
                            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                            peak = torch.cuda.max_memory_allocated()
                            if not response.strip():
                                record = failure_record(
                                    base_row,
                                    failure_type="empty-generation",
                                    latency_ms=latency_ms,
                                    peak_hbm_bytes=peak,
                                    hot_resident_bytes=resident_bytes,
                                )
                            else:
                                parsed = extract_longbench_v2_choice(response)
                                generated = len(
                                    tokenizer.encode(response, add_special_tokens=False)
                                )
                                record = {
                                    **base_row,
                                    "status": "scored",
                                    "raw_response": response,
                                    "parsed_response": parsed,
                                    "score": score_longbench_v2(response, row["answer"]),
                                    "failure_type": None,
                                    "stop_reason": (
                                        "max-new-tokens"
                                        if generated >= GENERATION_RESERVE
                                        else "eos-or-special-token"
                                    ),
                                    "generated_tokens_observed": generated,
                                    "latency_ms": latency_ms,
                                    "peak_hbm_bytes": peak,
                                    "hot_resident_bytes": resident_bytes,
                                }
                        except torch.cuda.OutOfMemoryError as error:
                            record = failure_record(
                                base_row,
                                failure_type="oom",
                                latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                error=error,
                            )
                            torch.cuda.empty_cache()
                        except Exception as error:
                            record = failure_record(
                                base_row,
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
                            {
                                "arm": arm,
                                "completed_examples": index + 1,
                                "total_examples": len(rows),
                            }
                        ),
                        flush=True,
                    )
            if limit < len(rows):
                continue
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
                "experiment_manifest": {
                    "path": str(args.manifest),
                    "sha256": manifest_digest,
                },
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
                "fixed_baseline_selection": {
                    "path": str(args.fixed_selection),
                    "sha256": selection_digest,
                },
                "evaluation_source_inventory": {
                    "path": str(args.source_inventory),
                    "sha256": source_inventory_digest,
                },
                "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
                "p3_sequence_decision": decision,
                "environment": environment,
                "raw_records": {"path": str(records_path), "sha256": sha256(records_path)},
            }
            atomic_json(root / "cell.json", cell)
            progress_path.unlink()
    finally:
        gpu_lock.close()


if __name__ == "__main__":
    main()
