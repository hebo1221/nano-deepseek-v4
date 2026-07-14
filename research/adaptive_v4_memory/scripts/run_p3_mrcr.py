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
from p3_cross_family_sequence_gate import require_cross_family_sequence_gate
from p3_natural_metrics import score_mrcr
from p3_natural_workloads import (
    encode_rendered_segments_exact,
    mrcr_generation_reserve,
    render_chat_split_last_user,
    select_mrcr_primary_rows,
)
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_ruler_matrix import (
    KVPRESS_REVISION,
    git_dirty,
    git_head,
    kvpress_runtime_binding,
    load_evaluator,
)
from transformers import DynamicCache
from validate_p3_natural_adaptive_quota_mrcr_manifest import (
    validate_manifest as validate_adaptive_mrcr_manifest,
)
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "MRCR"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
ADAPTIVE_QUOTA_ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
DEFAULT_OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/mrcr")
ADAPTIVE_QUOTA_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/mrcr-qwen3-4b"
)
NEEDLE_COUNTS = (2, 4, 8)
EXPECTED_EXAMPLES = 1500
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_adaptive_prerequisite(
    path: Path, *, experiment_id: str, predictions: int, label: str
) -> dict[str, str]:
    _require(path.is_file(), f"Adaptive MRCR prerequisite is unavailable: {label}.")
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
    elif experiment_id == "p3-natural-mrcr-audit-v1":
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
            "Baseline MRCR prerequisite is not a complete audited result.",
        )
    else:
        raise ValueError(f"Unsupported adaptive MRCR prerequisite: {experiment_id}.")
    return {"path": str(path), "sha256": sha256(path)}


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
    token_ids = torch.cat((context_ids, query_ids), dim=1)
    return {
        "context_ids": context_ids,
        "question_ids": query_ids,
        "exact_input_tokens": int(context_ids.shape[1] + query_ids.shape[1]),
        "raw_prompt_sha256": hashlib.sha256((context + query).encode()).hexdigest(),
        "input_token_ids_sha256": hashlib.sha256(
            token_ids.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
        ).hexdigest(),
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
        "kvpress_binding": kvpress_runtime_binding(),
        "tiktoken": importlib.metadata.version("tiktoken"),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B MRCR primary cells.")
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
            "p3-natural-adaptive-quota-mrcr-v1.json"
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
        "--baseline-mrcr-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/mrcr.summary.json"),
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
    adaptive_arm_config: Any = None
    if args.cohort == "adaptive-quota":
        from run_p3_natural_ruler import compatibility_arm_config

        adaptive_arm_config = compatibility_arm_config
        sequence_decision = require_cross_family_sequence_gate(
            primary_core=args.primary_core_summary,
            primary_causal=args.causal_gate,
            nine_seed_causal=args.nine_seed_causal_summary,
            fixed_selection=args.fixed_selection,
        )
        adaptive_manifest = json.loads(args.adaptive_quota_manifest.read_text())
        validate_adaptive_mrcr_manifest(adaptive_manifest)
        adaptive_prerequisites = {
            "adaptive_ruler": load_adaptive_prerequisite(
                args.adaptive_ruler_summary,
                experiment_id="p3-natural-adaptive-quota-ruler-audit-v1",
                predictions=65_000,
                label="Qwen3-4B adaptive-quota RULER audit",
            ),
            "baseline_mrcr": load_adaptive_prerequisite(
                args.baseline_mrcr_summary,
                experiment_id="p3-natural-mrcr-audit-v1",
                predictions=3_000,
                label="Qwen3-4B baseline MRCR audit",
            ),
        }
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = ADAPTIVE_QUOTA_OUTPUT_ROOT
    else:
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
    _require(
        args.seed == manifest["benchmarks"][BENCHMARK]["generation_seed"],
        "MRCR generation seed drifted from the frozen manifest.",
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
    if adaptive_manifest is not None:
        _require(
            adaptive_manifest["benchmark"]["predictions_per_arm"] == len(rows)
            and adaptive_manifest["model"]["snapshot_digest_set_sha256"]
            == manifest["model"]["snapshot_digest_set_sha256"]
            and adaptive_manifest["benchmark"]["dataset_file_sha256"]
            == [row["sha256"] for row in manifest["benchmarks"][BENCHMARK]["dataset"]["files"]]
            and adaptive_manifest["benchmark"]["scorer_sha256"]
            == sha256(Path(__file__).with_name("p3_natural_metrics.py")),
            "Adaptive MRCR immutable inputs drifted from the base suite.",
        )
    runner_digest = sha256(Path(__file__))
    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    scorer_digest = sha256(Path(__file__).with_name("p3_natural_metrics.py"))
    adaptive_manifest_digest = (
        sha256(args.adaptive_quota_manifest) if adaptive_manifest is not None else None
    )
    maximum_context = manifest["model"]["maximum_supported_context_tokens"]
    cohort_arms = ADAPTIVE_QUOTA_ARMS if args.cohort == "adaptive-quota" else ARMS
    selected_arms = tuple(args.arm or cohort_arms)
    _require(
        len(selected_arms) == len(set(selected_arms))
        and all(arm in cohort_arms for arm in selected_arms),
        f"Selected MRCR arms do not belong to the {args.cohort} cohort.",
    )
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "inventory_sha256": inventory_digest,
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
            "arm_config": (
                adaptive_arm_config(arm, selection, selection_digest)
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
    _require(torch.cuda.is_available(), "MRCR evaluation requires CUDA.")

    lock = acquire_gpu_lock("p3-mrcr")
    try:
        from p3_protected_prefix_press import (
            wrap_same_budget_adaptive_quota_protected_prefix,
            wrap_same_budget_protected_prefix,
        )

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
            active_press: Any = runner.press
            compatibility_press: Any = None
            if args.cohort == "adaptive-quota":
                if arm == "fixed+pins":
                    compatibility_press = wrap_same_budget_protected_prefix(runner.press)
                else:
                    compatibility_press = wrap_same_budget_adaptive_quota_protected_prefix(
                        runner.press,
                        max_adjustment_fraction=arm_settings["max_adjustment_fraction"],
                    )
                active_press = compatibility_press
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
                        "input_token_ids_sha256": rendered["input_token_ids_sha256"],
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
                    if args.cohort == "adaptive-quota":
                        base["quota_physical_audit"] = None
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
                            if compatibility_press is not None:
                                span = arm_settings["protected_prefix_token_span"]
                                compatibility_press.configure(
                                    protected_start=span["start"], protected_end=span["end"]
                                )
                            response, resident_bytes = infer_one(
                                pipeline=runner.pipeline,
                                press=active_press,
                                rendered=rendered,
                                max_new_tokens=reserve,
                            )
                            compatibility_audit = (
                                compatibility_press.audit()
                                if compatibility_press is not None
                                else None
                            )
                            if args.cohort == "adaptive-quota":
                                base["quota_physical_audit"] = compatibility_audit
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
