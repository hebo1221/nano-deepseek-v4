from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_cross_family_sequence_gate import require_cross_family_sequence_gate
from p3_natural_workloads import build_longmem_full_history_prompt
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_longbench_v2 import rendered_input
from run_p3_mrcr import (
    arm_config,
    atomic_json,
    failure_record,
    infer_one,
    runtime_environment,
)
from run_p3_natural_ruler import compatibility_arm_config
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from validate_p3_natural_adaptive_quota_longmemeval_manifest import (
    validate_manifest as validate_adaptive_longmemeval_manifest,
)
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "LongMemEval"
ARMS = ("native-dense", "strongest-memory-matched-fixed")
ADAPTIVE_QUOTA_ARMS = ("fixed+pins", "natural-adaptive-quota+pins")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural/longmemeval"
)
ADAPTIVE_QUOTA_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
    "longmemeval-qwen3-4b"
)
EXPECTED_EXAMPLES = 500
GENERATION_RESERVE = 512
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
CODE_REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
JUDGE_MODEL = "gpt-4o-2024-08-06"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_adaptive_prerequisite(
    path: Path, *, experiment_id: str, predictions: int, label: str
) -> dict[str, str]:
    _require(path.is_file(), f"Adaptive LongMemEval prerequisite is unavailable: {label}.")
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
    elif experiment_id == "p3-natural-longmemeval-audit-v1":
        arms = payload.get("arms", {})
        expected_arms = {"native-dense", "strongest-memory-matched-fixed"}
        _require(
            audit.get("all_raw_artifacts_verified") is True
            and audit.get("all_failure_accounting_complete") is True
            and audit.get("all_required_arms_input_paired") is True
            and audit.get("all_reported_scores_recomputed_from_raw_response") is True
            and isinstance(arms, dict)
            and set(arms) == expected_arms
            and predictions % len(expected_arms) == 0
            and all(
                isinstance(arms[arm], dict)
                and arms[arm].get("accounted_examples") == predictions // len(expected_arms)
                for arm in expected_arms
            ),
            "Baseline LongMemEval prerequisite is not a complete audited generation result.",
        )
    else:
        raise ValueError(f"Unsupported adaptive LongMemEval prerequisite: {experiment_id}.")
    return {"path": str(path), "sha256": sha256(path)}


def prompt_parts(row: dict[str, Any]) -> tuple[str, str]:
    prompt = build_longmem_full_history_prompt(row)
    question_date, question = row.get("question_date"), row.get("question")
    _require(
        isinstance(question_date, str) and isinstance(question, str),
        "LongMemEval question fields must be text.",
    )
    suffix = f"\n\nCurrent Date: {question_date}\nQuestion: {question}\nAnswer:"
    _require(prompt.endswith(suffix), "LongMemEval prompt suffix drifted.")
    context = prompt[: -len(suffix)]
    _require(bool(context), "LongMemEval history context is empty.")
    return context, suffix


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    _require(
        isinstance(payload, list) and len(payload) == EXPECTED_EXAMPLES,
        "LongMemEval must contain exactly 500 rows.",
    )
    identifiers: set[str] = set()
    required = {
        "question_id",
        "question_type",
        "question",
        "question_date",
        "answer",
        "haystack_dates",
        "haystack_sessions",
    }
    for row in payload:
        _require(isinstance(row, dict) and required.issubset(row), "Invalid LongMemEval row.")
        identifier = row["question_id"]
        _require(
            isinstance(identifier, str) and bool(identifier) and identifier not in identifiers,
            "LongMemEval question id is missing or duplicated.",
        )
        identifiers.add(identifier)
        _require(
            isinstance(row["question_type"], str)
            and bool(row["question_type"])
            and isinstance(row["answer"], str),
            "LongMemEval question type and answer must be text.",
        )
        prompt_parts(row)
    return payload


def _load_official_judge_module(source_root: Path, expected_sha256: str) -> Any:
    path = source_root / "src" / "evaluation" / "evaluate_qa.py"
    _require(path.is_file() and sha256(path) == expected_sha256, "Judge source drifted.")
    spec = importlib.util.spec_from_file_location("adaptive_v4_longmem_judge", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load official judge source.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def judge_response(
    *, module: Any, client: Any, row: dict[str, Any], response: str
) -> tuple[float, dict[str, Any]]:
    prompt = module.get_anscheck_prompt(
        row["question_type"],
        row["question"],
        row["answer"],
        response,
        abstention="_abs" in row["question_id"],
    )
    started = time.perf_counter_ns()
    completion = module.chat_completions_with_backoff(
        client,
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    raw = completion.choices[0].message.content.strip()
    return float("yes" in raw.lower()), {
        "model": JUDGE_MODEL,
        "returned_model": getattr(completion, "model", None),
        "response_id": getattr(completion, "id", None),
        "created": getattr(completion, "created", None),
        "status": "scored",
        "prompt": prompt,
        "raw_response": raw,
        "latency_ms": latency_ms,
    }


def judge_blocked_record(
    base: dict[str, Any],
    *,
    response: str,
    latency_ms: float,
    peak_hbm_bytes: int,
    hot_resident_bytes: int,
    reason: str,
    generated_tokens: int,
    generation_stop_reason: str,
) -> dict[str, Any]:
    return {
        **base,
        "status": "failure",
        "raw_response": response,
        "parsed_response": response,
        "score": None,
        "failure_type": "judge-blocked",
        "stop_reason": generation_stop_reason,
        "generated_tokens_observed": generated_tokens,
        "latency_ms": latency_ms,
        "peak_hbm_bytes": peak_hbm_bytes,
        "hot_resident_bytes": hot_resident_bytes,
        "judge": {
            "model": JUDGE_MODEL,
            "status": "blocked",
            "reason": reason,
            "latency_ms": 0.0,
        },
    }


def load_dependencies(
    manifest_path: Path,
    inventory_path: Path,
    source_inventory_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    manifest = json.loads(manifest_path.read_text())
    _require(manifest["model"]["revision"] == MODEL_REVISION, "Model revision drifted.")
    contract = manifest["benchmarks"][BENCHMARK]
    _require(
        contract["dataset"]["revision"] == DATASET_REVISION
        and contract["upstream_code"]["revision"] == CODE_REVISION,
        "LongMemEval revision drifted.",
    )
    inventory = json.loads(inventory_path.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-dataset-inventory-v1"
        and inventory.get("source", {}).get("dirty") is False,
        "Natural dataset inventory is not a clean frozen artifact.",
    )
    observed = inventory["benchmarks"][BENCHMARK]
    _require(observed["revision"] == DATASET_REVISION, "Dataset inventory revision drifted.")
    _require(len(observed["files"]) == 1, "LongMemEval inventory must contain one file.")
    data_path = Path(observed["files"][0]["path"])
    expected_data_digest = contract["dataset"]["files"][0]["sha256"]
    _require(
        data_path.is_file()
        and observed["files"][0]["sha256"] == expected_data_digest
        and sha256(data_path) == expected_data_digest,
        "LongMemEval dataset artifact drifted.",
    )

    source_inventory = json.loads(source_inventory_path.read_text())
    _require(
        source_inventory.get("experiment_id") == "p3-natural-source-inventory-v1"
        and source_inventory.get("status") == "verified"
        and source_inventory.get("source", {}).get("dirty") is False,
        "Natural source inventory is not a clean verified artifact.",
    )
    source_entry = source_inventory["benchmarks"][BENCHMARK]
    _require(
        source_entry["revision"] == CODE_REVISION
        and source_entry.get("clean_tracked_tree") is True,
        "LongMemEval source revision drifted.",
    )
    source_root = Path(source_entry["path"])
    expected_source_files = contract["upstream_code"]["files_sha256"]
    observed_source_files = {row["path"]: row["sha256"] for row in source_entry["files"]}
    _require(observed_source_files == expected_source_files, "Source file inventory drifted.")
    for relative, digest in expected_source_files.items():
        _require(
            (source_root / relative).is_file() and sha256(source_root / relative) == digest,
            f"LongMemEval source drifted: {relative}.",
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
    return manifest, selection, data_path, source_root


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
    _require(progress.is_file() and partial.is_file(), "Partial LongMemEval state is incomplete.")
    _require(
        json.loads(progress.read_text()) == identity, "Partial LongMemEval provenance drifted."
    )
    records = [json.loads(line) for line in partial.read_text().splitlines() if line]
    _require(len(records) <= EXPECTED_EXAMPLES, "Partial LongMemEval has too many records.")
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
        f"Completed LongMemEval provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"]["sha256"] == sha256(records),
        f"Completed LongMemEval records drifted: {records}.",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B LongMemEval arms.")
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
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--primary-core-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-core-quality-matrix.strict.summary.json"
        ),
    )
    parser.add_argument(
        "--nine-seed-causal-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"
        ),
    )
    parser.add_argument(
        "--adaptive-quota-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longmemeval-v1.json"
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
        "--baseline-longmemeval-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural/longmemeval.summary.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--judge-mode", choices=("blocked", "openai"), default="blocked")
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
        _require(
            args.judge_mode == "blocked",
            "Adaptive LongMemEval is frozen as judge-blocked generation evidence only.",
        )
        sequence_decision = require_cross_family_sequence_gate(
            primary_core=args.primary_core_summary,
            primary_causal=args.causal_gate,
            nine_seed_causal=args.nine_seed_causal_summary,
            fixed_selection=args.fixed_selection,
        )
        adaptive_manifest = json.loads(args.adaptive_quota_manifest.read_text())
        validate_adaptive_longmemeval_manifest(adaptive_manifest)
        adaptive_prerequisites = {
            "adaptive_ruler": load_adaptive_prerequisite(
                args.adaptive_ruler_summary,
                experiment_id="p3-natural-adaptive-quota-ruler-audit-v1",
                predictions=65_000,
                label="Qwen3-4B adaptive-quota RULER audit",
            ),
            "baseline_longmemeval": load_adaptive_prerequisite(
                args.baseline_longmemeval_summary,
                experiment_id="p3-natural-longmemeval-audit-v1",
                predictions=1_000,
                label="Qwen3-4B baseline LongMemEval audit",
            ),
        }
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = ADAPTIVE_QUOTA_OUTPUT_ROOT
    else:
        sequence_decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    if git_dirty(Path.cwd()):
        raise RuntimeError("LongMemEval requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    manifest, selection, data_path, source_root = load_dependencies(
        args.manifest, args.dataset_inventory, args.source_inventory, args.fixed_selection
    )
    _require(
        args.seed == manifest["benchmarks"][BENCHMARK]["generation_seed"],
        "LongMemEval generation seed drifted from the frozen manifest.",
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])
    rows = load_rows(data_path)
    if adaptive_manifest is not None:
        benchmark_contract = adaptive_manifest["benchmark"]
        base_contract = manifest["benchmarks"][BENCHMARK]
        _require(
            benchmark_contract["examples_per_arm"] == len(rows)
            and benchmark_contract["generation_seed"] == args.seed
            and benchmark_contract["generation_reserve_tokens"] == GENERATION_RESERVE
            and adaptive_manifest["model"]["snapshot_digest_set_sha256"]
            == manifest["model"]["snapshot_digest_set_sha256"]
            and benchmark_contract["dataset_sha256"]
            == base_contract["dataset"]["files"][0]["sha256"]
            and benchmark_contract["official_judge_source_sha256"]
            == base_contract["upstream_code"]["files_sha256"][
                "src/evaluation/evaluate_qa.py"
            ],
            "Adaptive LongMemEval immutable inputs drifted from the base suite.",
        )

    judge_module = None
    judge_client = None
    if args.judge_mode == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        _require(bool(api_key), "OPENAI_API_KEY is required for explicit OpenAI judge mode.")
        judge_source_digest = manifest["benchmarks"][BENCHMARK]["upstream_code"]["files_sha256"][
            "src/evaluation/evaluate_qa.py"
        ]
        judge_module = _load_official_judge_module(source_root, judge_source_digest)
        judge_client = judge_module.OpenAI(api_key=api_key)

    manifest_digest = sha256(args.manifest)
    inventory_digest = sha256(args.dataset_inventory)
    source_inventory_digest = sha256(args.source_inventory)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    runner_digest = sha256(Path(__file__).resolve())
    adaptive_manifest_digest = (
        sha256(args.adaptive_quota_manifest) if adaptive_manifest is not None else None
    )
    scorer_digest = manifest["benchmarks"][BENCHMARK]["upstream_code"]["files_sha256"][
        "src/evaluation/evaluate_qa.py"
    ]
    maximum_context = manifest["model"]["maximum_supported_context_tokens"]
    cohort_arms = ADAPTIVE_QUOTA_ARMS if args.cohort == "adaptive-quota" else ARMS
    selected_arms = tuple(args.arm or cohort_arms)
    _require(
        len(selected_arms) == len(set(selected_arms))
        and all(arm in cohort_arms for arm in selected_arms),
        f"Selected LongMemEval arms do not belong to the {args.cohort} cohort.",
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
            "arm_config": (
                compatibility_arm_config(arm, selection, selection_digest)
                if args.cohort == "adaptive-quota"
                else arm_config(arm, selection, selection_digest)
            ),
            "judge_mode": args.judge_mode,
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
    _require(torch.cuda.is_available(), "LongMemEval requires CUDA.")

    lock = acquire_gpu_lock("p3-longmemeval")
    try:
        from p3_protected_prefix_press import (
            wrap_same_budget_adaptive_quota_protected_prefix,
            wrap_same_budget_protected_prefix,
        )

        EvaluationConfig, EvaluationRunner, _scorer = load_evaluator(kvpress_root)
        config = EvaluationConfig(
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
            partial = root / "records.partial.jsonl"
            existing = _existing_records(progress, partial, identities[arm])
            _require(
                all(
                    existing[index]["example_id"] == rows[index]["question_id"]
                    for index in range(len(existing))
                ),
                "Partial LongMemEval example order drifted.",
            )
            limit = len(rows)
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            with partial.open("a") as handle:
                for index in range(len(existing), limit):
                    row = rows[index]
                    context, question = prompt_parts(row)
                    rendered = rendered_input(tokenizer, context, question)
                    base = {
                        "example_id": row["question_id"],
                        "benchmark": BENCHMARK,
                        "arm": arm,
                        "exact_input_tokens": rendered["exact_input_tokens"],
                        "generation_reserve_tokens": GENERATION_RESERVE,
                        "raw_prompt_sha256": rendered["raw_prompt_sha256"],
                        "input_token_ids_sha256": rendered["input_token_ids_sha256"],
                        "token_boundary_retreat": rendered["token_boundary_retreat"],
                        "arm_config": settings,
                        "revisions": {
                            "model_revision": MODEL_REVISION,
                            "dataset_revision": DATASET_REVISION,
                            "code_revision": CODE_REVISION,
                            "scorer_sha256": scorer_digest,
                        },
                        "question_type": row["question_type"],
                        "abstention": "_abs" in row["question_id"],
                    }
                    if args.cohort == "adaptive-quota":
                        base["quota_physical_audit"] = None
                    if rendered["exact_input_tokens"] + GENERATION_RESERVE > maximum_context:
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
                        response: str | None = None
                        resident_bytes = 0
                        try:
                            if compatibility_press is not None:
                                span = settings["protected_prefix_token_span"]
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
                            elif args.judge_mode == "blocked":
                                generated = len(
                                    tokenizer.encode(response, add_special_tokens=False)
                                )
                                generation_stop = (
                                    "max-new-tokens"
                                    if generated >= GENERATION_RESERVE
                                    else "eos-or-special-token"
                                )
                                record = judge_blocked_record(
                                    base,
                                    response=response,
                                    latency_ms=latency_ms,
                                    peak_hbm_bytes=peak_hbm,
                                    hot_resident_bytes=resident_bytes,
                                    reason="explicit-no-paid-judge-mode",
                                    generated_tokens=generated,
                                    generation_stop_reason=generation_stop,
                                )
                            else:
                                score, judge = judge_response(
                                    module=judge_module,
                                    client=judge_client,
                                    row=row,
                                    response=response,
                                )
                                generated = len(
                                    tokenizer.encode(response, add_special_tokens=False)
                                )
                                record = {
                                    **base,
                                    "status": "scored",
                                    "raw_response": response,
                                    "parsed_response": response,
                                    "score": score,
                                    "failure_type": None,
                                    "stop_reason": (
                                        "max-new-tokens"
                                        if generated >= GENERATION_RESERVE
                                        else "eos-or-special-token"
                                    ),
                                    "generated_tokens_observed": generated,
                                    "latency_ms": latency_ms,
                                    "peak_hbm_bytes": peak_hbm,
                                    "hot_resident_bytes": resident_bytes,
                                    "judge": judge,
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
                            failure_type = (
                                "judge-blocked"
                                if args.judge_mode == "openai" and response is not None
                                else "runtime-error"
                            )
                            if failure_type == "judge-blocked":
                                generated = len(
                                    tokenizer.encode(response or "", add_special_tokens=False)
                                )
                                generation_stop = (
                                    "max-new-tokens"
                                    if generated >= GENERATION_RESERVE
                                    else "eos-or-special-token"
                                )
                                record = judge_blocked_record(
                                    base,
                                    response=response or "",
                                    latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                                    peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                                    hot_resident_bytes=resident_bytes,
                                    reason=f"{type(error).__name__}: {error}",
                                    generated_tokens=generated,
                                    generation_stop_reason=generation_stop,
                                )
                            else:
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
    finally:
        lock.close()


if __name__ == "__main__":
    main()
