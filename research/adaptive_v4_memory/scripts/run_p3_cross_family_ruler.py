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
from p3_cross_family_sequence_gate import require_cross_family_sequence_gate
from prepare_p3_cross_family_ruler_dataset import (
    DATASET_EXPERIMENT_ID,
    EXPECTED_ROWS_PER_LENGTH,
    LENGTHS,
    MODEL_REVISION,
    RULER_REVISION,
    SAMPLES_PER_TASK,
    TASKS,
)
from run_p3_mrcr import atomic_json, failure_record, infer_one, runtime_environment
from run_p3_natural_ruler import rendered_input, select_score_compatible_baseline
from run_p3_ruler_matrix import (
    KVPRESS_REVISION,
    example_score,
    git_dirty,
    git_head,
    load_dataset,
    load_evaluator,
)
from validate_p3_cross_family_adaptive_quota_manifest import (
    validate_manifest as validate_adaptive_manifest,
)
from validate_p3_cross_family_ruler_manifest import validate_manifest
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "RULER"
ARMS = ("native-dense", "qwen-selected-memory-matched")
ADAPTIVE_QUOTA_ARMS = ("fixed+pins", "cross-family-adaptive-quota+pins")
EXPECTED_EXAMPLES = len(LENGTHS) * EXPECTED_ROWS_PER_LENGTH
CELL_EXPERIMENT_ID = "p3-cross-family-ruler-arm-cell-v1"
OFFICIAL_SCORER_PATH = Path("evaluation/benchmarks/ruler/calculate_metrics.py")
OFFICIAL_SCORER_SHA256 = "1df51402a394b1348f14d96e1fe87b1a4aff10f619f81f80f8840d8e0118fc9b"
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler/results"
)
ADAPTIVE_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/"
    "phi4-mini-adaptive-quota-ruler/results"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_example_ids() -> list[str]:
    return [
        f"{length}:{task}:{row_number}"
        for length in LENGTHS
        for task in TASKS
        for row_number in range(SAMPLES_PER_TASK)
    ]


def arm_config(arm: str, selection: dict[str, Any], selection_digest: str) -> dict[str, Any]:
    if arm == "native-dense":
        return {"press_name": "no_press", "compression_ratio": 0.0}
    _require(arm == "qwen-selected-memory-matched", f"Unknown cross-family arm: {arm}.")
    return {
        "press_name": selection["selected_arm"],
        "compression_ratio": selection["selected_compression_ratio"],
        "selection_sha256": selection_digest,
        "selection_model_family": "Qwen3",
        "phi_specific_reselection": False,
    }


def adaptive_quota_arm_config(
    arm: str, selection: dict[str, Any], selection_digest: str
) -> dict[str, Any]:
    _require(arm in ADAPTIVE_QUOTA_ARMS, f"Unknown Phi adaptive arm: {arm}.")
    selected = select_score_compatible_baseline(selection)
    return {
        "press_name": selected["arm"],
        "compression_ratio": 0.5,
        "protected_prefix_token_span": {"start": 0, "end": 4},
        "quota_policy": "fixed-per-layer" if arm == "fixed+pins" else "causal-adaptive",
        "max_adjustment_fraction": 0.0 if arm == "fixed+pins" else 0.25,
        "selection_sha256": selection_digest,
        "score_compatible_selection_rule": (
            "highest frozen Qwen3-1.7B row-weighted mean; lexicographic tie-break"
        ),
        "selection_model_family": "Qwen3",
        "phi_specific_reselection": False,
    }


def require_qwen_adaptive_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    _require(
        payload.get("experiment_id") == "p3-natural-adaptive-quota-ruler-audit-v1"
        and payload.get("status") == "terminal"
        and audit.get("total_predictions") == 65_000
        and audit.get("paired_examples") == 32_500
        and audit.get("all_raw_records_verified") is True
        and audit.get("quota_physical_audits_verified") is True
        and audit.get("outcome_dependent_execution") is False,
        "Phi adaptive transfer requires the terminal Qwen adaptive audit.",
    )
    return payload


def load_dataset_contracts(
    root: Path, *, manifest_digest: str, generator_digest: str
) -> tuple[dict[int, dict[str, Any]], str]:
    manifests: dict[int, dict[str, Any]] = {}
    digests: list[str] = []
    for length in LENGTHS:
        path = root / str(length) / "dataset-manifest.json"
        _require(path.is_file(), f"Missing cross-family RULER dataset manifest: {path}.")
        payload = json.loads(path.read_text())
        _require(
            payload.get("experiment_id") == DATASET_EXPERIMENT_ID,
            f"Wrong cross-family RULER dataset id at {length}.",
        )
        _require(
            payload.get("source", {}).get("dirty") is False
            and payload.get("source", {}).get("implementation_sha256") == generator_digest,
            f"Cross-family RULER generator provenance drifted at {length}.",
        )
        _require(
            payload.get("cross_family_manifest", {}).get("sha256") == manifest_digest,
            f"Cross-family RULER manifest drifted at {length}.",
        )
        tokenizer = payload.get("tokenizer", {})
        _require(
            payload.get("ruler", {}).get("revision") == RULER_REVISION
            and payload.get("ruler", {}).get("clean_tracked_tree") is True
            and tokenizer.get("model_revision") == MODEL_REVISION
            and tokenizer.get("trust_remote_code") is False,
            f"Cross-family RULER upstream revision drifted at {length}.",
        )
        generation = payload.get("generation", {})
        _require(
            generation.get("length_tokens") == length
            and generation.get("samples_per_task") == SAMPLES_PER_TASK
            and generation.get("random_seed") == 42
            and generation.get("tasks") == list(TASKS)
            and generation.get("total_rows") == EXPECTED_ROWS_PER_LENGTH
            and generation.get("silently_truncated_examples") == 0,
            f"Cross-family RULER generation contract drifted at {length}.",
        )
        task_artifacts = payload.get("task_artifacts", {})
        _require(set(task_artifacts) == set(TASKS), f"RULER task coverage drifted at {length}.")
        for task, metadata in task_artifacts.items():
            artifact = Path(metadata.get("path", ""))
            _require(
                artifact.is_file()
                and metadata.get("rows") == SAMPLES_PER_TASK
                and metadata.get("sha256") == sha256(artifact),
                f"Cross-family RULER task artifact drifted: {length}/{task}.",
            )
        manifests[length] = payload
        digests.append(f"{length}:{sha256(path)}")
    return manifests, hashlib.sha256("\n".join(digests).encode()).hexdigest()


def load_dependencies(
    manifest_path: Path, selection_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    _require(manifest["model"]["revision"] == MODEL_REVISION, "Phi model revision drifted.")
    benchmark = manifest["benchmark"]
    _require(
        tuple(benchmark["lengths_tokens"]) == LENGTHS
        and tuple(benchmark["tasks"]) == TASKS
        and benchmark["samples_per_task_length"] == SAMPLES_PER_TASK
        and benchmark["predictions_per_arm"] == EXPECTED_EXAMPLES,
        "Cross-family RULER manifest coverage drifted.",
    )
    selection = json.loads(selection_path.read_text())
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False
        and selection.get("selected_compression_ratio") == 0.5,
        "Cross-family RULER fixed selection is not terminal at 50% compression.",
    )
    return manifest, selection


def _existing_records(
    progress: Path, partial: Path, identity: dict[str, Any]
) -> list[dict[str, Any]]:
    if not progress.exists() and not partial.exists():
        atomic_json(progress, identity)
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.touch()
        return []
    if progress.is_file() and not partial.exists():
        partial.touch()
    if partial.is_file() and partial.stat().st_size == 0 and not progress.exists():
        atomic_json(progress, identity)
    _require(progress.is_file() and partial.is_file(), "Partial Phi RULER state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial Phi provenance drifted.")
    records = [json.loads(line) for line in partial.read_text().splitlines() if line]
    _require(len(records) <= EXPECTED_EXAMPLES, "Partial Phi arm has too many records.")
    return records


def _completed(cell_path: Path, arm: str, identity: dict[str, Any]) -> bool:
    if not cell_path.is_file():
        return False
    cell = json.loads(cell_path.read_text())
    _require(
        cell.get("experiment_id") == CELL_EXPERIMENT_ID
        and cell.get("benchmark") == BENCHMARK
        and cell.get("arm") == arm
        and cell.get("status") == "terminal"
        and cell.get("run_identity") == identity,
        f"Completed Phi RULER provenance drifted: {cell_path}.",
    )
    records = Path(cell.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and cell["raw_records"].get("sha256") == sha256(records),
        f"Completed Phi RULER records drifted: {records}.",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Phi-4-mini RULER transfer.")
    parser.add_argument("--cohort", choices=("baseline", "adaptive-quota"), default="baseline")
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/cross-family/phi4-mini-ruler/data"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
        ),
    )
    parser.add_argument(
        "--primary-core",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.strict.summary.json"
        ),
    )
    parser.add_argument(
        "--primary-causal",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"),
    )
    parser.add_argument(
        "--nine-seed-causal",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"),
    )
    parser.add_argument(
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--adaptive-quota-manifest",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/manifests/p3-cross-family-adaptive-quota-ruler-v1.json"
        ),
    )
    parser.add_argument(
        "--qwen-adaptive-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-adaptive-quota/"
            "ruler-qwen3-4b.summary.json"
        ),
    )
    parser.add_argument("--arm", action="append", choices=(*ARMS, *ADAPTIVE_QUOTA_ARMS))
    parser.add_argument("--max-new-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _require(
        args.max_new_examples is None or args.max_new_examples > 0,
        "max-new-examples must be positive.",
    )
    source_commit = git_head(Path.cwd())
    _require(not git_dirty(Path.cwd()), "Phi RULER evaluation requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    official_scorer = kvpress_root / OFFICIAL_SCORER_PATH
    _require(
        official_scorer.is_file() and sha256(official_scorer) == OFFICIAL_SCORER_SHA256,
        "Pinned official RULER scorer drifted.",
    )
    manifest, selection = load_dependencies(args.manifest, args.fixed_selection)
    adaptive_manifest: dict[str, Any] | None = None
    qwen_adaptive_audit: dict[str, Any] | None = None
    if args.cohort == "adaptive-quota":
        adaptive_manifest = json.loads(args.adaptive_quota_manifest.read_text())
        validate_adaptive_manifest(adaptive_manifest)
        qwen_adaptive_audit = require_qwen_adaptive_audit(args.qwen_adaptive_audit)
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            args.output_root = ADAPTIVE_OUTPUT_ROOT
    _require(args.seed == manifest["benchmark"]["generation_seed"], "Phi RULER seed drifted.")
    sequence_gate = require_cross_family_sequence_gate(
        primary_core=args.primary_core,
        primary_causal=args.primary_causal,
        nine_seed_causal=args.nine_seed_causal,
        fixed_selection=args.fixed_selection,
    )
    model_snapshot = args.model_snapshot.resolve()
    verify_snapshot(model_snapshot, manifest["model"])
    manifest_digest = sha256(args.manifest)
    selection_digest = sha256(args.fixed_selection)
    generator_digest = sha256(Path(__file__).with_name("prepare_p3_cross_family_ruler_dataset.py"))
    dataset_manifests, dataset_digest_set = load_dataset_contracts(
        args.dataset_root,
        manifest_digest=manifest_digest,
        generator_digest=generator_digest,
    )
    runner_digest = sha256(Path(__file__))
    cohort_arms = ADAPTIVE_QUOTA_ARMS if args.cohort == "adaptive-quota" else ARMS
    selected_arms = tuple(args.arm or cohort_arms)
    _require(
        len(selected_arms) == len(set(selected_arms))
        and all(arm in cohort_arms for arm in selected_arms),
        "Cross-family arms must be unique and belong to the selected cohort.",
    )
    adaptive_manifest_digest = (
        sha256(args.adaptive_quota_manifest) if adaptive_manifest is not None else None
    )
    qwen_adaptive_audit_digest = (
        sha256(args.qwen_adaptive_audit) if qwen_adaptive_audit is not None else None
    )
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "dataset_manifest_digest_set_sha256": dataset_digest_set,
            "sequence_gate_dependencies": sequence_gate["dependencies"],
            "fixed_selection_sha256": selection_digest,
            "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
            "kvpress_revision": KVPRESS_REVISION,
            "official_scorer_sha256": OFFICIAL_SCORER_SHA256,
            "cohort": args.cohort,
            "adaptive_quota_manifest_sha256": adaptive_manifest_digest,
            "qwen_adaptive_audit_sha256": qwen_adaptive_audit_digest,
            "arm_config": (
                adaptive_quota_arm_config(arm, selection, selection_digest)
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
    _require(torch.cuda.is_available(), "Phi RULER evaluation requires CUDA.")

    lock = acquire_gpu_lock("p3-cross-family-ruler")
    try:
        from p3_protected_prefix_press import (
            wrap_same_budget_adaptive_quota_protected_prefix,
            wrap_same_budget_protected_prefix,
        )

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
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "sdpa",
                "trust_remote_code": False,
            },
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
        _require(len(expected_ids) == EXPECTED_EXAMPLES, "Phi identity grid drifted.")

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
                    existing[index]["example_id"] == expected_ids[index]
                    for index in range(len(existing))
                ),
                "Partial Phi RULER example order drifted.",
            )
            limit = EXPECTED_EXAMPLES
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            global_index = 0
            with partial.open("a") as handle:
                for length in LENGTHS:
                    frame = load_dataset(dataset_manifests[length])
                    _require(len(frame) == EXPECTED_ROWS_PER_LENGTH, "Phi RULER row drifted.")
                    for row in frame.to_dict(orient="records"):
                        if global_index < len(existing):
                            global_index += 1
                            continue
                        if global_index >= limit:
                            break
                        example_id = f"{length}:{row['row_id']}"
                        _require(
                            example_id == expected_ids[global_index],
                            "Phi RULER generated order drifted.",
                        )
                        rendered = rendered_input(
                            runner.pipeline,
                            context=row["context"],
                            question=row["question"],
                            answer_prefix=row["answer_prefix"],
                            maximum_context=maximum_context,
                        )
                        reserve = int(row["max_new_tokens"])
                        _require(
                            rendered["exact_input_tokens"] + reserve <= length,
                            f"Rendered Phi RULER prompt exceeds the {length}-token contract: "
                            f"{example_id}.",
                        )
                        base = {
                            "example_id": example_id,
                            "benchmark": BENCHMARK,
                            "arm": arm,
                            "length_tokens": length,
                            "task": row["task"],
                            "exact_input_tokens": rendered["exact_input_tokens"],
                            "generation_reserve_tokens": reserve,
                            "raw_prompt_sha256": rendered["raw_prompt_sha256"],
                            "input_token_ids_sha256": rendered["input_token_ids_sha256"],
                            "token_boundary_retreat": rendered["token_boundary_retreat"],
                            "silently_truncated": False,
                            "arm_config": settings,
                            "revisions": {
                                "model_revision": MODEL_REVISION,
                                "dataset_revision": RULER_REVISION,
                                "code_revision": KVPRESS_REVISION,
                                "official_scorer_sha256": OFFICIAL_SCORER_SHA256,
                            },
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
                                if compatibility_press is not None:
                                    span = settings["protected_prefix_token_span"]
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
                                    record["quota_physical_audit"] = compatibility_audit
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
                                        "quota_physical_audit": compatibility_audit,
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
            _require(global_index == EXPECTED_EXAMPLES, "Phi RULER run did not close.")
            records = root / "records.jsonl"
            partial.replace(records)
            progress.unlink()
            cell = {
                "schema_version": 1,
                "experiment_id": CELL_EXPERIMENT_ID,
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
                "qwen_adaptive_audit": (
                    {
                        "path": str(args.qwen_adaptive_audit),
                        "sha256": qwen_adaptive_audit_digest,
                    }
                    if qwen_adaptive_audit_digest is not None
                    else None
                ),
                "sequence_gate": sequence_gate,
                "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
                "benchmark_dataset_digest_set_sha256": dataset_digest_set,
                "environment": environment,
                "raw_records": {"path": str(records), "sha256": sha256(records)},
            }
            atomic_json(root / "cell.json", cell)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
