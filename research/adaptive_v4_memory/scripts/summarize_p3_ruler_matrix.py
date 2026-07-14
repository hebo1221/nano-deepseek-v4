from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from run_p3_ruler_matrix import ARMS, LENGTHS, cell_dir, cells, sha256
from summarize_p2_core_matrix import bootstrap_paired_mean, holm_bonferroni

EXPECTED_ROWS_PER_CELL = 6_500
EXPECTED_CELLS = 39


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _records_digest(frame: pd.DataFrame) -> str:
    columns = ["row_id", "task", "string_match"]
    payload = frame[columns].sort_values("row_id").to_dict(orient="records")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_cell(
    path: Path,
    *,
    expected_length: int,
    expected_arm: str,
    expected_ratio: float,
    manifest_digest: str,
    runner_digest: str,
    causal_gate_digest: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    audit_path = path / "audit.json"
    _require(audit_path.is_file(), f"Missing P3 audit: {audit_path}")
    audit = json.loads(audit_path.read_text())
    cell = audit.get("cell", {})
    _require(audit.get("experiment_id") == "p3-ruler-qwen3-1.7b-cell-v1", "Wrong cell id.")
    _require(audit.get("status") == "complete", "P3 cell is not complete.")
    _require(audit.get("source", {}).get("dirty") is False, "Dirty P3 cell source.")
    _require(
        audit.get("source", {}).get("implementation_sha256") == runner_digest,
        "P3 runner implementation drifted.",
    )
    _require(
        audit.get("experiment_manifest", {}).get("sha256") == manifest_digest,
        "P3 manifest drifted.",
    )
    _require(
        audit.get("upstreams", {}).get("causal_gate_sha256") == causal_gate_digest,
        "P3 causal-gate dependency drifted.",
    )
    _require(
        cell.get("length_tokens") == expected_length
        and cell.get("arm") == expected_arm
        and cell.get("ratio") == expected_ratio,
        "P3 cell coordinates drifted.",
    )
    outputs = audit.get("outputs", {})
    for name in ("predictions", "metrics", "config"):
        metadata = outputs.get(name, {})
        artifact = Path(metadata.get("path", ""))
        _require(artifact.is_file(), f"Missing P3 {name}: {artifact}")
        _require(metadata.get("sha256") == sha256(artifact), f"P3 {name} drifted.")
    predictions = pd.read_csv(outputs["predictions"]["path"])
    _require(len(predictions) == EXPECTED_ROWS_PER_CELL, "P3 prediction count drifted.")
    _require(predictions["row_id"].is_unique, "P3 row ids are not unique.")
    _require(predictions["string_match"].notna().all(), "P3 scores contain missing values.")
    _require(
        predictions["string_match"].between(0.0, 1.0).all(),
        "P3 scores are outside [0, 1].",
    )
    _require(
        int(audit.get("measurements", {}).get("rows", -1)) == EXPECTED_ROWS_PER_CELL,
        "P3 audit row count drifted.",
    )
    return audit, predictions


def paired_cell_statistics(
    candidate: pd.DataFrame,
    native: pd.DataFrame,
    *,
    label: str,
    resamples: int = 10_000,
) -> dict[str, Any]:
    paired = candidate[["row_id", "task", "string_match"]].merge(
        native[["row_id", "task", "string_match"]],
        on="row_id",
        suffixes=("_candidate", "_native"),
        validate="one_to_one",
    )
    _require(len(paired) == EXPECTED_ROWS_PER_CELL, "P3 paired row coverage drifted.")
    _require(
        (paired["task_candidate"] == paired["task_native"]).all(),
        "P3 paired tasks drifted.",
    )
    paired["difference"] = (
        paired["string_match_candidate"] - paired["string_match_native"]
    )
    overall = bootstrap_paired_mean(
        paired["difference"].tolist(), label=f"{label}:overall", resamples=resamples
    )
    task_rows: list[dict[str, Any]] = []
    for task, group in paired.groupby("task_candidate", sort=True):
        task_rows.append(
            {
                "task": str(task),
                **bootstrap_paired_mean(
                    group["difference"].tolist(),
                    label=f"{label}:task:{task}",
                    resamples=resamples,
                ),
            }
        )
    adjusted = holm_bonferroni(
        {row["task"]: row["paired_sign_flip_two_sided_p"] for row in task_rows}
    )
    for row in task_rows:
        row["holm_adjusted_p"] = adjusted[row["task"]]
    return {
        "paired_rows": len(paired),
        "candidate_accuracy": float(candidate["string_match"].mean()),
        "native_accuracy": float(native["string_match"].mean()),
        "overall": overall,
        "by_task_with_holm_bonferroni": task_rows,
        "worst_task": min(task_rows, key=lambda row: row["mean_difference"]),
        "inference_boundary": (
            "example-level descriptive uncertainty for one frozen model snapshot; "
            "not training-seed or model-population inference"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete P3 RULER matrix.")
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
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    _require(manifest.get("experiment_id") == "p3-ruler-qwen3-1.7b-v1", "Wrong manifest.")
    _require(manifest.get("execution", {}).get("total_cells") == EXPECTED_CELLS, "Cell drift.")
    _require(
        manifest.get("execution", {}).get("total_predictions")
        == EXPECTED_CELLS * EXPECTED_ROWS_PER_CELL,
        "Prediction design drifted.",
    )
    manifest_digest = sha256(args.manifest)
    causal_gate_digest = sha256(args.causal_gate)
    runner_path = Path(__file__).with_name("run_p3_ruler_matrix.py")
    runner_digest = sha256(runner_path)
    expected_cells = cells(LENGTHS, tuple(ARMS), None)
    _require(len(expected_cells) == EXPECTED_CELLS, "Runner cell design drifted.")
    loaded: dict[tuple[int, str, float], tuple[dict[str, Any], pd.DataFrame]] = {}
    for length, arm, _press, ratio in expected_cells:
        loaded[(length, arm, ratio)] = _load_cell(
            cell_dir(args.output_root, length, arm, ratio),
            expected_length=length,
            expected_arm=arm,
            expected_ratio=ratio,
            manifest_digest=manifest_digest,
            runner_digest=runner_digest,
            causal_gate_digest=causal_gate_digest,
        )

    cell_summary: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    raw_digests: list[str] = []
    for (length, arm, ratio), (audit, predictions) in sorted(loaded.items()):
        raw_digests.append(_records_digest(predictions))
        tasks = [
            {
                "task": str(task),
                "rows": len(group),
                "accuracy": float(group["string_match"].mean()),
            }
            for task, group in predictions.groupby("task", sort=True)
        ]
        cell_summary.append(
            {
                "length_tokens": length,
                "arm": arm,
                "compression_ratio": ratio,
                "rows": len(predictions),
                "accuracy": float(predictions["string_match"].mean()),
                "by_task": tasks,
                "elapsed_seconds": audit["measurements"]["elapsed_seconds"],
                "peak_cuda_allocated_bytes": audit["measurements"][
                    "peak_cuda_allocated_bytes"
                ],
                "peak_cuda_reserved_bytes": audit["measurements"][
                    "peak_cuda_reserved_bytes"
                ],
            }
        )
        if arm != "native":
            native = loaded[(length, "native", 0.0)][1]
            comparisons.append(
                {
                    "length_tokens": length,
                    "arm": arm,
                    "compression_ratio": ratio,
                    **paired_cell_statistics(
                        predictions,
                        native,
                        label=f"p3-ruler:{length}:{arm}:{ratio:.2f}",
                    ),
                }
            )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "P3 RULER summarization requires a clean source tree.")
    total_predictions = sum(row["rows"] for row in cell_summary)
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-ruler-qwen3-1.7b-audit-v1",
        "source": {"commit": source_commit, "dirty": False},
        "experiment_manifest": {"path": str(args.manifest), "sha256": manifest_digest},
        "causal_gate": {"path": str(args.causal_gate), "sha256": causal_gate_digest},
        "audit": {
            "all_cells_verified": True,
            "all_output_digests_verified": True,
            "completed_cells": len(loaded),
            "total_predictions": total_predictions,
            "prediction_record_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "cell_summary": cell_summary,
        "paired_vs_native": comparisons,
        "benchmark_complete": len(loaded) == EXPECTED_CELLS,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "claim_boundary": (
            "Complete RULER evidence on one compatible Qwen3-1.7B snapshot; this is not "
            "official DeepSeek-V4 evidence and does not estimate model-population variance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "completed_cells": len(loaded),
                "total_predictions": total_predictions,
                "benchmark_complete": payload["benchmark_complete"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
