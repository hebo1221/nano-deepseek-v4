from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ALLOWED_CLASSES = {"success", "bounded-result", "negative-result", "unverified"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing required P5 input: {path}")
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"P5 input is not a JSON object: {path}")
    return payload


def _clean_source() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return commit, dirty


def _validate_evidence(
    name: str, path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    payload = _load(path)
    _require(
        payload.get("experiment_id") == contract["experiment_id"],
        f"Wrong {name} experiment id.",
    )
    _require(payload.get("source", {}).get("dirty") is False, f"Dirty {name} evidence.")
    audit = payload.get("audit", {})
    for field, expected in contract["required_audit"].items():
        _require(audit.get(field) == expected, f"{name} audit field {field} drifted.")
    return payload


def classify_evidence(
    p2_core: dict[str, Any],
    p2_causal: dict[str, Any],
    p3_ruler: dict[str, Any],
    p4_systems: dict[str, Any],
) -> dict[str, str]:
    core_passed = any(
        row.get("passes_fixed_baseline_component") is True
        for row in p2_core["quality_gate"]
    )
    causal_passed = p2_causal["primary_causal_gate"].get("passed") is True
    p3_complete = p3_ruler.get("benchmark_complete") is True
    p4_audit = p4_systems["audit"]
    p4_full = (
        p4_audit.get("terminal_cells") == 108
        and p4_audit.get("partial_cells") == 0
        and p4_audit.get("failed_cells") == 0
    )
    result = {
        "p2_core": "success" if core_passed else "negative-result",
        "p2_causal": "success" if causal_passed else "bounded-result",
        "p3_ruler": "bounded-result" if p3_complete else "unverified",
        "p4_systems": "success" if p4_full else "bounded-result",
        "p3_natural_remaining": "unverified",
        "official_deepseek_v4": "unverified",
    }
    _require(set(result.values()).issubset(ALLOWED_CLASSES), "Unknown conclusion class.")
    return result


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _p2_quality_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "holm_significant_positive_families_by_scale": json.dumps(
                row["holm_significant_positive_families_by_scale"], sort_keys=True
            ),
        }
        for row in payload["quality_gate"]
    ]


def _causal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload["primary_causal_gate"]["cells"])


def _p3_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "length_tokens": row["length_tokens"],
            "arm": row["arm"],
            "compression_ratio": row["compression_ratio"],
            "rows": row["rows"],
            "accuracy": row["accuracy"],
            "elapsed_seconds": row["elapsed_seconds"],
            "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": row["peak_cuda_reserved_bytes"],
        }
        for row in payload["cell_summary"]
    ]


def _metric_mean(cell: dict[str, Any], metric: str, policy: str) -> Any:
    value = cell.get("metrics", {}).get(metric, {}).get(policy)
    return "" if value is None else value["mean"]


def _p4_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]:
        row = {**cell["cell"], "status": cell["status"]}
        row.update(
            {
                "paired_repetitions": cell["paired_repetitions"],
                "resident_ttft_p95_ms_mean": _metric_mean(
                    cell, "ttft_p95_ms", "resident"
                ),
                "tiered_ttft_p95_ms_mean": _metric_mean(
                    cell, "ttft_p95_ms", "tiered"
                ),
                "resident_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "resident"
                ),
                "tiered_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "tiered"
                ),
                "resident_peak_hbm_mean": _metric_mean(
                    cell, "peak_allocated_bytes", "resident"
                ),
                "tiered_peak_hbm_mean": _metric_mean(
                    cell, "peak_allocated_bytes", "tiered"
                ),
                "failure": "",
            }
        )
        rows.append(row)
    for failure in payload["failure_table"]:
        rows.append(
            {
                **failure["cell"],
                "status": "failed",
                "paired_repetitions": 0,
                "failure": json.dumps(failure.get("policy_status"), sort_keys=True),
            }
        )
    return rows


def _report(
    *,
    classifications: dict[str, str],
    p2_core: dict[str, Any],
    p2_causal: dict[str, Any],
    p3_ruler: dict[str, Any],
    p4_systems: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> str:
    causal = p2_causal["primary_causal_gate"]
    p4 = p4_systems["audit"]
    evidence_lines = "\n".join(
        f"| {row['name']} | {classifications[row['name']]} | `{row['sha256']}` |"
        for row in inputs
        if row["name"] in classifications
    )
    return f"""# Adaptive V4 Memory: paper-grade empirical report

This report is generated only from digest-bound audit artifacts. Conclusion labels are
mechanical and deliberately narrower than the motivating hypothesis.

## Evidence ledger

| Evidence | Classification | SHA-256 |
|---|---|---|
{evidence_lines}
| p3_natural_remaining | {classifications['p3_natural_remaining']} | see frozen manifest |
| official_deepseek_v4 | {classifications['official_deepseek_v4']} | see blocker manifest |

## Experiment volume

- P2 core: {p2_core['audit']['unique_shards']:,} verified shards, 5 training seeds,
  2 scales, 9 workload families, 5 contexts, and 1,000 examples per
  seed-scale-family.
- P2 causal: {p2_causal['audit']['unique_shards']:,} verified factorial shards;
  the calibrated+pins versus fixed+pins gate passed: **{causal['passed']}**.
- P3 RULER: {p3_ruler['audit']['completed_cells']} cells and
  {p3_ruler['audit']['total_predictions']:,} predictions on one pinned compatible model.
- P4 systems: {p4['terminal_cells']} terminal cells, {p4['complete_cells']} complete,
  {p4['partial_cells']} partial, and {p4['failed_cells']} failed.

## Claim boundary

The P2 result is synthetic Tier-S evidence. A failed causal gate bounds only the tested
controller family. P3 RULER is transfer evidence for one Qwen3 snapshot, not official
DeepSeek-V4 evidence or model-population inference. P4 is single-accelerator reference
PyTorch evidence, not fused-kernel or production-serving evidence. The remaining natural
suite and official DeepSeek-V4 run stay unverified until their frozen execution and resource
contracts are satisfied.

## Reproduction

The CSV tables next to this report are generated from the same four audits. Their digests,
the input digests, source commit, and protocol manifests are recorded in
`artifact-index.json`; missing or incomplete evidence causes generation to fail rather than
being imputed.
"""


def build_package(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    _require(
        manifest.get("experiment_id") == "adaptive-v4-memory-p5-paper-package-v1",
        "Wrong P5 package manifest.",
    )
    loaded: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for name, contract in manifest["evidence"].items():
        path = Path(contract["path"])
        loaded[name] = _validate_evidence(name, path, contract)
        inputs.append({"name": name, "path": str(path), "sha256": sha256(path)})
    for name, raw_path in manifest["boundary_manifests"].items():
        path = Path(raw_path)
        _load(path)
        inputs.append({"name": name, "path": str(path), "sha256": sha256(path)})

    classes = classify_evidence(
        loaded["p2_core"],
        loaded["p2_causal"],
        loaded["p3_ruler"],
        loaded["p4_systems"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_rows = [
        {**row, "classification": classes.get(row["name"], "unverified")}
        for row in inputs
    ]
    _write_csv(
        output_root / "table-evidence.csv",
        evidence_rows,
        ["name", "classification", "path", "sha256"],
    )
    quality = _p2_quality_rows(loaded["p2_core"])
    _write_csv(output_root / "table-p2-quality-gate.csv", quality, list(quality[0]))
    causal = _causal_rows(loaded["p2_causal"])
    _write_csv(output_root / "table-p2-causal-gate.csv", causal, list(causal[0]))
    p3 = _p3_rows(loaded["p3_ruler"])
    _write_csv(output_root / "table-p3-ruler-cells.csv", p3, list(p3[0]))
    p4 = _p4_rows(loaded["p4_systems"])
    p4_fields = [
        "scale",
        "context",
        "generation",
        "profile",
        "batch",
        "concurrency",
        "status",
        "paired_repetitions",
        "resident_ttft_p95_ms_mean",
        "tiered_ttft_p95_ms_mean",
        "resident_throughput_mean",
        "tiered_throughput_mean",
        "resident_peak_hbm_mean",
        "tiered_peak_hbm_mean",
        "failure",
    ]
    _write_csv(output_root / "table-p4-system-cells.csv", p4, p4_fields)
    report = _report(
        classifications=classes,
        p2_core=loaded["p2_core"],
        p2_causal=loaded["p2_causal"],
        p3_ruler=loaded["p3_ruler"],
        p4_systems=loaded["p4_systems"],
        inputs=inputs,
    )
    (output_root / "paper-report.md").write_text(report)
    generated = [
        output_root / name
        for name in manifest["generated_files"]
        if name != "artifact-index.json"
    ]
    commit, dirty = _clean_source()
    _require(not dirty, "P5 package generation requires a clean source tree.")
    index = {
        "schema_version": 1,
        "experiment_id": "adaptive-v4-memory-p5-artifact-index-v1",
        "source": {"commit": commit, "dirty": False},
        "package_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "inputs": inputs,
        "classifications": classes,
        "generated": [
            {"path": str(path), "sha256": sha256(path)} for path in generated
        ],
    }
    target = output_root / "artifact-index.json"
    target.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict P5 paper package.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p5-paper-package-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p5"),
    )
    args = parser.parse_args()
    index = build_package(args.manifest, args.output_root)
    print(json.dumps(index["classifications"], sort_keys=True))


if __name__ == "__main__":
    main()
