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
P4_EXPECTED_CELLS = 216


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


def _validate_evidence(name: str, path: Path, contract: dict[str, Any]) -> dict[str, Any]:
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


def _classify_500k_preflight(payload: dict[str, Any]) -> str:
    audit = payload["audit"]
    successes = audit.get("successful_policy_attempts")
    failures = audit.get("failed_policy_attempts")
    terminal = (
        audit.get("all_terminal_cells_verified") is True
        and audit.get("all_artifact_digests_verified") is True
        and audit.get("context_tokens") == 500_000
        and audit.get("generation_tokens") == 128
        and audit.get("scales_attempted") == 2
        and audit.get("terminal_policy_attempts") == 4
        and type(successes) is int
        and type(failures) is int
        and successes >= 0
        and failures >= 0
        and successes + failures == 4
        and audit.get("performance_claim_available") is False
    )
    if not terminal:
        return "unverified"
    return "bounded-result" if successes > 0 else "negative-result"


def classify_evidence(
    p2_core: dict[str, Any],
    m5_one_token_pilot: dict[str, Any],
    m3_offline_learned_risk_pilot: dict[str, Any],
    p1_online_learned_lookahead: dict[str, Any],
    p2_causal: dict[str, Any],
    p3_ruler: dict[str, Any],
    p3_natural: dict[str, Any],
    p3_safety: dict[str, Any],
    p3_natural_safety: dict[str, Any],
    p3_ifeval: dict[str, Any],
    p3_longsafety: dict[str, Any],
    p4_500k_context: dict[str, Any],
    p4_reference_systems: dict[str, Any],
    p4_production_systems: dict[str, Any],
) -> dict[str, str]:
    core_passed = any(
        row.get("passes_fixed_baseline_component") is True for row in p2_core["quality_gate"]
    )
    m5_audit = m5_one_token_pilot["audit"]
    m5_complete = (
        m5_audit.get("raw_artifacts_verified") is True
        and m5_audit.get("scales_verified") == 2
        and m5_audit.get("workloads_per_scale") == 3
        and m5_audit.get("required_arms_verified") == 4
        and m5_audit.get("one_token_semantics_verified") is True
        and m5_audit.get("pilot_negative_result_verified") is True
    )
    m3_audit = m3_offline_learned_risk_pilot["audit"]
    m3_complete = (
        m3_audit.get("raw_summaries_verified") is True
        and m3_audit.get("scales_verified") == 2
        and m3_audit.get("independent_splits_verified") is True
        and m3_audit.get("train_examples_per_scale") == 768
        and m3_audit.get("calibration_examples_per_scale") == 384
        and m3_audit.get("test_examples_per_scale") == 768
        and m3_audit.get("ablation_variants_verified") == 4
        and m3_audit.get("pareto_failure_verified") is True
        and m3_audit.get("refresh_ablation_available") is False
        and m3_audit.get("offline_native_probe_semantics_verified") is True
        and m3_audit.get("online_lookahead_evidence") is False
        and m3_audit.get("implementation_sources_verified") is True
    )
    learned_audit = p1_online_learned_lookahead["audit"]
    learned_complete = (
        learned_audit.get("label_shards_verified") == 6_750
        and learned_audit.get("policies_verified") == 20
        and learned_audit.get("test_shards_verified") == 9_000
        and learned_audit.get("paired_conversations") == 180_000
        and learned_audit.get("quality_arm_conversations") == 1_080_000
        and learned_audit.get("training_seeds") == 5
        and learned_audit.get("scales") == 2
        and learned_audit.get("families") == 9
        and learned_audit.get("contexts") == 5
        and learned_audit.get("budgets") == 2
        and learned_audit.get("all_raw_digests_verified") is True
        and learned_audit.get("all_dependencies_verified") is True
        and learned_audit.get("implementation_digests_verified") is True
        and learned_audit.get("dependency_artifact_digests_verified") is True
        and learned_audit.get("all_inputs_paired") is True
        and learned_audit.get("zero_budget_violations") is True
        and learned_audit.get("complete_failure_accounting") is True
        and learned_audit.get("online_token_offset_verified") is True
        and learned_audit.get("native_bootstrap_accounted") is True
        and learned_audit.get("cache_replay_contract_tested") is True
        and learned_audit.get("resolution_aware_gate_verified") is True
    )
    learned_passed = p1_online_learned_lookahead["primary_gate"].get("passed") is True
    causal_passed = p2_causal["primary_causal_gate"].get("passed") is True
    p3_complete = p3_ruler.get("benchmark_complete") is True
    natural_audit = p3_natural["audit"]
    natural_complete = (
        natural_audit.get("all_required_artifacts_verified") is True
        and natural_audit.get("all_required_baseline_cells_terminal") is True
        and natural_audit.get("all_failure_accounting_complete") is True
        and natural_audit.get("all_source_implementations_verified") is True
        and natural_audit.get("safety_stress_terminal") is True
        and natural_audit.get("natural_safety_terminal") is True
        and natural_audit.get("benchmarks_terminal") == 5
        and natural_audit.get("minimum_protocol_examples_accounted_per_arm") == 45_289
    )
    safety_audit = p3_safety["audit"]
    safety_complete = (
        safety_audit.get("required_arms_terminal") is True
        and safety_audit.get("failure_accounting_complete") is True
        and safety_audit.get("input_pairing_verified") is True
        and safety_audit.get("protected_prefix_physical_budget_verified") is True
        and safety_audit.get("examples_accounted_per_arm") == 1_200
        and safety_audit.get("families_terminal") == 4
        and safety_audit.get("contexts_terminal") == 3
    )
    natural_safety_audit = p3_natural_safety["audit"]
    natural_safety_complete = (
        natural_safety_audit.get("required_arms") == 2
        and natural_safety_audit.get("longsafety_generation_terminal") is True
        and natural_safety_audit.get("longsafety_input_pairing_verified") is True
        and natural_safety_audit.get("longsafety_expected_generations_per_arm") == 3_086
        and natural_safety_audit.get("longsafety_official_judge_status") == "blocked"
        and natural_safety_audit.get("longsafety_safety_scores_reported") is False
        and natural_safety_audit.get("ifeval_official_terminal") is True
        and natural_safety_audit.get("ifeval_input_pairing_verified") is True
        and natural_safety_audit.get("ifeval_expected_prompts_per_arm") == 541
        and natural_safety_audit.get("failure_accounting_complete") is True
        and natural_safety_audit.get("comparative_long_context_safety_claim_available") is False
    )
    ifeval_audit = p3_ifeval["audit"]
    ifeval_complete = (
        ifeval_audit.get("required_arms_terminal") is True
        and ifeval_audit.get("input_pairing_verified") is True
        and ifeval_audit.get("official_scoring_accounted") is True
        and ifeval_audit.get("expected_prompts_per_arm") == 541
    )
    longsafety_audit = p3_longsafety["audit"]
    longsafety_judged = (
        longsafety_audit.get("generation_arms_terminal") is True
        and longsafety_audit.get("input_pairing_verified") is True
        and longsafety_audit.get("generation_failure_accounting_complete") is True
        and longsafety_audit.get("expected_generations_total") == 6_172
        and longsafety_audit.get("official_judge_status") == "complete"
    )
    preflight_class = _classify_500k_preflight(p4_500k_context)
    reference_audit = p4_reference_systems["audit"]
    reference_complete = reference_audit.get("terminal_cells") == P4_EXPECTED_CELLS
    production_audit = p4_production_systems["audit"]
    production_full = (
        production_audit.get("terminal_cells") == P4_EXPECTED_CELLS
        and production_audit.get("complete_cells") == P4_EXPECTED_CELLS
        and production_audit.get("partial_cells") == 0
        and production_audit.get("failed_cells") == 0
        and production_audit.get("actual_concurrency_verified") is True
        and production_audit.get("all_required_metrics_verified") is True
        and production_audit.get("backend_provenance_consistent") is True
        and production_audit.get("tail_failure_accounting_complete") is True
        and production_audit.get("all_paired_predictions_identical") is True
    )
    production_terminal = production_audit.get("terminal_cells") == P4_EXPECTED_CELLS
    production_class = (
        "success"
        if production_full
        else "bounded-result"
        if production_terminal and production_audit.get("complete_cells", 0) > 0
        else "unverified"
    )
    result = {
        "p2_core": "success" if core_passed else "negative-result",
        "m5_one_token_pilot": "negative-result" if m5_complete else "unverified",
        "m3_offline_learned_risk_pilot": ("negative-result" if m3_complete else "unverified"),
        "p1_online_learned_lookahead": (
            "success"
            if learned_complete and learned_passed
            else "negative-result"
            if learned_complete
            else "unverified"
        ),
        "p2_causal": "success" if causal_passed else "bounded-result",
        "p3_ruler": "bounded-result" if p3_complete else "unverified",
        "p3_natural": "bounded-result" if natural_complete else "unverified",
        "p3_safety": "bounded-result" if safety_complete else "unverified",
        "p3_natural_safety": ("bounded-result" if natural_safety_complete else "unverified"),
        "p3_ifeval": "bounded-result" if ifeval_complete else "unverified",
        "p3_longsafety": "bounded-result" if longsafety_judged else "unverified",
        "p4_500k_context": preflight_class,
        "p4_reference_systems": "bounded-result" if reference_complete else "unverified",
        "p4_production_systems": production_class,
        "production_runtime_blocker": "unverified",
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


def _learned_lookahead_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    systems = {
        (row["scale"], row["budget"]): row for row in payload["primary_gate"]["system_cells"]
    }
    rows = []
    for cell in payload["primary_gate"]["cells"]:
        system = systems[(cell["scale"], cell["budget"])]
        rows.append(
            {
                **cell,
                "seed_cluster_bootstrap_ci": json.dumps(
                    cell["seed_cluster_bootstrap_ci"], separators=(",", ":")
                ),
                **{
                    name: value
                    for name, value in system.items()
                    if name not in {"scale", "budget", "passed"}
                },
                "quality_passed": cell["passed"],
                "system_passed": system["passed"],
            }
        )
    return rows


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


def _p3_safety_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"arm": arm, **row}
        for arm, arm_payload in payload["arms"].items()
        for row in arm_payload["slices"]
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
        coordinates = cell["cell"]
        row = {
            **coordinates,
            "active_requests": coordinates.get(
                "active_requests", coordinates.get("concurrency", "")
            ),
            "concurrency": coordinates.get("concurrency", ""),
            "status": cell["status"],
        }
        row.update(
            {
                "paired_repetitions": cell["paired_repetitions"],
                "resident_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "resident"),
                "tiered_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "tiered"),
                "resident_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "resident"
                ),
                "tiered_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "tiered"
                ),
                "resident_peak_hbm_mean": _metric_mean(cell, "peak_allocated_bytes", "resident"),
                "tiered_peak_hbm_mean": _metric_mean(cell, "peak_allocated_bytes", "tiered"),
                "failure": "",
            }
        )
        rows.append(row)
    for failure in payload["failure_table"]:
        rows.append(
            {
                **failure["cell"],
                "active_requests": failure["cell"].get(
                    "active_requests", failure["cell"].get("concurrency", "")
                ),
                "concurrency": failure["cell"].get("concurrency", ""),
                "status": "failed",
                "paired_repetitions": 0,
                "failure": json.dumps(failure.get("policy_status"), sort_keys=True),
            }
        )
    return rows


def _p4_500k_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scale": cell["scale"],
            "policy": policy,
            "context_tokens": payload["audit"]["context_tokens"],
            "generation_tokens": payload["audit"]["generation_tokens"],
            "status": attempt["status"],
            "prediction_digest": attempt.get("prediction_digest"),
            "peak_allocated_bytes": attempt.get("peak_allocated_bytes"),
            "pinned_host_bytes": attempt.get("pinned_host_bytes"),
            "error_type": attempt.get("error_type"),
            "error": attempt.get("error"),
        }
        for cell in payload["cells"]
        for policy, attempt in cell["policy_attempts"].items()
    ]


def _report(
    *,
    classifications: dict[str, str],
    p2_core: dict[str, Any],
    m5_one_token_pilot: dict[str, Any],
    m3_offline_learned_risk_pilot: dict[str, Any],
    p1_online_learned_lookahead: dict[str, Any],
    p2_causal: dict[str, Any],
    p3_ruler: dict[str, Any],
    p3_natural: dict[str, Any],
    p3_safety: dict[str, Any],
    p3_natural_safety: dict[str, Any],
    p3_ifeval: dict[str, Any],
    p3_longsafety: dict[str, Any],
    p4_500k_context: dict[str, Any],
    p4_reference_systems: dict[str, Any],
    p4_production_systems: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> str:
    causal = p2_causal["primary_causal_gate"]
    p4_500k = p4_500k_context["audit"]
    p4_500k_correctness = p4_500k_context["correctness"]
    p4_reference = p4_reference_systems["audit"]
    p4_production = p4_production_systems["audit"]
    evidence_lines = "\n".join(
        f"| {row['name']} | {classifications.get(row['name'], 'unverified')} | `{row['sha256']}` |"
        for row in inputs
    )
    return f"""# Adaptive V4 Memory: paper-grade empirical report

This report is generated only from digest-bound audit artifacts. Conclusion labels are
mechanical and deliberately narrower than the motivating hypothesis.

## Evidence ledger

| Evidence | Classification | SHA-256 |
|---|---|---|
{evidence_lines}

## Experiment volume

- P2 core: {p2_core["audit"]["unique_shards"]:,} verified shards, 5 training seeds,
  2 scales, 9 workload families, 5 contexts, and 1,000 examples per
  seed-scale-family.
- M5 one-token baseline: {m5_one_token_pilot["audit"]["scales_verified"]} scales,
  {m5_one_token_pilot["audit"]["workloads_per_scale"]} synthetic workloads per scale,
  classified only as a pilot negative result for the tested interface.
- M3 offline learned-risk pilot: {m3_offline_learned_risk_pilot["audit"]["scales_verified"]} scales with
  disjoint train/calibration/test splits and
  {m3_offline_learned_risk_pilot["audit"]["ablation_variants_verified"]} ablations; classified
  as a negative Pareto result. It used final-query probes from a full native pass to build an
  offline replay plan and is explicitly not evidence for deployable online learned lookahead.
- P1 online learned lookahead: {p1_online_learned_lookahead["audit"]["test_shards_verified"]:,}
  held-out shards, {p1_online_learned_lookahead["audit"]["paired_conversations"]:,} paired
  conversations per arm, 5 seeds and 2 scales; the separate exploratory gate passed:
  **{p1_online_learned_lookahead["primary_gate"]["passed"]}**.
- P2 causal: {p2_causal["audit"]["unique_shards"]:,} verified factorial shards;
  {p2_causal["audit"]["quality_execution_counts"]["executed"]:,} quality forwards were
  executed and {p2_causal["audit"]["quality_execution_counts"]["reused_exact_config"]:,}
  arm-batches reused an exact byte-identical config; the calibrated+pins versus
  fixed+pins gate passed: **{causal["passed"]}**.
- P2 supplemental baselines: fixed top-p 0.5/0.8 are evaluated on the complete
  factorial, and the target-aware registered-arm oracle is reported only as a
  non-causal upper bound over {len(p2_causal["offline_oracle_upper_bound"]["registered_arms"])} arms.
- P3 RULER: {p3_ruler["audit"]["completed_cells"]} cells and
  {p3_ruler["audit"]["total_predictions"]:,} predictions on one pinned compatible model.
- P3 natural suite: {p3_natural["audit"]["benchmarks_terminal"]} terminal benchmarks and
  at least {p3_natural["audit"]["minimum_protocol_examples_accounted_per_arm"]:,}
  examples accounted per required arm.
- P3 safety stress: {p3_safety["audit"]["examples_accounted_per_arm"]:,} examples per arm,
  {p3_safety["audit"]["families_terminal"]} families, and
  {p3_safety["audit"]["contexts_terminal"]} context lengths with paired inputs.
- P3 natural safety suite: {p3_natural_safety["audit"]["required_arms"]} paired arms;
  LongSafety generation and IFEval official scoring are terminal, while the paid
  LongSafety judge remains **{p3_natural_safety["audit"]["longsafety_official_judge_status"]}**.
- P3 IFEval control: {p3_ifeval["audit"]["expected_prompts_per_arm"]:,} officially scored
  prompts per required arm.
- P3 LongSafety: {p3_longsafety["audit"]["expected_generations_total"]:,} digest-bound
  generations; official paid judge status is **{p3_longsafety["audit"]["official_judge_status"]}**,
  so no comparative LongSafety safety score is claimed.
- P4 500K feasibility: {p4_500k["terminal_policy_attempts"]} terminal scale-policy
  attempts, {p4_500k["successful_policy_attempts"]} successful and
  {p4_500k["failed_policy_attempts"]} failed. This single-attempt preflight carries
  no performance claim; {p4_500k_correctness["scales_with_both_policies_successful"]}
  paired-success scales had prediction equality
  **{p4_500k_correctness["all_successful_pair_predictions_identical"]}**.
- P4 reference systems: {p4_reference["terminal_cells"]} terminal serial-interleaved cells,
  {p4_reference["complete_cells"]} complete, {p4_reference["partial_cells"]} partial, and
  {p4_reference["failed_cells"]} failed.
- P4 production systems: {p4_production["terminal_cells"]} terminal actual-concurrency cells,
  {p4_production["complete_cells"]} complete, {p4_production["partial_cells"]} partial, and
  {p4_production["failed_cells"]} failed.

## Claim boundary

The P2 result is synthetic Tier-S evidence. A failed causal gate bounds only the tested
controller family. P3 quality and synthetic safety-retention results are transfer evidence
for pinned Qwen3 snapshots, not comprehensive safety certification, official DeepSeek-V4
evidence, or model-population inference. P4 reference evidence is single-accelerator and
serial-interleaved. The 500K result is feasibility-only on Tier-S scales; the checked
production adapter is static continuous batching, not an
external dynamic/fused/multi-GPU serving runtime. Official DeepSeek-V4 stays unverified until its
frozen resource contract is satisfied.

## Reproduction

The CSV tables next to this report are generated from the same frozen audits. Their digests,
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
        loaded["m5_one_token_pilot"],
        loaded["m3_offline_learned_risk_pilot"],
        loaded["p1_online_learned_lookahead"],
        loaded["p2_causal"],
        loaded["p3_ruler"],
        loaded["p3_natural"],
        loaded["p3_safety"],
        loaded["p3_natural_safety"],
        loaded["p3_ifeval"],
        loaded["p3_longsafety"],
        loaded["p4_500k_context"],
        loaded["p4_reference_systems"],
        loaded["p4_production_systems"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_rows = [
        {**row, "classification": classes.get(row["name"], "unverified")} for row in inputs
    ]
    _write_csv(
        output_root / "table-evidence.csv",
        evidence_rows,
        ["name", "classification", "path", "sha256"],
    )
    quality = _p2_quality_rows(loaded["p2_core"])
    _write_csv(output_root / "table-p2-quality-gate.csv", quality, list(quality[0]))
    learned = _learned_lookahead_rows(loaded["p1_online_learned_lookahead"])
    _write_csv(
        output_root / "table-p1-online-learned-lookahead-gate.csv",
        learned,
        list(learned[0]),
    )
    causal = _causal_rows(loaded["p2_causal"])
    _write_csv(output_root / "table-p2-causal-gate.csv", causal, list(causal[0]))
    p3 = _p3_rows(loaded["p3_ruler"])
    _write_csv(output_root / "table-p3-ruler-cells.csv", p3, list(p3[0]))
    p3_safety = _p3_safety_rows(loaded["p3_safety"])
    _write_csv(
        output_root / "table-p3-safety-slices.csv",
        p3_safety,
        list(p3_safety[0]),
    )
    p4_500k = _p4_500k_rows(loaded["p4_500k_context"])
    _write_csv(
        output_root / "table-p4-500k-context.csv",
        p4_500k,
        list(p4_500k[0]),
    )
    p4_reference = _p4_rows(loaded["p4_reference_systems"])
    p4_production = _p4_rows(loaded["p4_production_systems"])
    p4_fields = [
        "scale",
        "context",
        "generation",
        "profile",
        "batch",
        "active_requests",
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
    _write_csv(
        output_root / "table-p4-reference-system-cells.csv",
        p4_reference,
        p4_fields,
    )
    _write_csv(
        output_root / "table-p4-production-system-cells.csv",
        p4_production,
        p4_fields,
    )
    report = _report(
        classifications=classes,
        p2_core=loaded["p2_core"],
        m5_one_token_pilot=loaded["m5_one_token_pilot"],
        m3_offline_learned_risk_pilot=loaded["m3_offline_learned_risk_pilot"],
        p1_online_learned_lookahead=loaded["p1_online_learned_lookahead"],
        p2_causal=loaded["p2_causal"],
        p3_ruler=loaded["p3_ruler"],
        p3_natural=loaded["p3_natural"],
        p3_safety=loaded["p3_safety"],
        p3_natural_safety=loaded["p3_natural_safety"],
        p3_ifeval=loaded["p3_ifeval"],
        p3_longsafety=loaded["p3_longsafety"],
        p4_500k_context=loaded["p4_500k_context"],
        p4_reference_systems=loaded["p4_reference_systems"],
        p4_production_systems=loaded["p4_production_systems"],
        inputs=inputs,
    )
    (output_root / "paper-report.md").write_text(report)
    generated = [
        output_root / name for name in manifest["generated_files"] if name != "artifact-index.json"
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
        "generated": [{"path": str(path), "sha256": sha256(path)} for path in generated],
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
