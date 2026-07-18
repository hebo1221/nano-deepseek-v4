from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_p2_targeted_stage_a_shard as stage_a

EXPERIMENT_ID = "p2-targeted-stage-a-structural-identifiability-audit-v1"
DEFAULT_OUTPUT = Path(
    "research/adaptive_v4_memory/results/p2-targeted-stage-a-structural-identifiability.audit.json"
)


def _payload_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return stage_a.json_digest(unsigned)


def build_report(
    *,
    training_root: Path = stage_a.TRAINING_ROOT,
    calibration_root: Path = stage_a.CALIBRATION_ROOT,
) -> dict[str, Any]:
    manifest = stage_a.load_manifest()
    source = stage_a.source_state()
    if source["dirty"]:
        raise RuntimeError("Stage-A structural-identifiability audit requires a clean source tree.")
    preflight = stage_a.structural_identifiability_preflight(
        training_root=training_root,
        calibration_root=calibration_root,
    )
    if (
        preflight["expected_seed_scale_budget_records"] != 20
        or preflight["observed_seed_scale_budget_records"] != 20
        or preflight["outcomes_or_807_inputs_inspected"] is not False
    ):
        raise ValueError("Stage-A structural-identifiability coverage drifted.")
    permitted = bool(preflight["current_stage_a_quality_execution_permitted"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": ("terminal_structural_go" if permitted else "terminal_structural_no_go"),
        "terminal": True,
        "observation_boundary": {
            "inputs_read": (
                "Frozen 707-series quota-calibration artifacts, their bound checkpoints, "
                "and controller configurations only."
            ),
            "inputs_not_read": [
                "807-series generated inputs",
                "807-series targets",
                "807-series predictions",
                "any Stage-A quality outcome",
            ],
            "outcome_independent": True,
        },
        "frozen_manifest": {
            "path": str(stage_a.MANIFEST_PATH),
            "sha256": stage_a.sha256(stage_a.MANIFEST_PATH),
            "experiment_id": manifest["experiment_id"],
            "frozen_at": manifest["frozen_at"],
        },
        "source": source,
        "preflight": preflight,
        "decision": {
            "current_clean_layer_identity_panel": "GO" if permitted else "NO_GO",
            "launch_current_3600_run_prospective_integrity_phase": permitted,
            "launch_current_36000_arm_conversation_quality_matrix": permitted,
            "reason": (
                "Every scale-budget cell retains all five checkpoint seeds with at least "
                "200 identified conversations per seed."
                if permitted
                else "At least one frozen checkpoint seed has a structurally uniform quota "
                "mapping in every scale-budget cell, so calibrated and shuffled arms are "
                "identical and the all-five-seed identification gate cannot be met."
            ),
            "allowed_next_action": (
                "Freeze a new outcome-independent, contrast-identified arm design before any "
                "prospective path or 807-series quality execution."
            ),
        },
        "scientific_interpretation": {
            "finding": (
                "The existing calibrated policy often collapses to a uniform layer quota. "
                "A permutation cannot create a layer-identity intervention when every quota "
                "value is equal."
            ),
            "not_a_quality_result": True,
            "not_evidence_of_zero_population_effect": True,
            "not_permission_to_change_the_old_gate_after_outcome_access": True,
        },
        "nonclaims": [
            "no 807-series quality or accuracy result",
            "no adaptive-quota effect estimate",
            "no protected-pin effect estimate",
            "no confirmatory inference",
            "no replacement of the paused 16-arm factorial",
        ],
    }
    report["payload_sha256"] = _payload_digest(report)
    return report


def validate_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    preflight = payload.get("preflight", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") not in {"terminal_structural_go", "terminal_structural_no_go"}
        or payload.get("terminal") is not True
        or payload.get("payload_sha256") != _payload_digest(payload)
        or payload.get("frozen_manifest", {}).get("path") != str(stage_a.MANIFEST_PATH)
        or payload.get("frozen_manifest", {}).get("sha256") != stage_a.sha256(stage_a.MANIFEST_PATH)
        or payload.get("source", {}).get("dirty") is not False
        or payload.get("source", {}).get("implementation_digest") != stage_a.implementation_digest()
        or preflight.get("outcomes_or_807_inputs_inspected") is not False
        or preflight.get("current_stage_a_quality_execution_permitted")
        is not (payload.get("status") == "terminal_structural_go")
    ):
        raise ValueError("Invalid Stage-A structural-identifiability report.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current Stage-A layer-identity feasibility from frozen 707-series "
            "calibrations before spending prospective or quality GPU compute."
        )
    )
    parser.add_argument("--training-root", type=Path, default=stage_a.TRAINING_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=stage_a.CALIBRATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Stage-A identifiability report already exists: {args.output}")
    payload = build_report(
        training_root=args.training_root,
        calibration_root=args.calibration_root,
    )
    stage_a.write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": payload["status"],
                "identified_records": payload["preflight"]["identified_records"],
                "unidentified_records": payload["preflight"]["unidentified_records"],
                "outcomes_or_807_inputs_inspected": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
