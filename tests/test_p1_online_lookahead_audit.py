from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import collect_p1_online_lookahead_labels as labels  # noqa: E402
import evaluate_p1_online_learned_lookahead_shard as evaluator  # noqa: E402
import evaluate_p2_causal_factorial_shard as causal  # noqa: E402
from summarize_p1_online_learned_lookahead import (  # noqa: E402
    _validate_label_payload,
    _validate_test_payload,
    sha256,
)


def _checkpoint(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def _label_shard(tmp_path: Path) -> dict[str, object]:
    family = labels.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = labels.CONTEXTS[0]
    replicate = labels.REPLICATES["train"][0]
    start = replicate * labels.EXAMPLES_PER_SHARD
    failures = [
        {
            "conversation_id": f"{family}:{context}:{index}",
            "input_sha256": hashlib.sha256(f"{family}:{context}:{index}".encode()).hexdigest(),
            "query_position": 1,
            "reason": "no_prior_token_csa_probe",
        }
        for index in range(start, start + labels.EXAMPLES_PER_SHARD)
    ]
    rows: list[dict[str, object]] = []
    return {
        "experiment_id": "p1-online-learned-lookahead-label-shard-v1",
        "scale": "s55",
        "training_seed": labels.TRAINING_SEEDS[0],
        "split": "train",
        "family": family,
        "context": context,
        "replicate": replicate,
        "generation_seed": labels.generation_seed(
            split="train",
            training_seed=labels.TRAINING_SEEDS[0],
            family=family,
            context=context,
            replicate=replicate,
        ),
        "conversations": labels.EXAMPLES_PER_SHARD,
        "risk_examples": 0,
        "failure_count": len(failures),
        "failures": failures,
        "registered_topk_sweep": list(labels.NORMAL_TOPK_SWEEP),
        "dense_topk": [max(labels.NORMAL_TOPK_SWEEP)],
        "rows": rows,
        "rows_digest": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "checkpoint": _checkpoint(tmp_path),
        "leakage_guard": {
            "features_use_prior_token_only": True,
            "causal_offset": 1,
            "benchmark_targets_used_for_labels": False,
            "dense_model_predictions_used_for_labels": True,
            "split_namespace": labels.SPLIT_NAMESPACES["train"],
        },
    }


def _accounting() -> dict[str, int]:
    return {
        "state_bytes": 1,
        "hca_bytes": 2,
        "csa_bytes": 3,
        "index_bytes": 4,
        "logical_cache_bytes": 10,
        "hot_resident_bytes": 6,
        "cold_resident_bytes": 4,
    }


def _test_shard(tmp_path: Path) -> dict[str, object]:
    family = labels.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = labels.CONTEXTS[0]
    replicate = 0
    budget = causal.BUDGET_LABELS[0]
    ids = [f"{family}:{context}:{index}" for index in range(labels.EXAMPLES_PER_SHARD)]
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for offset in range(0, labels.EXAMPLES_PER_SHARD, labels.BATCH_SIZE):
        rotation = (replicate * 5 + offset // labels.BATCH_SIZE) % len(evaluator.ARMS)
        order = (*evaluator.ARMS[rotation:], *evaluator.ARMS[:rotation])
        for execution_index, arm in enumerate(order):
            controller = None
            if arm != "native-resident":
                controller = {
                    "budget_violations": 0,
                    "control_points": 1,
                    "selected_blocks_sum": 1,
                    "budget_blocks_sum": 1,
                }
            for conversation_id in ids[offset : offset + labels.BATCH_SIZE]:
                records.append(
                    {
                        "arm": arm,
                        "conversation_id": conversation_id,
                        "batch_offset": offset,
                        "input_sha256": hashlib.sha256(conversation_id.encode()).hexdigest(),
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                        "budget": budget,
                        "targets": [1],
                        "predictions": [1],
                        "correct": [True],
                        "correct_count": 1,
                        "total": 1,
                        "controller": controller,
                        "failure": None,
                    }
                )
            metrics.append(
                {
                    "batch_offset": offset,
                    "arm": arm,
                    "execution_index": execution_index,
                    "wall_ms": 1.0,
                    "peak_cuda_allocated_bytes": 10,
                    "peak_cuda_reserved_bytes": 12,
                    "accounting": _accounting(),
                    "tier": {
                        "hot_blocks": 1,
                        "logical_blocks": 2,
                        "hot_bytes": 6,
                        "host_bytes": 4,
                        "h2d_bytes": 2,
                        "useful_h2d_bytes": 1,
                        "d2h_bytes": 1,
                        "late_misses": 0,
                        "evictions": 0,
                    },
                    "failure": None,
                    "physical_hot_budget_blocks_by_layer": (
                        None if arm == "native-resident" else {"0": 1}
                    ),
                    "budget_violations": 0,
                }
            )
    records.sort(key=lambda row: (str(row["arm"]), str(row["conversation_id"])))
    return {
        "experiment_id": "p1-online-learned-lookahead-test-shard-v1",
        "scale": "s55",
        "training_seed": labels.TRAINING_SEEDS[0],
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": replicate,
        "examples": labels.EXAMPLES_PER_SHARD,
        "arms": list(evaluator.ARMS),
        "records": records,
        "records_digest": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "batch_metrics": metrics,
        "checkpoint": _checkpoint(tmp_path),
    }


def test_label_audit_requires_exact_conversation_coverage(tmp_path: Path) -> None:
    shard = _label_shard(tmp_path)
    coordinate, input_digests, _checkpoint_digest = _validate_label_payload(
        shard, path=tmp_path / "label.json"
    )
    assert coordinate[0] == "s55"
    assert len(input_digests) == labels.EXAMPLES_PER_SHARD

    missing = deepcopy(shard)
    missing["failures"].pop()  # type: ignore[union-attr]
    missing["failure_count"] = len(missing["failures"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conversation coverage drifted"):
        _validate_label_payload(missing, path=tmp_path / "label.json")


def test_test_shard_audit_recomputes_pairing_and_physical_aggregates(
    tmp_path: Path,
) -> None:
    shard = _test_shard(tmp_path)
    coordinate, records, ids, metrics, violations, failures, _checkpoint_digest = (
        _validate_test_payload(shard, path=tmp_path / "test.json")
    )
    assert coordinate[2] == causal.BUDGET_LABELS[0]
    assert len(records) == labels.EXAMPLES_PER_SHARD * len(evaluator.ARMS)
    assert len(ids) == labels.EXAMPLES_PER_SHARD
    assert len(metrics) == (labels.EXAMPLES_PER_SHARD // labels.BATCH_SIZE) * len(evaluator.ARMS)
    assert violations == failures == 0

    unpaired = deepcopy(shard)
    unpaired["records"][0]["input_sha256"] = "0" * 64  # type: ignore[index]
    unpaired["records_digest"] = hashlib.sha256(
        json.dumps(unpaired["records"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="Paired test inputs drifted"):
        _validate_test_payload(unpaired, path=tmp_path / "test.json")

    wrong_aggregate = deepcopy(shard)
    wrong_aggregate["batch_metrics"][0]["budget_violations"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="budget violation aggregate drifted"):
        _validate_test_payload(wrong_aggregate, path=tmp_path / "test.json")
