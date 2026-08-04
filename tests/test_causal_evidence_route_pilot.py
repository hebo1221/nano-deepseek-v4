from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from nano_deepseek_v4 import AdaptiveMemoryWorkloadBatch, CSAProbeRecord, CSASelectionProbe

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
pilot = importlib.import_module("run_causal_evidence_route_pilot")


def _probe() -> CSASelectionProbe:
    scores = torch.full((1, 25, 6), float("-inf"))
    scores[0, 20, :5] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    scores[0, 24] = torch.tensor([1.0, 2.0, 9.0, 4.0, 7.0, 8.0])
    probe = CSASelectionProbe()
    probe.records.append(
        CSAProbeRecord(
            layer_index=2,
            scores=scores,
            block_end_positions=torch.tensor([[3, 7, 11, 15, 19, 23]]),
            query_positions=torch.arange(25).unsqueeze(0),
            query_features=torch.zeros(1, 25, 4),
            value_blocks=torch.zeros(1, 6, 4),
        )
    )
    return probe


def _workload() -> AdaptiveMemoryWorkloadBatch:
    return AdaptiveMemoryWorkloadBatch(
        family="multiple-independent-needles",
        input_ids=torch.zeros((1, 25), dtype=torch.long),
        targets=torch.tensor([[1, 2]]),
        query_positions=torch.tensor([[20, 24]]),
        evidence_positions=torch.tensor([[1, 9]]),
        conversation_ids=("test",),
    )


def test_matched_routes_differ_by_exactly_one_anchor() -> None:
    identity, evidence, control, receipts = pilot.build_counterfactual_plans(
        _probe(), _workload(), topk=2, trace_id="test"
    )

    assert [row["native_contains_evidence"] for row in receipts] == [False, True]
    assert [row["identity_arm"] for row in receipts] == ["matched-control", "force-evidence"]
    assert len(identity.selections) == len(evidence.selections) == len(control.selections) == 2
    for identity_row, evidence_row, control_row, receipt in zip(
        identity.selections, evidence.selections, control.selections, receipts, strict=True
    ):
        evidence_ends = set(evidence_row.block_end_positions)
        control_ends = set(control_row.block_end_positions)
        assert len(evidence_ends) == len(control_ends) == 2
        assert evidence_ends - control_ends == {receipt["evidence_block_end"]}
        assert control_ends - evidence_ends == {receipt["control_block_end"]}
        expected_identity = evidence_ends if receipt["native_contains_evidence"] else control_ends
        assert set(identity_row.block_end_positions) == expected_identity


def test_coordinate_seed_is_deterministic_and_namespaced() -> None:
    coordinate = {
        "scale": "s55",
        "training_seed": 6071406,
        "budget_multiplier": 1,
        "context": 512,
        "family": "single-remote-retrieval",
        "replicate": 0,
    }

    assert pilot._evaluation_seed(coordinate, "a") == pilot._evaluation_seed(coordinate, "a")
    assert pilot._evaluation_seed(coordinate, "a") != pilot._evaluation_seed(coordinate, "b")
