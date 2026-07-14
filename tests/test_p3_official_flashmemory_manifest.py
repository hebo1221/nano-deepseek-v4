from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "research/adaptive_v4_memory/manifests/p3-flashmemory-deepseek-v4-v1.json"
    )
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def test_official_flashmemory_manifest_cannot_be_misread_as_executed_evidence() -> None:
    manifest = _manifest()

    assert manifest["status"] == "blocked_before_execution"
    assert "not_an_executed_result" in manifest["evidence_tier"]
    assert manifest["decision"].startswith("Official FlashMemory-DeepSeek-V4 remains unexecuted")
    assert {blocker["id"] for blocker in manifest["blockers"]} >= {
        "checkpoint-runtime-contract",
        "local-memory-capacity",
        "accelerator-topology",
    }


def test_public_checkpoint_mismatch_and_mode_boundaries_are_explicit() -> None:
    manifest = _manifest()
    audit = manifest["public_release_contract_audit"]
    modes = manifest["runtime_modes"]

    assert audit["serving_loader"] == "torch.load(..., weights_only=True)"
    assert audit["published_checkpoint_format"] == "safetensors"
    assert audit["serving_architecture_constants"] != audit["published_model_card_architecture"]
    assert audit["pt_checkpoint_present_in_published_hf_snapshot"] is False
    assert audit["documented_safetensors_to_serving_conversion_present"] is False
    assert modes["mode_a_score_masking"]["full_kv_remains_on_gpu"] is True
    assert modes["mode_b_pd_disaggregated"]["total_accelerator_slots"] == 16


def test_cost_proxy_arithmetic_is_reproducible() -> None:
    cost = _manifest()["cost_proxy"]
    mode_a = cost["mode_a"]
    mode_b = cost["mode_b"]

    assert mode_a["estimated_usd_per_hour"] == (
        mode_a["accelerator_slots"]
        * cost["h100_sxm_price_per_gpu_hour"]["four_gpu_instance"]
    )
    assert mode_b["estimated_usd_per_hour"] == (
        mode_b["accelerator_slots"]
        * cost["h100_sxm_price_per_gpu_hour"]["eight_gpu_instance"]
    )
    assert mode_b["estimated_usd_for_24_hours"] == (
        24 * mode_b["estimated_usd_per_hour"]
    )
