from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import build_p5_paper_package as package  # noqa: E402
import run_p5_release_gate as release  # noqa: E402


def _result(name: str, *, passed: bool = True) -> dict[str, object]:
    return {"name": name, "passed": passed}


def test_release_gate_commands_match_the_frozen_p5_contract() -> None:
    assert [check.display_command for check in release.RELEASE_CHECKS] == (
        package.FINAL_RELEASE_COMMANDS
    )


def test_release_gate_separates_local_success_from_disabled_ci() -> None:
    checks = [_result(check.name) for check in release.RELEASE_CHECKS]

    payload = release.build_payload(
        commit="a" * 40,
        clean_before=True,
        clean_after=True,
        checks=checks,
    )

    assert payload["audit"]["all_local_checks_passed"] is True
    assert payload["audit"]["github_actions"] == {
        "status": "disabled_by_user",
        "passed": False,
        "required_for_completion": False,
    }
    assert "only the five local release checks" in payload["claim_boundary"]
    assert "outside the completion gate" in payload["claim_boundary"]


def test_release_gate_fails_closed_on_partial_failure_or_dirty_source() -> None:
    partial = [_result("ruff"), _result("mypy", passed=False)]
    failed = release.build_payload(
        commit="b" * 40,
        clean_before=True,
        clean_after=True,
        checks=partial,
    )
    dirty = release.build_payload(
        commit="b" * 40,
        clean_before=True,
        clean_after=False,
        checks=[_result(check.name) for check in release.RELEASE_CHECKS],
    )

    assert failed["audit"]["all_local_checks_passed"] is False
    assert failed["audit"]["failed_checks"] == ["mypy"]
    assert dirty["audit"]["all_local_checks_passed"] is False
    assert dirty["source"]["dirty"] is True
