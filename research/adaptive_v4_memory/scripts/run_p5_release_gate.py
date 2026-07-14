from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    display_command: str
    argv: tuple[str, ...]


RELEASE_CHECKS = (
    ReleaseCheck(
        "ruff",
        ".venv/bin/ruff check nano_deepseek_v4 research/adaptive_v4_memory/scripts tests",
        (
            ".venv/bin/ruff",
            "check",
            "nano_deepseek_v4",
            "research/adaptive_v4_memory/scripts",
            "tests",
        ),
    ),
    ReleaseCheck(
        "mypy",
        ".venv/bin/mypy nano_deepseek_v4 research/adaptive_v4_memory/scripts",
        (
            ".venv/bin/mypy",
            "nano_deepseek_v4",
            "research/adaptive_v4_memory/scripts",
        ),
    ),
    ReleaseCheck("pytest", ".venv/bin/pytest -q", (".venv/bin/pytest", "-q")),
    ReleaseCheck(
        "build",
        ".venv/bin/python -m build",
        (".venv/bin/python", "-m", "build"),
    ),
    ReleaseCheck(
        "twine",
        ".venv/bin/twine check dist/*",
        (".venv/bin/twine", "check", "dist/*"),
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True
    ).stdout.strip()


def remote_sync_state() -> dict[str, Any]:
    try:
        upstream = _git(
            "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        )
        counts = _git("rev-list", "--left-right", "--count", f"HEAD...{upstream}")
        ahead_text, behind_text = counts.split()
        ahead = int(ahead_text)
        behind = int(behind_text)
        error = None
    except Exception as caught:
        upstream = None
        ahead = None
        behind = None
        error = {"type": type(caught).__name__, "message": str(caught)}
    return {
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "verified": (
            isinstance(upstream, str)
            and upstream.startswith("origin/")
            and ahead == 0
            and behind == 0
        ),
        "verification_scope": (
            "local origin tracking ref only; does not fetch, prove network freshness, "
            "open a PR, or establish CI status"
        ),
        "error": error,
    }


def _resolve_argv(check: ReleaseCheck) -> tuple[str, ...]:
    if check.name != "twine":
        return check.argv
    distributions = tuple(
        str(path) for path in sorted(Path("dist").glob("*")) if path.is_file()
    )
    if not distributions:
        raise RuntimeError("twine release gate requires at least one dist artifact.")
    return (*check.argv[:-1], *distributions)


def run_check(check: ReleaseCheck) -> dict[str, Any]:
    started = time.monotonic()
    try:
        argv = _resolve_argv(check)
        process = subprocess.run(argv, check=False, capture_output=True)
        return_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
        error = None
    except Exception as caught:
        argv = check.argv
        return_code = 127
        stdout = b""
        stderr = str(caught).encode()
        error = {"type": type(caught).__name__, "message": str(caught)}
    return {
        "name": check.name,
        "command": check.display_command,
        "argv": list(argv),
        "return_code": return_code,
        "passed": return_code == 0,
        "duration_seconds": time.monotonic() - started,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha256_bytes(stdout),
        "stdout_tail": stdout.decode(errors="replace")[-2000:],
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha256_bytes(stderr),
        "stderr_tail": stderr.decode(errors="replace")[-2000:],
        "error": error,
    }


def build_payload(
    *,
    commit: str,
    clean_before: bool,
    clean_after: bool,
    checks: list[dict[str, Any]],
    remote_sync: dict[str, Any],
) -> dict[str, Any]:
    all_local_checks_passed = (
        clean_before
        and clean_after
        and len(checks) == len(RELEASE_CHECKS)
        and all(check.get("passed") is True for check in checks)
        and remote_sync.get("verified") is True
    )
    return {
        "schema_version": 1,
        "experiment_id": "adaptive-v4-memory-p5-local-release-gate-v1",
        "source": {"commit": commit, "dirty": not clean_after},
        "audit": {
            "registered_checks": len(RELEASE_CHECKS),
            "executed_checks": len(checks),
            "source_clean_before": clean_before,
            "source_clean_after": clean_after,
            "all_local_checks_passed": all_local_checks_passed,
            "source_remote_sync": remote_sync,
            "failed_checks": [check["name"] for check in checks if not check.get("passed")],
            "github_actions": {
                "status": "disabled_by_user",
                "passed": False,
                "required_for_completion": False,
            },
        },
        "checks": checks,
        "environment": {"python": platform.python_version()},
        "claim_boundary": (
            "This artifact proves the five local release checks at one clean source commit "
            "and equality with its local origin tracking ref. It does not fetch or prove "
            "remote freshness, PR state, or CI status. GitHub Actions remains disabled by "
            "user request, is outside the completion gate, and is not classified as passed."
        ),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and record the final local P5 gates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p5/local-release-gate.summary.json"
        ),
    )
    args = parser.parse_args()
    commit = _git("rev-parse", "HEAD")
    clean_before = not bool(_git("status", "--porcelain"))
    if not clean_before:
        raise RuntimeError("The P5 local release gate requires a clean source tree.")
    checks: list[dict[str, Any]] = []
    for check in RELEASE_CHECKS:
        result = run_check(check)
        checks.append(result)
        if not result["passed"]:
            break
    clean_after = not bool(_git("status", "--porcelain"))
    remote_sync = remote_sync_state()
    payload = build_payload(
        commit=commit,
        clean_before=clean_before,
        clean_after=clean_after,
        checks=checks,
        remote_sync=remote_sync,
    )
    _write(args.output, payload)
    print(json.dumps(payload["audit"], sort_keys=True))
    if not payload["audit"]["all_local_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
