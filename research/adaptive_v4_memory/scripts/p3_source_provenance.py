from __future__ import annotations

import hashlib
import subprocess
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_git_implementation(source: Any, *, expected_path: str, label: str) -> dict[str, str]:
    _require(isinstance(source, dict), f"Missing {label} source provenance.")
    commit = source.get("commit")
    _require(
        isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit),
        f"Invalid {label} source commit.",
    )
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    )
    _require(commit_check.returncode == 0, f"Unknown {label} source commit: {commit}")
    blob = subprocess.run(
        ["git", "show", f"{commit}:{expected_path}"],
        capture_output=True,
    )
    _require(blob.returncode == 0, f"Missing {label} implementation at source commit.")
    observed_digest = hashlib.sha256(blob.stdout).hexdigest()
    _require(
        source.get("implementation_sha256") == observed_digest,
        f"{label} implementation does not match its source commit.",
    )
    return {
        "commit": commit,
        "implementation_path": expected_path,
        "implementation_sha256": observed_digest,
    }
