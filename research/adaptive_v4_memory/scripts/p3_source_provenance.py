from __future__ import annotations

import hashlib
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

KVPRESS_REVISION = "6d965557a5b9f0201a2301b23c454473dd681d0d"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _verify_kvpress_checkout(root_text: str) -> Path:
    root = Path(root_text).resolve()
    _require(root.is_dir(), f"Missing runtime KVPress checkout: {root}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        revision == KVPRESS_REVISION and not dirty,
        "Runtime KVPress checkout revision or cleanliness drifted.",
    )
    return root


def verify_runtime_kvpress_binding(binding: Any) -> dict[str, str]:
    """Verify that recorded evaluator and press imports came from the pinned checkout."""

    _require(isinstance(binding, dict), "Missing runtime KVPress import binding.")
    root = _verify_kvpress_checkout(str(binding.get("checkout_root", "")))
    module = Path(binding.get("module_path", "")).resolve()
    registry = Path(binding.get("registry_path", "")).resolve()
    _require(
        module == root / "kvpress/__init__.py"
        and registry == root / "evaluation/evaluate_registry.py"
        and module.is_file()
        and registry.is_file()
        and binding.get("module_sha256") == _sha256(module)
        and binding.get("registry_sha256") == _sha256(registry),
        "Runtime KVPress import binding drifted.",
    )
    return {
        "checkout_root": str(root),
        "module_sha256": str(binding["module_sha256"]),
        "registry_sha256": str(binding["registry_sha256"]),
    }
