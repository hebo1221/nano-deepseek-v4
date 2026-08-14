from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.cuda_release_gate import (
    _PYTEST_ISOLATION_ARGUMENTS,
    COLLECTION_COMMAND,
    LOGICAL_COMMAND,
    PASS_STATUS,
    RECEIPT_KIND,
    SCHEMA_VERSION,
    CudaReleaseGateError,
    _run_command,
    build_source_inventory,
    generate_receipt,
    source_aggregate_sha256,
    verify_receipt,
)

NODE_IDS = [
    "tests/test_accelerator.py::test_cuda_cache",
    "tests/test_accelerator.py::test_cuda_forward[dtype0]",
]


def _write_source(root: Path) -> None:
    (root / "nano_deepseek_v4" / "nested").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts").mkdir()
    (root / "nano_deepseek_v4" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "nano_deepseek_v4" / "nested" / "module.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    (root / "nano_deepseek_v4" / "py.typed").write_bytes(b"")
    (root / "nano_deepseek_v4" / "ignored.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (root / "tests" / "test_accelerator.py").write_text(
        "def test_fixture():\n    pass\n", encoding="utf-8"
    )
    real_script = Path(__file__).resolve().parents[1] / "scripts" / "cuda_release_gate.py"
    (root / "scripts" / "cuda_release_gate.py").write_bytes(real_script.read_bytes())


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _write_source(root)
    return root


def _environment() -> dict[str, object]:
    uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    return {
        "python": {"implementation": "CPython", "version": "3.11.13"},
        "torch": {"version": "2.13.0+cu130"},
        "cuda": {"available": True, "build_version": "13.0", "cudnn_version": 92000},
        "device": {
            "index": 0,
            "name": "NVIDIA GB10",
            "uuid": uuid,
            "compute_capability": [12, 1],
            "total_memory_bytes": 128 * 1024**3,
            "multiprocessor_count": 48,
        },
        "driver": {"device_uuid": uuid, "version": "580.126.09"},
    }


def _valid_receipt(root: Path, *, clean: bool = True) -> dict[str, Any]:
    inventory = build_source_inventory(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": PASS_STATUS,
        "source": {
            "git_commit": "a" * 40,
            "worktree_clean": clean,
            "inventory": inventory,
            "aggregate_sha256": source_aggregate_sha256(inventory),
        },
        "tests": {
            "collection_command": list(COLLECTION_COMMAND),
            "logical_command": list(LOGICAL_COMMAND),
            "execution_command": [
                "python",
                "-m",
                "pytest",
                "-q",
                *_PYTEST_ISOLATION_ARGUMENTS,
                *NODE_IDS,
            ],
            "selected_node_ids": NODE_IDS,
            "selected_count": len(NODE_IDS),
            "passed_count": len(NODE_IDS),
            "skipped_count": 0,
            "failed_count": 0,
            "error_count": 0,
        },
        "environment": _environment(),
    }


def _write_receipt(path: Path, receipt: object) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _FakeRunner:
    def __init__(
        self,
        *,
        dirty: bool = False,
        outcome: str = "passed",
        mutate_after_tests: Path | None = None,
    ) -> None:
        self.dirty = dirty
        self.outcome = outcome
        self.mutate_after_tests = mutate_after_tests
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: Sequence[str],
        root: Path,
    ) -> subprocess.CompletedProcess[str]:
        args = list(command)
        self.commands.append(args)
        if args[:2] == ["git", "status"]:
            output = " M local-dogfood.txt\n" if self.dirty else ""
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(args, 0, "b" * 40 + "\n", "")
        if args[:2] == ["git", "hash-object"]:
            return subprocess.CompletedProcess(args, 0, "b" * 40 + "\n", "")
        if "--collect-only" in args:
            return subprocess.CompletedProcess(args, 0, "\n".join(reversed(NODE_IDS)) + "\n", "")

        junit_argument = next((value for value in args if value.startswith("--junitxml=")), None)
        if junit_argument is None:
            return subprocess.CompletedProcess(args, 2, "", "unexpected command")
        junit_path = Path(junit_argument.split("=", 1)[1])
        outcomes = {
            "passed": ("<testcase/><testcase/>", 0),
            "skipped": ("<testcase><skipped/></testcase><testcase/>", 0),
            "failed": ("<testcase><failure/></testcase><testcase/>", 1),
        }
        cases, returncode = outcomes[self.outcome]
        junit_path.write_text(
            f'<testsuites><testsuite tests="2">{cases}</testsuite></testsuites>',
            encoding="utf-8",
        )
        if self.mutate_after_tests is not None:
            self.mutate_after_tests.write_text("VALUE = 99\n", encoding="utf-8")
            self.mutate_after_tests = None
        return subprocess.CompletedProcess(args, returncode, "", "")


class _GitBackedFakeRunner(_FakeRunner):
    def __call__(
        self,
        command: Sequence[str],
        root: Path,
    ) -> subprocess.CompletedProcess[str]:
        args = list(command)
        if args and args[0] == "git":
            self.commands.append(args)
            return subprocess.run(
                args,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        return super().__call__(command, root)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_inventory_covers_dynamic_python_surface_and_required_files(source_root: Path):
    inventory = build_source_inventory(source_root)
    paths = [str(entry["path"]) for entry in inventory]

    assert paths == sorted(paths)
    assert paths == [
        "nano_deepseek_v4/__init__.py",
        "nano_deepseek_v4/nested/module.py",
        "nano_deepseek_v4/py.typed",
        "pyproject.toml",
        "scripts/cuda_release_gate.py",
        "tests/test_accelerator.py",
    ]
    assert all(len(str(entry["sha256"])) == 64 for entry in inventory)
    assert source_aggregate_sha256(inventory) == source_aggregate_sha256(
        build_source_inventory(source_root)
    )


def test_command_runner_ignores_ambient_pytest_selection_and_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nmarkers = ['gpu: fixture accelerator test']\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = -k nonexistent\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        "def pytest_collection_modifyitems(items):\n    items.clear()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_accelerator.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.gpu\n"
        "def test_first():\n    pass\n\n"
        "@pytest.mark.gpu\n"
        "def test_second():\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k cache_can_round_trip")
    monkeypatch.setenv("PYTEST_PLUGINS", "plugin_that_must_not_be_loaded")

    result = _run_command(
        (sys.executable, *COLLECTION_COMMAND[1:], "-p", "no:cacheprovider"),
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    node_ids = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("tests/test_accelerator.py::")
    ]
    assert node_ids == [
        "tests/test_accelerator.py::test_first",
        "tests/test_accelerator.py::test_second",
    ]


def test_verify_accepts_exact_receipt_without_importing_torch(source_root: Path, tmp_path: Path):
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(receipt_path, _valid_receipt(source_root))

    result = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(__file__).resolve().parents[1] / "scripts" / "cuda_release_gate.py"),
            "verify",
            "--receipt",
            str(receipt_path),
            "--root",
            str(source_root),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "verified CUDA release-gate receipt" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value.update(kind="other"), "kind drift"),
        (lambda value: value.update(status="incomplete"), "status"),
        (lambda value: value.update(unexpected=True), "schema drift"),
        (lambda value: value["source"].update(git_commit="bad"), "git_commit"),
        (lambda value: value["source"].update(worktree_clean=False), "dirty-generated"),
        (
            lambda value: value["source"].update(aggregate_sha256="0" * 64),
            "aggregate",
        ),
        (
            lambda value: value["tests"].update(logical_command=["pytest"]),
            "logical test command",
        ),
        (
            lambda value: value["tests"].update(collection_command=["pytest"]),
            "collection command",
        ),
        (
            lambda value: value["tests"].update(execution_command=["pytest"]),
            "execution command",
        ),
        (lambda value: value["tests"].update(selected_node_ids=[]), "node IDs"),
        (lambda value: value["tests"].update(selected_count=1), "counts"),
        (lambda value: value["tests"].update(passed_count=1), "counts"),
        (lambda value: value["tests"].update(skipped_count=1), "skips"),
        (lambda value: value["tests"].update(failed_count=1), "failures"),
        (lambda value: value["tests"].update(error_count=1), "errors"),
        (lambda value: value["environment"].pop("cuda"), "schema drift"),
        (
            lambda value: value["environment"]["cuda"].update(available=False),
            "available",
        ),
        (lambda value: value["environment"].pop("device"), "schema drift"),
        (
            lambda value: value["environment"]["device"].update(uuid="missing"),
            "uuid",
        ),
        (
            lambda value: value["environment"]["driver"].update(
                device_uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"
            ),
            "do not match",
        ),
    ],
)
def test_verify_rejects_receipt_drift(source_root: Path, tmp_path: Path, mutation, message: str):
    receipt = _valid_receipt(source_root)
    mutation(receipt)
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(receipt_path, receipt)

    with pytest.raises(CudaReleaseGateError, match=message):
        verify_receipt(receipt_path, root=source_root)


def test_verify_rejects_inventory_content_and_membership_drift(source_root: Path, tmp_path: Path):
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(receipt_path, _valid_receipt(source_root))
    (source_root / "nano_deepseek_v4" / "__init__.py").write_text("VALUE = 3\n", encoding="utf-8")

    with pytest.raises(CudaReleaseGateError, match="inventory drifted"):
        verify_receipt(receipt_path, root=source_root)

    _write_receipt(receipt_path, _valid_receipt(source_root))
    (source_root / "nano_deepseek_v4" / "new_module.py").write_text("VALUE = 4\n", encoding="utf-8")
    with pytest.raises(CudaReleaseGateError, match="inventory drifted"):
        verify_receipt(receipt_path, root=source_root)


def test_verify_rejects_receipt_inventory_hash_tampering(source_root: Path, tmp_path: Path):
    receipt = _valid_receipt(source_root)
    tampered = copy.deepcopy(receipt)
    tampered["source"]["inventory"][0]["sha256"] = "f" * 64
    tampered["source"]["aggregate_sha256"] = source_aggregate_sha256(
        tampered["source"]["inventory"]
    )
    receipt_path = tmp_path / "receipt.json"
    _write_receipt(receipt_path, tampered)

    with pytest.raises(CudaReleaseGateError, match="inventory drifted"):
        verify_receipt(receipt_path, root=source_root)


def test_verify_rejects_duplicate_json_keys(source_root: Path, tmp_path: Path):
    receipt = json.dumps(_valid_receipt(source_root), sort_keys=True)
    duplicate = receipt.replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(CudaReleaseGateError, match="duplicate JSON key"):
        verify_receipt(receipt_path, root=source_root)


def test_generate_clean_receipt_is_deterministic_and_verifiable(source_root: Path, tmp_path: Path):
    runner = _FakeRunner()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    generated = generate_receipt(
        first,
        root=source_root,
        runner=runner,
        environment_probe=_environment,
    )
    generate_receipt(
        second,
        root=source_root,
        runner=runner,
        environment_probe=_environment,
    )

    assert first.read_bytes() == second.read_bytes()
    assert generated == verify_receipt(first, root=source_root)
    collection = next(command for command in runner.commands if "--collect-only" in command)
    assert collection[1 : len(COLLECTION_COMMAND)] == list(COLLECTION_COMMAND[1:])
    execution = next(command for command in runner.commands if "--junitxml" in " ".join(command))
    expected_prefix = ["-m", "pytest", "-q", *_PYTEST_ISOLATION_ARGUMENTS]
    assert execution[1 : 1 + len(expected_prefix)] == expected_prefix
    node_start = 1 + len(expected_prefix)
    assert execution[node_start : node_start + len(NODE_IDS)] == NODE_IDS


def test_generate_dirty_requires_opt_in_and_writes_non_verifiable_receipt(
    source_root: Path,
    tmp_path: Path,
):
    runner = _FakeRunner(dirty=True)
    receipt_path = tmp_path / "dirty.json"

    with pytest.raises(CudaReleaseGateError, match="worktree and covered source bytes"):
        generate_receipt(
            receipt_path,
            root=source_root,
            runner=runner,
            environment_probe=_environment,
        )
    assert not receipt_path.exists()

    generated = generate_receipt(
        receipt_path,
        root=source_root,
        allow_dirty=True,
        runner=runner,
        environment_probe=_environment,
    )

    generated_payload = cast(dict[str, Any], generated)
    assert generated_payload["source"]["worktree_clean"] is False
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["source"]["worktree_clean"] is False
    with pytest.raises(CudaReleaseGateError, match="dirty-generated"):
        verify_receipt(receipt_path, root=source_root)


def test_generate_does_not_treat_hidden_covered_changes_as_clean(
    source_root: Path,
    tmp_path: Path,
):
    _git(source_root, "init", "-q")
    _git(source_root, "config", "user.name", "CUDA gate fixture")
    _git(source_root, "config", "user.email", "fixture@example.invalid")
    _git(source_root, "add", "--all")
    _git(source_root, "commit", "-q", "-m", "fixture")
    covered = source_root / "nano_deepseek_v4" / "__init__.py"
    covered_relative = covered.relative_to(source_root).as_posix()
    _git(source_root, "update-index", "--assume-unchanged", covered_relative)
    covered.write_text("VALUE = 99\n", encoding="utf-8")

    assert _git(source_root, "status", "--porcelain=v1") == ""
    receipt_path = tmp_path / "hidden-change.json"
    runner = _GitBackedFakeRunner()
    with pytest.raises(CudaReleaseGateError, match="not byte-identical at HEAD"):
        generate_receipt(
            receipt_path,
            root=source_root,
            runner=runner,
            environment_probe=_environment,
        )

    generated = generate_receipt(
        receipt_path,
        root=source_root,
        allow_dirty=True,
        runner=_GitBackedFakeRunner(),
        environment_probe=_environment,
    )

    assert cast(dict[str, Any], generated["source"])["worktree_clean"] is False
    with pytest.raises(CudaReleaseGateError, match="dirty-generated"):
        verify_receipt(receipt_path, root=source_root)


@pytest.mark.parametrize("outcome", ["skipped", "failed"])
def test_generate_requires_every_selected_test_to_pass(
    source_root: Path,
    tmp_path: Path,
    outcome: str,
):
    with pytest.raises(CudaReleaseGateError, match="all-pass|pytest run failed"):
        generate_receipt(
            tmp_path / "receipt.json",
            root=source_root,
            runner=_FakeRunner(outcome=outcome),
            environment_probe=_environment,
        )


def test_generate_rejects_source_changes_during_gpu_run(source_root: Path, tmp_path: Path):
    mutated = source_root / "nano_deepseek_v4" / "__init__.py"
    with pytest.raises(CudaReleaseGateError, match="inventory changed"):
        generate_receipt(
            tmp_path / "receipt.json",
            root=source_root,
            runner=_FakeRunner(mutate_after_tests=mutated),
            environment_probe=_environment,
        )


def test_generate_requires_cuda_probe(source_root: Path, tmp_path: Path):
    def unavailable() -> dict[str, object]:
        raise CudaReleaseGateError("CUDA is required")

    with pytest.raises(CudaReleaseGateError, match="CUDA is required"):
        generate_receipt(
            tmp_path / "receipt.json",
            root=source_root,
            runner=_FakeRunner(),
            environment_probe=unavailable,
        )
