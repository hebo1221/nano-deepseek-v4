from __future__ import annotations

import base64
import importlib.machinery
import json
import os
import py_compile
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_git_launcher_v1_3 as launcher  # noqa: E402


def test_launcher_uses_only_the_v1_3_5_operational_namespace() -> None:
    assert launcher.LAUNCHER_ID == "p2-direct-controller-git-object-launcher-v1-3-5"
    assert launcher.EXPECTED_EXPERIMENT_ID.endswith("v1.3.5")
    assert launcher.MANIFEST_RELATIVE_PATH.endswith("v1-3-5.json")
    assert launcher.EXPECTED_MANIFEST_STATUS == "frozen_v1_3_5_mixed_device_block_site"
    assert "V1_3_5" in launcher.RUNNER_FD_ENV
    assert "V1_3_5" in launcher.SOURCE_BUNDLE_FD_ENV
    assert "V1_3_5" in launcher.LAUNCH_ROUTING_FD_ENV
    assert launcher.SEALED_LAUNCH_SENTINEL_NAME.endswith("V1_3_5")
    assert launcher.PYTHON_RUNTIME_BINDING_NAME.endswith("V1_3_5")
    assert launcher.SOURCE_PROVENANCE_BINDING_NAME.endswith("V1_3_5")
    assert launcher.LAUNCH_ROUTING_BINDING_NAME.endswith("V1_3_5")
    assert "V1_3_1" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    assert "v1-3-1" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    assert "V1_3_2" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    assert "v1-3-2" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    assert "V1_3_3" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    assert "v1-3-3" not in launcher.RUNNER_BOOTSTRAP_SOURCE
    operational_values = (
        launcher.LAUNCHER_ID,
        launcher.EXPECTED_EXPERIMENT_ID,
        launcher.MANIFEST_RELATIVE_PATH,
        launcher.RUNNER_FD_ENV,
        launcher.SOURCE_BUNDLE_FD_ENV,
        launcher.LAUNCH_ROUTING_FD_ENV,
        launcher.SEALED_LAUNCH_SENTINEL_NAME,
        launcher.PYTHON_RUNTIME_BINDING_NAME,
        launcher.SOURCE_PROVENANCE_BINDING_NAME,
        launcher.LAUNCH_ROUTING_BINDING_NAME,
    )
    assert all("v1-3-3" not in value and "V1_3_3" not in value for value in operational_values)


def _git(root: Path, arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "XDG_CONFIG_HOME": "/nonexistent",
        },
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, ["add", "-A"])
    _git(
        root,
        [
            "-c",
            "user.name=Adaptive V4 Test",
            "-c",
            "user.email=adaptive-v4-test@example.invalid",
            "commit",
            "-m",
            message,
        ],
    )
    return _git(root, ["rev-parse", "HEAD"])


def _write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _runtime_site_packages(root: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return root / ".venv" / "lib" / version / "site-packages"


def _prepare_runtime(root: Path) -> Path:
    executable = root / launcher.PYTHON_RELATIVE_PATH
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.symlink_to(Path("/proc/self/exe").resolve(strict=True))
    site_packages = _runtime_site_packages(root)
    site_packages.mkdir(parents=True)
    _write(
        site_packages / "fixture_external_dependency.py",
        'VALUE = "safe venv dependency"\n',
    )
    return site_packages


def _frozen_repository(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    root = (tmp_path / "repository").resolve()
    root.mkdir()
    _git(root, ["init", "-q"])
    _prepare_runtime(root)
    launcher_path = launcher.LAUNCHER_RELATIVE_PATH
    runner_path = launcher.RUNNER_RELATIVE_PATH
    helper_path = "research/adaptive_v4_memory/scripts/helper.py"
    manifest_path = launcher.MANIFEST_RELATIVE_PATH
    safe_runner = f"""if globals().get({launcher.SEALED_LAUNCH_SENTINEL_NAME!r}) != {launcher.SEALED_LAUNCH_SENTINEL!r}:
    raise RuntimeError("sealed launcher sentinel missing")
runtime = globals().get({launcher.PYTHON_RUNTIME_BINDING_NAME!r})
if not isinstance(runtime, dict) or runtime.get("schema_version") != 1:
    raise RuntimeError("sealed Python runtime binding missing")
provenance = globals().get({launcher.SOURCE_PROVENANCE_BINDING_NAME!r})
if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
    raise RuntimeError("sealed Git source provenance missing")
routing = globals().get({launcher.LAUNCH_ROUTING_BINDING_NAME!r})
if not isinstance(routing, dict) or routing.get("schema_version") != 1:
    raise RuntimeError("sealed Git launch routing missing")
import base64
from pathlib import Path
import fixture_external_dependency
import helper
import sys

if helper.VALUE != "safe frozen helper" or fixture_external_dependency.VALUE != "safe venv dependency":
    raise RuntimeError("frozen helper source drifted")
if Path(__file__) != Path(provenance["repository_root"]) / routing["entrypoint_relative_path"]:
    raise RuntimeError("sealed Git launch routing path drifted")
manifest_binding = provenance["head_manifest"]
live_manifest = Path(provenance["repository_root"]) / manifest_binding["path"]
if live_manifest.read_bytes() != base64.b64decode(
    manifest_binding["source_base64"], validate=True
):
    raise RuntimeError("live manifest differs from sealed HEAD manifest")
Path(sys.argv[1]).write_text("safe", encoding="utf-8")
"""
    _write(root / launcher_path, Path(launcher.__file__).read_bytes())
    for selector, entrypoint_path in launcher.ENTRYPOINT_RELATIVE_PATHS.items():
        _write(root / entrypoint_path, f"# frozen fixture entrypoint: {selector}\n{safe_runner}")
    _write(root / helper_path, 'VALUE = "safe frozen helper"\n')
    source_commit = _commit(root, "freeze implementation")
    paths = (
        launcher_path,
        *launcher.ENTRYPOINT_RELATIVE_PATHS.values(),
        helper_path,
    )
    rows = launcher._tree_rows(root, source_commit, paths)
    tree_digest = launcher._implementation_index_digest(paths, rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": launcher.EXPECTED_EXPERIMENT_ID,
        "status": launcher.EXPECTED_MANIFEST_STATUS,
        "implementation": {
            "paths": list(paths),
            "tree_digest": tree_digest,
            "source_commit": source_commit,
        },
    }
    _write(
        root / manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    head = _commit(root, "freeze manifest")
    return root, head, manifest_path, launcher_path, runner_path


def _fixture_bundle(
    root: Path,
    head: str,
    manifest_path: str,
    launcher_path: str,
    runner_path: str,
) -> tuple[dict[str, Any], bytes]:
    return launcher.build_frozen_source_bundle(
        root,
        head,
        manifest_relative_path=manifest_path,
        launcher_relative_path=launcher_path,
        runner_relative_path=runner_path,
        expected_experiment_id=launcher.EXPECTED_EXPERIMENT_ID,
        expected_manifest_status=launcher.EXPECTED_MANIFEST_STATUS,
    )


def _isolated_bootstrap(
    bundle: dict[str, Any],
    runner_source: bytes,
    arguments: Sequence[str],
    *,
    entrypoint: str = "matrix",
    routing_override: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    routing = (
        launcher._launch_routing_binding(bundle, runner_source, entrypoint)
        if routing_override is None
        else routing_override
    )
    runner_fd = launcher._create_sealed_memfd("launcher-test-runner", runner_source)
    bundle_fd = launcher._create_sealed_memfd(
        "launcher-test-bundle", launcher._canonical_json(bundle)
    )
    routing_fd = launcher._create_sealed_memfd(
        "launcher-test-routing", launcher._canonical_json(routing)
    )
    try:
        environment = launcher._python_environment()
        environment[launcher.RUNNER_FD_ENV] = str(runner_fd)
        environment[launcher.SOURCE_BUNDLE_FD_ENV] = str(bundle_fd)
        environment[launcher.LAUNCH_ROUTING_FD_ENV] = str(routing_fd)
        runtime = bundle["python_runtime"]
        return subprocess.run(
            [
                runtime["executable"],
                "-I",
                "-S",
                "-B",
                "-c",
                launcher.RUNNER_BOOTSTRAP_SOURCE,
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(str(bundle["repository_root"])) if cwd is None else cwd,
            env=environment,
            pass_fds=(runner_fd, bundle_fd, routing_fd),
        )
    finally:
        os.close(routing_fd)
        os.close(bundle_fd)
        os.close(runner_fd)


def test_git_object_launch_ignores_assume_unchanged_worktree_tampering(
    tmp_path: Path,
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    assert launcher.pin_starting_head(root, head) == head
    safe_marker = tmp_path / "safe-marker"
    rogue_marker = tmp_path / "rogue-marker"
    _write(
        root / runner_path,
        f"from pathlib import Path\nPath({str(rogue_marker)!r}).write_text('rogue')\n",
    )
    helper_path = str(Path(runner_path).with_name("helper.py"))
    _write(root / helper_path, 'raise RuntimeError("tampered helper executed")\n')
    _git(
        root,
        [
            "update-index",
            "--assume-unchanged",
            runner_path,
            helper_path,
        ],
    )

    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    assert bundle["pinned_head_oid"] == head
    assert bundle["frozen_source_commit"] != head
    assert launcher.launch_frozen_runner(bundle, runner_source, [str(safe_marker)]) == 0
    assert safe_marker.read_text(encoding="utf-8") == "safe"
    assert not rogue_marker.exists()

    _write(root / manifest_path, b'{"tampered":true}\n')
    _git(root, ["update-index", "--assume-unchanged", manifest_path])
    rejected_marker = tmp_path / "manifest-tamper-must-not-run"
    rejected_bundle, rejected_runner = _fixture_bundle(
        root, head, manifest_path, launcher_path, runner_path
    )
    assert (
        launcher.launch_frozen_runner(rejected_bundle, rejected_runner, [str(rejected_marker)]) != 0
    )
    assert not rejected_marker.exists()

    direct_globals = {"__name__": "__main__", "__file__": str(root / runner_path)}
    with pytest.raises(RuntimeError, match="sentinel missing"):
        exec(compile(runner_source, str(root / runner_path), "exec"), direct_globals)


def test_executable_venv_pth_is_never_processed_before_frozen_runner(
    tmp_path: Path,
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    pth_marker = tmp_path / "executable-pth-ran"
    _write(
        _runtime_site_packages(root) / "rogue_startup.pth",
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('rogue')\n",
    )
    safe_marker = tmp_path / "safe-with-inert-pth"

    assert launcher.launch_frozen_runner(bundle, runner_source, [str(safe_marker)]) == 0
    assert safe_marker.read_text(encoding="utf-8") == "safe"
    assert not pth_marker.exists()


def test_deleted_frozen_helper_cannot_fall_through_to_venv_shadow(
    tmp_path: Path,
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    (root / Path(runner_path).with_name("helper.py")).unlink()
    rogue_marker = tmp_path / "venv-helper-shadow-ran"
    _write(
        _runtime_site_packages(root) / "helper.py",
        (
            "from pathlib import Path\n"
            f"Path({str(rogue_marker)!r}).write_text('rogue')\n"
            'VALUE = "malicious venv helper"\n'
        ),
    )
    safe_marker = tmp_path / "safe-sealed-helper"

    assert launcher.launch_frozen_runner(bundle, runner_source, [str(safe_marker)]) == 0
    assert safe_marker.read_text(encoding="utf-8") == "safe"
    assert not rogue_marker.exists()


def test_bootstrap_rejects_python_runtime_binding_drift(tmp_path: Path) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    runtime = launcher._python_runtime_binding(root)
    assert bundle["python_runtime"] == runtime
    assert runtime["executable"] == str(Path("/proc/self/exe").resolve(strict=True))
    assert runtime["venv_executable"] == str(root / launcher.PYTHON_RELATIVE_PATH)
    assert runtime["site_packages"] == str(_runtime_site_packages(root))

    tampered = json.loads(json.dumps(bundle))
    tampered["python_runtime"]["executable_sha256"] = "0" * 64
    unsigned = dict(tampered)
    unsigned.pop("bundle_sha256")
    tampered["bundle_sha256"] = launcher._json_digest(unsigned)
    result = _isolated_bootstrap(
        tampered, runner_source, [str(tmp_path / "runtime-drift-safe-marker")]
    )
    assert result.returncode != 0
    assert "Frozen Python runtime binding drifted" in result.stderr


def test_bootstrap_rejects_rewritten_head_manifest_source_authority(tmp_path: Path) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    tampered = json.loads(json.dumps(bundle))
    tampered["head_manifest"]["source_base64"] = base64.b64encode(b'{"rewritten":true}\n').decode(
        "ascii"
    )
    unsigned = dict(tampered)
    unsigned.pop("bundle_sha256")
    tampered["bundle_sha256"] = launcher._json_digest(unsigned)

    result = _isolated_bootstrap(
        tampered, runner_source, [str(tmp_path / "manifest-authority-marker")]
    )
    assert result.returncode != 0
    assert "Frozen HEAD manifest source bytes drifted" in result.stderr


def test_bootstrap_rejects_launch_routing_that_differs_from_selected_blob(
    tmp_path: Path,
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    routing = launcher._launch_routing_binding(bundle, runner_source, "matrix")
    routing["git_blob_oid"] = "0" * len(routing["git_blob_oid"])

    result = _isolated_bootstrap(
        bundle,
        runner_source,
        [str(tmp_path / "routing-drift-marker")],
        routing_override=routing,
    )

    assert result.returncode != 0
    assert "Frozen launch routing differs from its selected Git blob" in result.stderr


def test_runner_child_uses_bound_interpreter_and_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    observed: dict[str, Any] = {}

    def fake_run(arguments: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed["arguments"] = list(arguments)
        observed.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setenv("LD_PRELOAD", "/tmp/rogue-launcher-preload.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/rogue-launcher-pythonpath")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)  # type: ignore[attr-defined]

    assert launcher.launch_frozen_runner(bundle, runner_source, ["--fixture"]) == 0
    runtime = bundle["python_runtime"]
    assert observed["arguments"][:4] == [runtime["executable"], "-I", "-S", "-B"]
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert "LD_PRELOAD" not in environment
    assert "PYTHONPATH" not in environment
    assert set(environment) == {
        *launcher._python_environment(),
        launcher.RUNNER_FD_ENV,
        launcher.SOURCE_BUNDLE_FD_ENV,
        launcher.LAUNCH_ROUTING_FD_ENV,
    }
    pass_fds = observed["pass_fds"]
    assert isinstance(pass_fds, tuple)
    assert len(pass_fds) == len(set(pass_fds)) == 3
    assert observed["cwd"] == root


def test_bootstrap_rejects_non_repository_working_directory(tmp_path: Path) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    outside = tmp_path / "outside-working-directory"
    outside.mkdir()
    marker = tmp_path / "wrong-cwd-runner-must-not-execute"

    result = _isolated_bootstrap(
        bundle,
        runner_source,
        [str(marker)],
        cwd=outside,
    )

    assert result.returncode != 0
    assert "working directory differs from the exact repository root" in result.stderr
    assert not marker.exists()


def test_starting_head_is_pinned_across_a_later_ref_move(tmp_path: Path) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    pinned = launcher.pin_starting_head(root, head)
    _write(root / "later.txt", "later commit\n")
    later_head = _commit(root, "move head after pin")
    assert later_head != pinned

    bundle, runner_source = _fixture_bundle(root, pinned, manifest_path, launcher_path, runner_path)
    assert bundle["pinned_head_oid"] == pinned
    assert b"sealed launcher sentinel missing" in runner_source
    with pytest.raises(ValueError, match="Starting HEAD changed"):
        launcher.pin_starting_head(root, pinned)


def test_git_replace_ref_cannot_substitute_the_pinned_launcher_or_tree(tmp_path: Path) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    original_launcher = (root / launcher_path).read_bytes()
    _write(root / launcher_path, b"raise SystemExit('replacement launcher executed')\n")
    replacement_commit = _commit(root, "malicious replacement object")
    _git(root, ["reset", "--hard", head])
    _git(root, ["replace", head, replacement_commit])

    raw_replaced = subprocess.run(
        ["/usr/bin/git", "-C", str(root), "cat-file", "blob", f"{head}:{launcher_path}"],
        check=True,
        capture_output=True,
    ).stdout
    assert raw_replaced != original_launcher

    pinned_bytes = launcher._git(root, ["cat-file", "blob", f"{head}:{launcher_path}"]).stdout
    assert pinned_bytes == original_launcher
    canonical_outer_bytes = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            *(f"{key}={value}" for key, value in launcher._git_environment().items()),
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            "cat-file",
            "blob",
            f"{head}:{launcher_path}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert canonical_outer_bytes == original_launcher
    assert launcher.pin_starting_head(root, head) == head
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    assert bundle["pinned_head_oid"] == head
    assert b"replacement launcher executed" not in runner_source


def test_raw_commit_ancestry_ignores_repository_grafts(tmp_path: Path) -> None:
    root, _head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    source_commit = manifest["implementation"]["source_commit"]
    source_tree = _git(root, ["rev-parse", f"{source_commit}^{{tree}}"])
    unrelated_source = _git(
        root,
        [
            "-c",
            "user.name=Adaptive V4 Test",
            "-c",
            "user.email=adaptive-v4-test@example.invalid",
            "commit-tree",
            source_tree,
            "-m",
            "unrelated frozen source",
        ],
    )
    assert unrelated_source != source_commit
    manifest["implementation"]["source_commit"] = unrelated_source
    _write(
        root / manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    forged_head = _commit(root, "manifest names unrelated source")
    _write(root / ".git" / "info" / "grafts", f"{forged_head} {unrelated_source}\n")

    assert (
        launcher._git(
            root,
            ["merge-base", "--is-ancestor", unrelated_source, forged_head],
            check=False,
        ).returncode
        == 0
    )
    with pytest.raises(ValueError, match="not an ancestor"):
        _fixture_bundle(
            root,
            forged_head,
            manifest_path,
            launcher_path,
            runner_path,
        )


def test_repo_local_venv_is_allowed_but_sourceless_and_extension_shadows_are_blocked(
    tmp_path: Path,
) -> None:
    root, head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    bundle, runner_source = _fixture_bundle(root, head, manifest_path, launcher_path, runner_path)
    safe_marker = tmp_path / "safe-venv-import"
    assert launcher.launch_frozen_runner(bundle, runner_source, [str(safe_marker)]) == 0
    assert safe_marker.read_text(encoding="utf-8") == "safe"

    scripts = root / Path(runner_path).parent
    rogue_marker = tmp_path / "rogue-pyc-executed"
    source_path = scripts / "fixture_external_dependency.py"
    pyc_path = scripts / "fixture_external_dependency.pyc"
    _write(
        source_path,
        f"from pathlib import Path\nPath({str(rogue_marker)!r}).write_text('rogue')\n",
    )
    py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
    source_path.unlink()
    try:
        result = _isolated_bootstrap(bundle, runner_source, [str(tmp_path / "pyc-safe")])
    finally:
        pyc_path.unlink(missing_ok=True)
    assert result.returncode != 0
    assert "Blocked non-frozen repository import before execution" in result.stderr
    assert not rogue_marker.exists()

    extension_path = (
        scripts / f"fixture_external_dependency{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    _write(extension_path, b"not a loadable extension")
    try:
        result = _isolated_bootstrap(bundle, runner_source, [str(tmp_path / "extension-safe")])
    finally:
        extension_path.unlink(missing_ok=True)
    assert result.returncode != 0
    assert "Blocked non-frozen repository import before execution" in result.stderr
    assert "file too short" not in result.stderr.lower()


def test_bundle_rejects_manifest_tree_digest_drift(tmp_path: Path) -> None:
    root, _head, manifest_path, launcher_path, runner_path = _frozen_repository(tmp_path)
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    manifest["implementation"]["tree_digest"] = "f" * 64
    _write(
        root / manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    bad_head = _commit(root, "commit invalid tree digest")
    with pytest.raises(ValueError, match="tree digest"):
        _fixture_bundle(root, bad_head, manifest_path, launcher_path, runner_path)


def test_canonical_command_uses_pinned_git_and_isolated_stdin_launcher(tmp_path: Path) -> None:
    root, head, *_rest = _frozen_repository(tmp_path)
    command = launcher.canonical_command(root, head)
    assert command.startswith(f"set -euo pipefail; head={head}; /usr/bin/env -i")
    assert "GIT_CONFIG_GLOBAL=/dev/null" in command
    assert "GIT_CONFIG_SYSTEM=/dev/null" in command
    assert "GIT_LITERAL_PATHSPECS=1" in command
    assert "GIT_NO_LAZY_FETCH=1" in command
    assert "GIT_NO_REPLACE_OBJECTS=1" in command
    assert "/usr/bin/git -c core.hooksPath=/dev/null -c core.commitGraph=false" in command
    assert " cat-file blob $head:" in command
    assert command.count("/usr/bin/env -i") == 2
    assert " | /usr/bin/env -i " in command
    assert " -I -S -B - " in command
    runtime = launcher._python_runtime_binding(root)
    assert runtime["executable"] in command
    assert launcher._json_digest(runtime) in command
    assert launcher.LAUNCHER_RELATIVE_PATH in command
    assert " matrix --" in command
    assert launcher.validate_pinned_git_executable() == Path("/usr/bin/git")
    assert "LD_PRELOAD" not in launcher._git_environment()
    assert launcher._git_environment()["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert launcher._git_environment()["GIT_NO_LAZY_FETCH"] == "1"
    assert launcher._git_environment()["GIT_LITERAL_PATHSPECS"] == "1"
    with pytest.raises(ValueError, match="pinned Git blob over stdin"):
        launcher.main([str(root), head, "--"])
    missing = launcher.canonical_command(root, "0" * 40)
    failed = subprocess.run(
        ["/bin/bash", "-c", missing],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "entrypoint", tuple(launcher.ENTRYPOINT_RELATIVE_PATHS)
)
def test_canonical_command_executes_each_allowlisted_frozen_entrypoint(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    root, head, *_rest = _frozen_repository(tmp_path)
    marker = tmp_path / f"{entrypoint}-sealed-entrypoint-marker"
    arbitrary_cwd = tmp_path / f"{entrypoint}-arbitrary-cwd"
    arbitrary_cwd.mkdir()
    command = launcher.canonical_command(root, head, entrypoint=entrypoint)
    completed = subprocess.run(
        ["/bin/bash", "-c", f"{command} {shlex.quote(str(marker))}"],
        check=False,
        capture_output=True,
        text=True,
        cwd=arbitrary_cwd,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "safe"
    assert launcher.ENTRYPOINT_RELATIVE_PATHS[entrypoint] in (
        row["path"]
        for row in launcher.build_frozen_source_bundle(
            root,
            head,
            runner_relative_path=launcher.ENTRYPOINT_RELATIVE_PATHS[entrypoint],
        )[0]["files"]
    )


def test_common_source_authority_is_selector_invariant_and_routing_is_per_invocation(
    tmp_path: Path,
) -> None:
    root, head, *_rest = _frozen_repository(tmp_path)
    bundles: dict[str, dict[str, Any]] = {}
    routings: dict[str, dict[str, Any]] = {}
    sources: dict[str, bytes] = {}
    for selector, relative_path in launcher.ENTRYPOINT_RELATIVE_PATHS.items():
        bundle, source = launcher.build_frozen_source_bundle(
            root,
            head,
            runner_relative_path=relative_path,
        )
        bundles[selector] = bundle
        sources[selector] = source
        routings[selector] = launcher._launch_routing_binding(
            bundle,
            source,
            selector,
        )

    assert bundles["matrix"] == bundles["audit"] == bundles["summary"]
    assert len({bundle["bundle_sha256"] for bundle in bundles.values()}) == 1
    assert bundles["matrix"]["entrypoint_relative_paths"] == (launcher.ENTRYPOINT_RELATIVE_PATHS)
    assert len(set(sources.values())) == len(launcher.ENTRYPOINT_RELATIVE_PATHS)
    assert len({launcher._canonical_json(binding) for binding in routings.values()}) == 3
    for selector, routing in routings.items():
        relative_path = launcher.ENTRYPOINT_RELATIVE_PATHS[selector]
        selected_row = next(
            row for row in bundles[selector]["files"] if row["path"] == relative_path
        )
        assert routing == {
            "schema_version": 1,
            "launcher": launcher.LAUNCHER_ID,
            "entrypoint_selector": selector,
            "entrypoint_relative_path": relative_path,
            "source_bundle_sha256": bundles[selector]["bundle_sha256"],
            "git_mode": selected_row["git_mode"],
            "git_blob_oid": selected_row["git_blob_oid"],
            "sha256": selected_row["sha256"],
            "bytes": selected_row["bytes"],
        }


def test_canonical_command_rejects_non_allowlisted_entrypoint_before_launch(
    tmp_path: Path,
) -> None:
    root, head, *_rest = _frozen_repository(tmp_path)
    with pytest.raises(ValueError, match="not allowlisted"):
        launcher.canonical_command(root, head, entrypoint="../arbitrary-python")
    with pytest.raises(ValueError, match="not allowlisted"):
        launcher.build_frozen_source_bundle(
            root,
            head,
            runner_relative_path="research/adaptive_v4_memory/scripts/arbitrary.py",
        )


def test_all_pinned_git_queries_disable_commit_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(arguments: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed["arguments"] = list(arguments)
        observed.update(kwargs)
        return subprocess.CompletedProcess(arguments, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    launcher._git(tmp_path.resolve(), ["rev-parse", "HEAD"], check=False)

    assert observed["arguments"][:5] == [
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.commitGraph=false",
    ]
