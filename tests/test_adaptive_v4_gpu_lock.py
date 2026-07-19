from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from adaptive_v4_gpu_lock import (  # noqa: E402
    SAFE_LOCK_MODE,
    acquire_device_guard,
    acquire_gpu_lock,
    canonical_device_guard_path,
)


def test_gpu_lock_is_exclusive_persistent_and_reacquirable(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    lease = acquire_gpu_lock("first", path=path)
    assert path.stat().st_mode & 0o777 == SAFE_LOCK_MODE
    assert path.stat().st_nlink == 1
    assert path.read_text(encoding="utf-8") == f"pid={os.getpid()} label=first\n"
    with pytest.raises(RuntimeError, match="already locked"):
        acquire_gpu_lock("second", path=path)
    lease.assert_held()
    lease.close()
    assert path.exists()
    reacquired = acquire_gpu_lock("second", path=path)
    reacquired.close()


def test_gpu_lock_rejects_symlink_hardlink_and_unsafe_mode(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o600)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="symbolic link"):
        acquire_gpu_lock("symlink", path=symlink)

    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)
    with pytest.raises(RuntimeError, match="ownership, links, or mode"):
        acquire_gpu_lock("hardlink", path=hardlink)
    hardlink.unlink()

    target.chmod(0o644)
    with pytest.raises(RuntimeError, match="ownership, links, or mode"):
        acquire_gpu_lock("mode", path=target)


def test_gpu_lock_release_detects_path_replacement(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    lease = acquire_gpu_lock("replace", path=path)
    path.unlink()
    path.write_text("replacement", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="replaced"):
        lease.close()
    assert lease.closed


def test_inherited_descriptor_keeps_kernel_lease_after_originating_fd_closes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gpu.lock"
    lease = acquire_gpu_lock("originating-runner", path=path)
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(1.0)"],
        pass_fds=(lease.fileno(),),
    )
    os.close(lease.file_descriptor)
    lease._closed = True
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            acquire_gpu_lock("competing-runner", path=path)
    finally:
        child.wait(timeout=5)

    reacquired = acquire_gpu_lock("after-child-exit", path=path)
    reacquired.close()


def test_device_guard_is_identity_derived_and_cannot_be_bypassed_by_scheduler_path(
    tmp_path: Path,
) -> None:
    identity = {"identity_type": "uuid", "identity": f"GPU-{tmp_path.name}"}
    other = {"identity_type": "uuid", "identity": f"GPU-other-{tmp_path.name}"}
    first = acquire_device_guard("first-scheduler", identity)
    try:
        assert first.path == canonical_device_guard_path(identity)
        with pytest.raises(RuntimeError, match="already locked"):
            acquire_device_guard("different-scheduler-path", identity)
        independent = acquire_device_guard("different-device", other)
        independent.close()
    finally:
        first.close()


def test_sigkill_parent_leaves_scheduler_and_device_guard_held_by_orphan_child(
    tmp_path: Path,
) -> None:
    scheduler_path = tmp_path / "arbitrary-scheduler.lock"
    identity = {"identity_type": "uuid", "identity": f"GPU-orphan-{tmp_path.name}"}
    runner_source = "\n".join(
        (
            "import subprocess,sys,time",
            f"sys.path.insert(0, {str(SCRIPTS)!r})",
            "import adaptive_v4_gpu_lock as lock",
            f"identity = {identity!r}",
            f"scheduler = lock.acquire_gpu_lock('runner', path=lock.Path({str(scheduler_path)!r}))",
            "guard = lock.acquire_device_guard('runner', identity)",
            "child = subprocess.Popen(",
            "    [sys.executable, '-I', '-c', 'import time; time.sleep(30)'],",
            "    pass_fds=(scheduler.fileno(), guard.fileno()),",
            ")",
            "print(child.pid, flush=True)",
            "time.sleep(30)",
        )
    )
    runner = subprocess.Popen(
        [sys.executable, "-I", "-c", runner_source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    orphan_pid: int | None = None
    try:
        assert runner.stdout is not None
        line = runner.stdout.readline().strip()
        if not line:
            assert runner.stderr is not None
            raise AssertionError(runner.stderr.read())
        orphan_pid = int(line)
        runner.send_signal(signal.SIGKILL)
        runner.wait(timeout=5)

        with pytest.raises(RuntimeError, match="already locked"):
            acquire_gpu_lock("competitor", path=scheduler_path)
        with pytest.raises(RuntimeError, match="already locked"):
            acquire_device_guard("competitor", identity)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        if orphan_pid is not None:
            try:
                os.kill(orphan_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + 5
    while True:
        scheduler = None
        guard = None
        try:
            guard = acquire_device_guard("after-orphan", identity)
            scheduler = acquire_gpu_lock("after-orphan", path=scheduler_path)
            break
        except RuntimeError:
            if scheduler is not None:
                scheduler.close()
            if guard is not None:
                guard.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    guard.close()
    scheduler.close()


@pytest.mark.parametrize("label", ["", " padded", "padded ", "line\nbreak", "x" * 257])
def test_gpu_lock_rejects_ambiguous_labels(tmp_path: Path, label: str) -> None:
    with pytest.raises(RuntimeError, match="label"):
        acquire_gpu_lock(label, path=tmp_path / "gpu.lock")
