from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

DEFAULT_LOCK_PATH = Path("/tmp/nano-deepseek-v4-adaptive-v4-memory-gpu.lock")


def acquire_gpu_lock(label: str, *, path: Path = DEFAULT_LOCK_PATH) -> IO[str]:
    """Hold an exclusive process-lifetime lock for paper-grade GPU experiments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        holder = handle.read().strip() or "unknown holder"
        handle.close()
        raise RuntimeError(f"Adaptive V4 GPU is already locked by {holder}.") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} label={label}\n")
    handle.flush()
    return handle
