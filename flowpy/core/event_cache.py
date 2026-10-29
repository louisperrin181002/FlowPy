"""Temporary event storage for large FCS files.

The events of a big FCS file (hundreds of MB, e.g. a full-spectrum file
with 188 channels) are kept in a temporary file on the local disk and
memory-mapped, instead of in RAM. The operating system then pages in only
the channels actually used and can drop them again under memory pressure
instead of swapping, so 80 files × 600 MB work on a Mac with less RAM.

Nothing is kept: the files live in a per-session folder in the system's
temporary directory, removed when FlowPy closes (and, after a crash, the
next time FlowPy starts). The FCS files themselves are never copied.
"""

from __future__ import annotations
import atexit
import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np

MIN_FILE_BYTES = 64 * 2**20          # smaller files simply stay in RAM
KEEP_FREE_BYTES = 10 * 2**30         # never fill the disk beyond this
_PREFIX = "flowpy-events-"

_lock = threading.Lock()
_dir: Path | None = None
# source file → (metadata, column names, memmap) for re-opening a file
# (e.g. a workspace) within the same session
_open: dict[tuple, tuple] = {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _session_dir() -> Path:
    global _dir
    with _lock:
        if _dir is None:
            root = Path(tempfile.gettempdir())
            # leftovers of FlowPy sessions that crashed
            for old in root.glob(_PREFIX + "*"):
                try:
                    if not _alive(int(old.name.rsplit("-", 1)[1])):
                        shutil.rmtree(old, ignore_errors=True)
                except ValueError:
                    pass
            _dir = root / f"{_PREFIX}{os.getpid()}"
            _dir.mkdir(exist_ok=True)
            atexit.register(cleanup)
        return _dir


def cleanup():
    """Delete this session's temporary event files."""
    global _dir
    _open.clear()
    if _dir is not None:
        shutil.rmtree(_dir, ignore_errors=True)
        _dir = None


def _key(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (path, st.st_size, st.st_mtime_ns)


def load(path: str):
    """(metadata, column names, (channels × events) float32 memmap) if
    *path* was already read in this session, else None."""
    k = _key(path)
    return _open.get(k) if k else None


def store(path: str, meta: dict, columns: list[str], arr: np.ndarray):
    """Move *arr* (channels × events, float32) out of RAM into a temporary
    memory-mapped file. Returns the memmap, or None (then keep *arr*)."""
    k = _key(path)
    if k is None:
        return None
    try:
        root = _session_dir()
        if shutil.disk_usage(root).free - arr.nbytes < KEEP_FREE_BYTES:
            return None
        name = hashlib.sha1(repr(k).encode("utf-8", "surrogateescape")
                            ).hexdigest()
        target = root / f"{name}.f32"
        tmp = root / f"{name}.tmp{threading.get_ident()}"
        np.ascontiguousarray(arr, dtype=np.float32).tofile(tmp)
        os.replace(tmp, target)
        # copy-on-write: edits stay in memory, never touch the file
        mm = np.memmap(target, dtype=np.float32, mode="c", shape=arr.shape)
    except OSError:
        return None
    _open[k] = (meta, list(columns), mm)
    return mm
