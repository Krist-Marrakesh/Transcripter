"""One copy of the application per machine.

A second launch does not open a second window — it raises the one already open.
The copies do not clash outright, since the system hands each audio server a free
port, but every copy keeps its own history and its own busy flag while sharing the
output folder and the cache. A person sees the divergence as files that went
missing, with nothing to explain where.

The lock is a file holding a process number and the time that process started.
Numbers get reused, so it is not enough to confirm the process is alive: someone
else's program may be sitting under the old number, and the start time is what
tells them apart.

Every question about the process itself goes to `process` — the systems answer
them in ways that have nothing in common, and the decision made here should not
depend on which one is answering.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import paths
from . import process


def lock_file() -> Path:
    """The lock sits beside the window state, not in the cache: caches get wiped."""
    return paths.config_dir() / "app.pid"


def running() -> int | None:
    """The number of a copy already running, or `None` if there is none."""
    try:
        pid_text, _, started = lock_file().read_text(encoding="utf-8").partition("\n")
        pid = int(pid_text.strip())
    except (OSError, ValueError):
        # No lock, or a damaged one — treat it as no copy running.
        return None

    if pid == os.getpid() or not process.alive(pid):
        return None

    # Numbers are reused: someone else's program may be under the old one. The
    # start time identifies it more reliably than the command line, where a
    # console script shows the path to the interpreter and nothing of ours.
    return pid if started.strip() and process.started_at(pid) == started.strip() else None


def focus(pid: int) -> bool:
    """Brings the window of the running copy to the front."""
    return process.focus(pid)


@contextmanager
def claim() -> Iterator[None]:
    """Holds the lock while the window is open and clears it on the way out."""
    path = lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path.write_text(f"{pid}\n{process.started_at(pid)}", encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
