"""Questions put to the operating system about a process.

These run against the real system rather than a stub: the whole point of the
module is that each platform answers differently, and a stub would only confirm
our idea of the answer.
"""

from __future__ import annotations

import os
import subprocess
import sys

from transcriber.app import process

# A number no system hands out: PIDs are capped far below this on both platforms.
NOBODY = 99_999_999


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def test_a_live_process_is_seen_and_left_alone():
    """Regression: on Windows the null signal is not a question but an order.

    CPython implements `os.kill` there through `TerminateProcess`, so the POSIX
    way of asking "are you alive" would kill the copy it was looking for — the
    second launch would close the first window instead of raising it.
    """
    child = _sleeper()
    try:
        assert process.alive(child.pid) is True
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_a_finished_process_is_not_alive():
    """A lock left by a killed copy must not forbid opening the application."""
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait(timeout=10)

    assert process.alive(child.pid) is False


def test_nobody_holds_an_impossible_number():
    assert process.alive(NOBODY) is False
    assert process.started_at(NOBODY) == ""


def test_our_own_start_time_is_known():
    """Without it the lock cannot tell us from a program that took our number."""
    assert process.started_at(os.getpid()) != ""


def test_start_time_survives_a_second_look():
    """It is written into the lock once and compared later — it must not drift."""
    assert process.started_at(os.getpid()) == process.started_at(os.getpid())
