"""Starting outside programs quietly.

The whole point of the module is a branch that only Windows takes, so the system
is faked here. That is honest: the function reads `sys.platform` and nothing
else, and what the number does to a real child process is not ours to check.
"""

from __future__ import annotations

import sys

from transcriber.subproc import CREATE_NO_WINDOW, quiet_flags


def test_nothing_to_hide_outside_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    assert quiet_flags() == {}


def test_a_console_child_gets_no_window_of_its_own(monkeypatch):
    """Регрессия: окно приложения консоли не имеет, а ffmpeg — консольный.

    Без флага на каждый вызов ffprobe поверх интерфейса мигал бы чёрный
    прямоугольник, и на длинном файле это десятки вспышек.
    """
    monkeypatch.setattr(sys, "platform", "win32")

    assert quiet_flags() == {"creationflags": CREATE_NO_WINDOW}


def test_the_system_is_read_at_the_call(monkeypatch):
    """Значение, замороженное при импорте, нельзя было бы проверить отсюда."""
    monkeypatch.setattr(sys, "platform", "win32")
    windows = quiet_flags()
    monkeypatch.setattr(sys, "platform", "darwin")

    assert windows != quiet_flags()
