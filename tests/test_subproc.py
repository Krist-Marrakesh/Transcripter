"""Starting outside programs quietly.

The whole point of the module is a branch that only Windows takes, so the system
is faked here. That is honest: the function reads `sys.platform` and nothing
else, and what the number does to a real child process is not ours to check.
"""

from __future__ import annotations

import sys

from transcriber import subproc
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


def test_interpreter_runs_inside_our_environment(monkeypatch, tmp_path):
    """Найдено вживую: окно запускается через Python.app, а у того нет venv.

    Загрузчик тогда падал с ModuleNotFoundError мгновенно, и окно показывало
    нулевую скорость — при исправной сети и живом с виду процессе.
    """
    venv = tmp_path / "bin"
    venv.mkdir()
    (venv / "python").write_text("", encoding="utf-8")
    monkeypatch.setattr(subproc.sys, "prefix", str(tmp_path))

    assert subproc.interpreter() == str(venv / "python")


def test_interpreter_falls_back_to_the_current_interpreter(monkeypatch, tmp_path):
    """Вне venv брать больше нечего — и это нормально."""
    monkeypatch.setattr(subproc.sys, "prefix", str(tmp_path))

    assert subproc.interpreter() == subproc.sys.executable


def test_interpreter_knows_the_windows_venv_layout(monkeypatch, tmp_path):
    """Там интерпретатор лежит в Scripts и называется python.exe.

    По пути `bin/python` его не найти, и загрузчик молча ушёл бы к чужому
    интерпретатору — той же ошибкой, что уже стоила нам нулевой скорости.
    """
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "python.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(subproc.sys, "platform", "win32")
    monkeypatch.setattr(subproc.sys, "prefix", str(tmp_path))

    assert subproc.interpreter() == str(scripts / "python.exe")
