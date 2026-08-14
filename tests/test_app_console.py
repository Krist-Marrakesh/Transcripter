"""Консоль за окном и журнал запуска.

Ярлык целится в `pythonw.exe` — программу без консоли, — но в окружении, собранном
uv, это трамплин, и запускает он консольную сборку интерпретатора. Чёрное окно
стоит за интерфейсом весь сеанс, а `sys.stderr` при этом не пуст, из-за чего не
писался и журнал: любая неудача запуска оставалась без следа.
"""

from __future__ import annotations

import io
import sys

import pytest

from transcriber import paths
from transcriber.app import _console_of_our_own, _hide_our_console, _keep_a_log


def test_nothing_to_hide_where_there_are_no_consoles(monkeypatch):
    """На маке и в Linux вопроса не существует, и спрашивать систему не за чем."""
    monkeypatch.setattr(sys, "platform", "darwin")

    assert _console_of_our_own() == 0
    assert _hide_our_console() is False


@pytest.mark.skipif(sys.platform != "win32", reason="консоль бывает только на Windows")
def test_a_terminal_of_someone_elses_is_left_alone():
    """Главная опасность: спрятать окно оболочки, в которой команду и набрали.

    Тесты запускаются из терминала, где приглашение уже напечатано, — курсор
    сдвинут, и консоль по этому признаку чужая. Спрятать её было бы худшим из
    возможных исходов: человек набрал команду и лишился окна, в котором работал.
    """
    assert _console_of_our_own() == 0


def test_a_launch_without_streams_gets_a_log(tmp_path, monkeypatch):
    """`pythonw` оставляет `sys.stderr` пустым: `print` молчит, ошибки пропадают."""
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    _keep_a_log()
    stream = sys.stderr

    assert (tmp_path / "launch.log").exists()
    stream.close()


def test_a_hidden_console_gets_one_too(tmp_path, monkeypatch):
    """Потоки живы, но их окна больше нет — писать в него всё равно что никуда.

    Регрессия по замыслу: журнал заводился только по пустому `sys.stderr`, а
    спрятанная консоль оставляет его настоящим. Без этой ветки та самая
    страховка исчезала ровно там, где её и придумали, — при запуске ярлыком.
    """
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    _keep_a_log(hidden=True)
    stream = sys.stderr

    assert "запуск" in (tmp_path / "launch.log").read_text(encoding="utf-8")
    stream.close()


def test_streams_that_someone_can_see_are_left_as_they_are(tmp_path, monkeypatch):
    """Из терминала журнал не нужен: там всё и так видно."""
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    kept = io.StringIO()
    monkeypatch.setattr(sys, "stderr", kept)

    _keep_a_log()

    assert sys.stderr is kept
    assert not (tmp_path / "launch.log").exists()
