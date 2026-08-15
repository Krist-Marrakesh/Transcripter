"""Вывод командного интерфейса на системе, чья кодовая страница не знает русского.

Всё, что интерфейс печатает, написано по-русски, а Windows отдаёт процессу
кодировку консоли: на английской системе это `cp1252`. Падало не на ошибке, а на
обычной строке вывода — `transcript info` умирал `UnicodeEncodeError` на первом
же заголовке таблицы, ещё ничего не сообщив.

Проверяется настоящим потоком в `cp1252`, а не заглушкой: заглушка приняла бы
любую строку и молчала бы ровно там, где ломается настоящий вывод.
"""

from __future__ import annotations

import io
import subprocess
import sys

from transcriber import cli

RUSSIAN = "Устройство · Транскрибатор"


def encoded(encoding: str) -> io.TextIOWrapper:
    """Поток, который умеет ровно то, что умеет названная кодировка."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding)


def test_a_western_code_page_breaks_an_untouched_stream():
    """Сначала — что ломается. Иначе следующий тест ничего не доказывает."""
    stream = encoded("cp1252")

    try:
        stream.write(RUSSIAN)
        stream.flush()
    except UnicodeEncodeError:
        return
    raise AssertionError("cp1252 внезапно принял кириллицу — проверка потеряла смысл")


def test_the_streams_are_made_able_to_say_it(monkeypatch):
    """Регрессия: правится поток, а не консоль поверх него.

    У rich кодировка не настраивается — он берёт её у файла, в который пишет, —
    поэтому чинить надо там, куда он пишет.
    """
    out, err = encoded("cp1252"), encoded("cp1252")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    cli._let_it_speak_russian()

    assert (out.encoding, err.encoding) == ("utf-8", "utf-8")
    out.write(RUSSIAN)
    out.flush()
    assert RUSSIAN.encode("utf-8") in out.buffer.getvalue()


def test_a_missing_stream_is_not_an_error(monkeypatch):
    """Под `pythonw` консоли нет вовсе, и потоки там `None`."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    cli._let_it_speak_russian()


def test_the_command_speaks_where_the_console_cannot():
    """Настоящий запуск, а не вызов функции: важен весь путь до печати.

    Кодировка навязывается процессу так же, как её навязывает Windows, — и без
    правки `info` падал здесь ровно тем же `UnicodeEncodeError`, что на раннере.
    """
    done = subprocess.run(
        [sys.executable, "-m", "transcriber.cli", "info"],
        capture_output=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "cp1252"},
    )

    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-400:]
    assert "Устройство".encode() in done.stdout
