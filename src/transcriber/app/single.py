"""Одна копия приложения на машину.

Второй запуск не открывает второе окно, а поднимает уже открытое. Копии не
конфликтуют явно — порт раздачи звука система выдаёт свободный, — но у каждой
своя история и свой признак занятости, а папка вывода и кэш общие. Расхождение
между ними человек видит как пропавшие файлы, и понять причину неоткуда.

Замок — файл с номером процесса и временем его запуска. Номера переиспользуются,
поэтому мало убедиться, что процесс жив: под старым номером может оказаться
чужая программа, и отличает её именно время запуска.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import paths


def lock_file() -> Path:
    """Замок лежит рядом с состоянием окна, а не в кэше: кэш чистят не глядя."""
    return paths.config_dir() / "app.pid"


def running() -> int | None:
    """Номер уже работающей копии или `None`, если её нет."""
    try:
        pid_text, _, started = lock_file().read_text(encoding="utf-8").partition("\n")
        pid = int(pid_text.strip())
    except (OSError, ValueError):
        # Замка нет или он испорчен — считаем, что копия не запущена.
        return None

    if pid == os.getpid() or not _alive(pid):
        return None

    # Номера переиспользуются: под старым может оказаться чужая программа. Время
    # запуска отличает её надёжнее, чем командная строка, — у консольного скрипта
    # в ней стоит путь к интерпретатору, по которому себя не узнать.
    return pid if started.strip() and _started(pid) == started.strip() else None


def focus(pid: int) -> bool:
    """Поднимает окно работающей копии на передний план.

    Через System Events, потому что бандл у нас скриптовый: у процесса нет
    собственного идентификатора приложения, за который можно взяться иначе.
    """
    if sys.platform != "darwin":
        return False
    script = (
        "tell application \"System Events\" to set frontmost of "
        f"(first process whose unix id is {pid}) to true"
    )
    done = subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    return done.returncode == 0


@contextmanager
def claim() -> Iterator[None]:
    """Держит замок, пока открыто окно, и убирает его при выходе."""
    path = lock_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path.write_text(f"{pid}\n{_started(pid)}", encoding="utf-8")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    """Существует ли процесс. Нулевой сигнал ничего не посылает — только спрашивает."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Процесс существует, но принадлежит другому пользователю — значит не наш.
        return False
    return True


def _started(pid: int) -> str:
    """Когда процесс запущен. Пустая строка — узнать не удалось."""
    done = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, check=False
    )
    return done.stdout.strip()
