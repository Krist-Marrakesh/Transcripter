"""Сборка того, что уезжает людям.

Проверяется без самой сборки: она тянет колесо, uv и архив на два десятка
мегабайт. Здесь — только то, что ломается тихо и целиком, а замечается позже
всех: содержимое лаунчеров.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    """Модуль сборки. Он лежит вне пакета, поэтому загружается по пути."""
    spec = importlib.util.spec_from_file_location("release", ROOT / "tools" / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_windows_launcher_is_ascii_to_the_last_character(release):
    """Регрессия: два типографских тире, и Windows-архив не собирался вовсе.

    Скрипт пишется `encoding="ascii"` не из вкуса: PowerShell читает файл без
    метки порядка байтов в системной кодировке, и любой не-ASCII символ приезжает
    мусором на всякой машине, где кодировка не авторская. Правило было записано
    комментарием и ничем не подкреплено — в комментарии же его и нарушили, а
    сборка падала `UnicodeEncodeError` в самом конце, после колеса и окружения.
    """
    bad = sorted({character for character in release._LAUNCHER_PS1 if not character.isascii()})

    assert bad == [], f"не-ASCII в launcher.ps1: {bad}"


def test_the_cmd_that_starts_it_is_ascii_too(release):
    """Тот же запрет и по той же причине: `.cmd` читается той же кодировкой."""
    assert release._LAUNCHER_CMD.isascii()


def test_the_windows_launcher_starts_the_windowed_interpreter(release):
    """Консольный держал бы чёрное окно за интерфейсом весь сеанс."""
    assert "pythonw.exe" in release._LAUNCHER_PS1
    assert "Start-Process -FilePath $windowed" in release._LAUNCHER_PS1


def test_the_launchers_hand_down_the_folder_they_live_in(release):
    """Веса кладутся внутрь приложения, и путь к нему процесс сам не найдёт.

    `sys.prefix` указывает на окружение, а оно лежит рядом, но не там. Пропадёт
    переменная — веса разъедутся по общим кэшам машины и переживут её удаление.
    """
    assert "TRANSCRIPT_MODELS" in release._LAUNCHER_PS1
    assert "TRANSCRIPT_MODELS" in release._LAUNCHER
