"""Иконка окна.

Систему подменяем: `_the_icon` спрашивает `sys.platform` в момент вызова, и обе
ветки должны проверяться с любой машины — та, где иконку называем мы, и та, где
её называет бандл.
"""

from __future__ import annotations

import sys

from transcriber import app


def test_the_icon_is_shipped():
    """Ярлык и окно берут один файл; пропадёт — обоим нечего показать."""
    assert (app.ASSETS / "icon.ico").exists()


def test_windows_window_is_given_the_icon(monkeypatch):
    """Регрессия: окно открывалось со стандартной иконкой WinForms.

    Своя иконка в проекте была, но доставалась только ярлыку — а его иконка
    кончается на рабочем столе. Окну pywebview ищет её сам в `sys.executable`, и
    в окружении, собранном uv, там трамплин без единого ресурса иконки.
    """
    monkeypatch.setattr(sys, "platform", "win32")

    assert app._window_icon() == str(app.ASSETS / "icon.ico")


def test_macos_leaves_the_icon_to_the_bundle(monkeypatch):
    """На macOS иконку несёт `Info.plist`, и вторая только перебила бы её."""
    monkeypatch.setattr(sys, "platform", "darwin")

    assert app._window_icon() is None
