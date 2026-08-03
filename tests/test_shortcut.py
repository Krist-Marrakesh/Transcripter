"""Сборка ярлыка."""

from __future__ import annotations

import platform
import plistlib
import sys

import pytest

from transcriber.app.shortcut import BUNDLE_NAME, create

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="бандл только для macOS")


def test_builds_launchable_bundle(tmp_path):
    bundle = create(tmp_path)

    launcher = bundle / "Contents" / "MacOS" / "launcher"
    assert launcher.exists()
    # Без бита исполнения Finder откажется открывать бандл.
    assert launcher.stat().st_mode & 0o111
    assert sys.executable in launcher.read_text(encoding="utf-8")


def test_launcher_pins_architecture(tmp_path):
    """Регрессия: Python универсальный, и Finder запускал x86_64-срез.

    Скомпилированные пакеты собраны под одну архитектуру, поэтому импорт падал
    ещё до окна — со стороны это выглядело как «иконка подпрыгнула и погасла».
    """
    script = (create(tmp_path) / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")

    assert f"/usr/bin/arch -{platform.machine()}" in script


def test_launcher_keeps_a_log(tmp_path):
    """Finder не даёт терминала: без журнала любая ошибка теряется молча."""
    script = (create(tmp_path) / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")

    assert "Library/Logs" in script


def test_plist_points_at_launcher_and_icon(tmp_path):
    bundle = create(tmp_path)
    info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())

    assert info["CFBundleExecutable"] == "launcher"
    assert info["CFBundleIconFile"] == "icon"
    assert (bundle / "Contents" / "Resources" / "icon.icns").exists()


def test_rebuild_replaces_previous(tmp_path):
    first = create(tmp_path)
    marker = first / "Contents" / "мусор.txt"
    marker.write_text("остаток прошлой сборки", encoding="utf-8")

    create(tmp_path)

    assert not marker.exists()


def test_refuses_to_delete_something_else(tmp_path):
    """Каталог с тем же именем, но не наш, трогать нельзя."""
    stranger = tmp_path / BUNDLE_NAME
    (stranger / "Contents").mkdir(parents=True)
    (stranger / "важное.txt").write_text("чужое", encoding="utf-8")

    with pytest.raises(RuntimeError, match="не похож на наш ярлык"):
        create(tmp_path)

    assert (stranger / "важное.txt").exists()
