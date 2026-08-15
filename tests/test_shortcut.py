"""Сборка ярлыка: бандл на macOS, `.lnk` на Windows.

Половины проверяются на своих системах, и иначе никак: каждая зовёт средства
сборки, которых на чужой машине нет. Пропуск здесь — не обход неудобства, а
отсутствие самого случая. Windows-половина не проверялась вовсе, пока весь файл
пропускался вне macOS, — а ярлык там устроен сложнее, чем на маке: его пишет
PowerShell через COM, потому что формат `.lnk` не описан.
"""

from __future__ import annotations

import platform
import plistlib
import sys

import pytest

from transcriber.app import shortcut
from transcriber.app.shortcut import BUNDLE_NAME, LINK_NAME, create

macos = pytest.mark.skipif(sys.platform != "darwin", reason="бандл только для macOS")
windows = pytest.mark.skipif(sys.platform != "win32", reason="`.lnk` только для Windows")


@macos
def test_builds_launchable_bundle(tmp_path):
    bundle = create(tmp_path)

    launcher = bundle / "Contents" / "MacOS" / "launcher"
    assert launcher.exists()
    # Без бита исполнения Finder откажется открывать бандл.
    assert launcher.stat().st_mode & 0o111
    assert sys.executable in launcher.read_text(encoding="utf-8")


@macos
def test_launcher_pins_architecture(tmp_path):
    """Регрессия: Python универсальный, и Finder запускал x86_64-срез.

    Скомпилированные пакеты собраны под одну архитектуру, поэтому импорт падал
    ещё до окна — со стороны это выглядело как «иконка подпрыгнула и погасла».
    """
    script = (create(tmp_path) / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")

    assert f"/usr/bin/arch -{platform.machine()}" in script


@macos
def test_launcher_keeps_a_log(tmp_path):
    """Finder не даёт терминала: без журнала любая ошибка теряется молча."""
    script = (create(tmp_path) / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")

    assert "Library/Logs" in script


@macos
def test_launcher_restores_the_homebrew_path(tmp_path):
    """Регрессия: из Finder PATH пустой, и ffmpeg выглядел неустановленным.

    Профиль оболочки при запуске приложения не читается, поэтому папок Homebrew
    в PATH нет — а весь ingest стоит на ffmpeg и ffprobe.
    """
    script = (create(tmp_path) / "Contents" / "MacOS" / "launcher").read_text(encoding="utf-8")

    assert "/opt/homebrew/bin" in script
    assert "/usr/local/bin" in script


@macos
def test_plist_points_at_launcher_and_icon(tmp_path):
    bundle = create(tmp_path)
    info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())

    assert info["CFBundleExecutable"] == "launcher"
    assert info["CFBundleIconFile"] == "icon"
    assert (bundle / "Contents" / "Resources" / "icon.icns").exists()


@macos
def test_rebuild_replaces_previous(tmp_path):
    first = create(tmp_path)
    marker = first / "Contents" / "мусор.txt"
    marker.write_text("остаток прошлой сборки", encoding="utf-8")

    create(tmp_path)

    assert not marker.exists()


@macos
def test_refuses_to_delete_something_else(tmp_path):
    """Каталог с тем же именем, но не наш, трогать нельзя."""
    stranger = tmp_path / BUNDLE_NAME
    (stranger / "Contents").mkdir(parents=True)
    (stranger / "важное.txt").write_text("чужое", encoding="utf-8")

    with pytest.raises(RuntimeError, match="не похож на наш ярлык"):
        create(tmp_path)


def inside(link, text: str) -> bool:
    """Есть ли строка внутри `.lnk`.

    Формат не описан и разбирать его нечем, зато строки лежат в нём как есть, в
    UTF-16. Этого хватает, чтобы проверить главное — на что ярлык указывает и с
    чем запускает; читать его обратно через COM значило бы проверять COM.
    """
    return text.encode("utf-16-le") in link.read_bytes()


@windows
def test_builds_a_link_beside_the_asked_folder(tmp_path):
    link = create(tmp_path)

    assert link == tmp_path / LINK_NAME
    assert link.is_file()


@windows
def test_the_link_runs_our_application(tmp_path):
    """Ярлык зовёт модуль, а не копирует окружение: гигабайты внутрь не уедут."""
    link = create(tmp_path)

    assert inside(link, "-m transcriber.cli app")


@windows
def test_the_link_asks_for_the_windowed_interpreter(tmp_path):
    """`python.exe` держал бы за интерфейсом чёрное окно весь сеанс."""
    link = create(tmp_path)

    assert inside(link, "pythonw.exe")


@windows
def test_the_link_carries_the_icon(tmp_path):
    """Иконка ярлыка — единственная, которую видно до запуска."""
    link = create(tmp_path)

    assert inside(link, "icon.ico")


@windows
def test_a_folder_in_the_way_is_refused(tmp_path):
    """Папка с именем ярлыка: снести её молча — значит однажды снести чужое."""
    (tmp_path / LINK_NAME).mkdir()

    with pytest.raises(RuntimeError, match="папка, а не ярлык"):
        create(tmp_path)


def test_the_windowed_interpreter_falls_back_to_the_plain_one(tmp_path, monkeypatch):
    """Оконного рядом может не оказаться — тогда лучше консольный, чем ничего."""
    plain = tmp_path / "python.exe"
    plain.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(plain))

    assert shortcut._windowed() == plain

    (tmp_path / "pythonw.exe").write_bytes(b"")

    assert shortcut._windowed() == tmp_path / "pythonw.exe"


def test_the_script_reaches_powershell_whatever_the_code_page():
    """Регрессия: имя ярлыка терялось по дороге и файл не сохранялся.

    Скрипт уходил обычной `-Command`, а она проходит через кодовую страницу
    консоли. На машине, где кириллицы в ней нет, «Транскрибатор» приезжал
    тринадцатью вопросительными знаками; `?` в имени файла Windows не допускает,
    и `$link.Save()` отвечал `FileNotFoundException`, не поминая ни кодировки, ни
    имени. На русской Windows всё собиралось, поэтому дефект и жил.

    Проверяется свойство, а не способ: то, что уходит в PowerShell, — чистый
    ASCII, и потому кодовой странице нечего портить. Обратный разбор здесь же,
    иначе тест прошёл бы и на пустой строке.
    """
    from base64 import b64decode

    script = shortcut._LINK.format(
        path="C:/Users/аня/Транскрибатор.lnk",
        python="C:/py/pythonw.exe",
        workdir="C:/Users/аня",
        icon="",
    )

    wire = shortcut._encoded(script)

    assert wire.isascii()
    assert b64decode(wire).decode("utf-16-le") == script
