"""Сборка ярлыка на рабочем столе.

Бандл получается тонким: внутри не копия окружения, а запуск того интерпретатора,
которым собран проект. Полноценная упаковка через PyInstaller здесь была бы
вредной — она потянула бы внутрь torch и веса моделей, то есть гигабайты, ради
двойного клика.

Обратная сторона честная: ярлык привязан к каталогу проекта. Переедет проект —
ярлык надо пересобрать.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
BUNDLE_NAME = "Транскрибатор.app"

_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Транскрибатор</string>
  <key>CFBundleDisplayName</key><string>Транскрибатор</string>
  <key>CFBundleIdentifier</key><string>local.transcriber.app</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
"""

_LAUNCHER = """#!/bin/sh
# Ярлык не копирует окружение, а зовёт то, которым собран проект.
# Finder запускает без терминала, поэтому вывод уходит в журнал: иначе любая
# ошибка выглядит как «иконка подпрыгнула и ничего не произошло».
LOG="$HOME/Library/Logs/Транскрибатор.log"
mkdir -p "$(dirname "$LOG")"
{{
  echo "--- запуск $(date '+%Y-%m-%d %H:%M:%S') ---"
  # Архитектура задаётся явно. Python универсальный, и для скриптового бандла
  # LaunchServices выбирает x86_64-срез, а скомпилированные пакеты собраны под
  # {arch} — из Finder это выглядело как «иконка подпрыгнула и погасла».
  exec /usr/bin/arch -{arch} "{python}" -m transcriber.cli app
}} >> "$LOG" 2>&1
"""


def create(destination: Path, *, version: str = "0.1.0") -> Path:
    """Собирает .app в указанном каталоге и возвращает путь к нему."""
    if sys.platform != "darwin":
        raise RuntimeError("ярлык собирается только на macOS")

    bundle = destination.expanduser().resolve() / BUNDLE_NAME
    _clear(bundle)

    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    (bundle / "Contents" / "Info.plist").write_text(
        _PLIST.format(version=version), encoding="utf-8"
    )

    launcher = macos / "launcher"
    launcher.write_text(
        _LAUNCHER.format(python=sys.executable, arch=platform.machine()), encoding="utf-8"
    )
    launcher.chmod(0o755)

    icon = ASSETS / "icon.icns"
    if icon.exists():
        shutil.copy2(icon, resources / "icon.icns")

    return bundle


def _clear(bundle: Path) -> None:
    """Убирает прежний ярлык, убедившись, что это именно он."""
    if not bundle.exists():
        return
    if not (bundle / "Contents" / "MacOS" / "launcher").exists():
        raise RuntimeError(
            f"{bundle} существует, но не похож на наш ярлык — удали его сам, "
            "чтобы ничего чужого не пропало"
        )
    shutil.rmtree(bundle)
