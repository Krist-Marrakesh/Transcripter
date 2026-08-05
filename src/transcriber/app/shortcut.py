"""Building the shortcut the application is started from.

The result is thin: what it holds is not a copy of the environment but a call to
the interpreter the project was built with. Full packaging would be harmful here
— it would drag torch and the model weights inside, gigabytes of them, for the
sake of a double click. The release bundle in `tools/release.py` solves the other
problem, of handing the application to someone who has no project folder at all.

The honest downside: the shortcut is tied to the project folder. Move the project
and the shortcut has to be built again.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ..subproc import quiet_flags

ASSETS = Path(__file__).parent / "assets"
BUNDLE_NAME = "Транскрибатор.app"
LINK_NAME = "Транскрибатор.lnk"

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

# Where Homebrew puts binaries on either chip. Finder starts an application with
# a bare PATH — no shell profile is ever read — so ffmpeg looks missing even on a
# machine where it has been installed for years. The message then blames the
# person for something they already did.
BREW_PATHS = "/opt/homebrew/bin:/usr/local/bin"

_LAUNCHER = """#!/bin/sh
# The shortcut does not copy the environment, it calls the one the project was
# built with. Finder gives no terminal, so output goes to a log: without it any
# error looks like "the icon bounced and nothing happened".
export PATH="{brew}:$PATH"
LOG="$HOME/Library/Logs/Транскрибатор.log"
mkdir -p "$(dirname "$LOG")"
{{
  echo "--- запуск $(date '+%Y-%m-%d %H:%M:%S') ---"
  # The architecture is named explicitly. Python is universal, and for a script
  # bundle LaunchServices picks the x86_64 slice, while the compiled packages are
  # built for {arch} — from Finder that looked like "the icon bounced and died".
  exec /usr/bin/arch -{arch} "{python}" -m transcriber.cli app
}} >> "$LOG" 2>&1
"""

# WScript.Shell is the only way to write a .lnk without extra packages: the
# format is undocumented and pywin32 would be a dependency for one call.
_LINK = """$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut('{path}')
$link.TargetPath = '{python}'
$link.Arguments = '-m transcriber.cli app'
$link.WorkingDirectory = '{workdir}'
$link.Description = 'Транскрибатор'
{icon}$link.Save()
"""


def create(destination: Path, *, version: str = "0.1.0") -> Path:
    """Builds a shortcut in the given folder and returns the path to it."""
    destination = destination.expanduser().resolve()
    if sys.platform == "darwin":
        return _bundle(destination, version)
    if sys.platform == "win32":
        return _link(destination)
    raise RuntimeError("ярлык собирается только на macOS и Windows")


def _bundle(destination: Path, version: str) -> Path:
    bundle = destination / BUNDLE_NAME
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
        _LAUNCHER.format(python=sys.executable, arch=platform.machine(), brew=BREW_PATHS),
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    icon = ASSETS / "icon.icns"
    if icon.exists():
        shutil.copy2(icon, resources / "icon.icns")

    return bundle


def _link(destination: Path) -> Path:
    link = destination / LINK_NAME
    if link.is_dir():
        raise RuntimeError(f"{link} — папка, а не ярлык: убери её сам, чтобы ничего не пропало")

    icon = ASSETS / "icon.ico"
    script = _LINK.format(
        path=_quote(link),
        python=_quote(_windowed()),
        workdir=_quote(Path.home()),
        # The index picks an image inside the file; ours is the only group there.
        icon=f"$link.IconLocation = '{_quote(icon)},0'\n" if icon.exists() else "",
    )

    done = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
        **quiet_flags(),
    )
    if done.returncode != 0:
        reason = done.stderr.strip().splitlines()[-3:]
        raise RuntimeError("\n".join(["не удалось собрать ярлык", *reason]))
    return link


def _windowed() -> Path:
    """The interpreter without a console.

    `python.exe` would keep a black window open behind the interface for as long
    as the application runs; `pythonw.exe` is the same interpreter built as a
    windowed program and lives beside it.
    """
    windowed = Path(sys.executable).with_name("pythonw.exe")
    return windowed if windowed.exists() else Path(sys.executable)


def _quote(path: Path) -> str:
    """A path inside PowerShell's single quotes, where a quote is doubled."""
    return str(path).replace("'", "''")


def _clear(bundle: Path) -> None:
    """Removes a previous shortcut, having made sure it is ours."""
    if not bundle.exists():
        return
    if not (bundle / "Contents" / "MacOS" / "launcher").exists():
        raise RuntimeError(
            f"{bundle} существует, но не похож на наш ярлык — удали его сам, "
            "чтобы ничего чужого не пропало"
        )
    shutil.rmtree(bundle)
