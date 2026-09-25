"""Builds the application and the archive that is published as a release.

The difference from `transcriber.app.shortcut` is who the result is for. The
shortcut is for this machine: it calls the interpreter the project was built
with, and moving the project breaks it. What is published has to work for someone
who has no project folder and no environment — and, most likely, no Python.

The way out is not to freeze everything inside. A frozen bundle of this project
is about a gigabyte and a half of libraries, of which torch alone is half, and it
would still leave for the model weights on first use — the download does not go
away, it only moves. So the bundle carries a wheel of our package and `uv`, and
builds the environment on the machine it lands on: ten megabytes to hand over
instead of nine hundred, and a native Python instead of one frozen for the
architecture the build happened to run on.

**It builds for the system it is standing on**, and that is not a limitation
waiting to be lifted. The one thing the archive carries besides our wheel is a
copy of `uv`, and `uv` is a native binary: a Mac has no Windows one to put in,
short of downloading a stranger's build and hoping. So the Windows archive is
assembled on Windows — in CI, by the same runner that has just run the tests on
it.

Run from the project root:

    .venv/bin/python tools/release.py
"""

from __future__ import annotations

import hashlib
import plistlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "src" / "transcriber" / "app" / "assets"

BUNDLE_NAME = "Транскрибатор.app"
FOLDER_NAME = "Транскрибатор"

# The archive keeps an ASCII name even though what is inside it does not: it
# becomes a link on a release page, and a Cyrillic file name there turns into
# percent escapes that no one can read or type.
ARCHIVE_STEM = "Transcripter-{version}-{system}"

# What the wheel is installed with. `app` brings pywebview in; without it
# everything installs, the launch reports no error, and no window ever opens.
# `documents` brings what writes PDF and DOCX — without it those two buttons
# failed for everyone but the developer, who had the libraries installed by hand.
#
# Written here rather than in each launcher, and handed down to the window as
# well: the update inside the window installs the package a second time, and if
# it asked for a smaller set than the launcher did, the first dependency added to
# `documents` would go missing on every machine that updated without noticing.
EXTRAS = "app,documents"

# Substituted into both launchers. A token rather than `str.format`, because both
# scripts are full of braces and dollars of their own.
_SLOT = "@EXTRAS@"

_PLIST = {
    "CFBundleName": "Транскрибатор",
    "CFBundleDisplayName": "Транскрибатор",
    "CFBundleIdentifier": "local.transcriber.app",
    "CFBundleExecutable": "launcher",
    "CFBundleIconFile": "icon",
    "CFBundlePackageType": "APPL",
    "NSHighResolutionCapable": True,
    "LSMinimumSystemVersion": "11.0",
}

# What the launcher does on first run, and why it is a shell script rather than
# Python: at that moment there is no Python on the machine yet that we may use.
#
# The architecture is not pinned here, unlike the developer shortcut. There the
# `arch` call fixed a real problem — LaunchServices picks the x86_64 slice of a
# universal Python for a script bundle, while the compiled packages are built for
# one chip. Here uv installs an interpreter for the machine it is standing on, so
# there is no universal binary left to choose the wrong half of, and a pinned
# architecture would only make the bundle fail on an Intel Mac.
_LAUNCHER = """#!/bin/sh
# Finder gives an application a bare PATH — the shell profile is never read — so
# Homebrew's folder is missing and ffmpeg looks uninstalled on a machine that has
# had it for years.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

RESOURCES="$(cd "$(dirname "$0")/../Resources" && pwd)"
RUNTIME="$HOME/Library/Application Support/transcript/runtime"
LOG="$HOME/Library/Logs/Транскрибатор.log"

# The environment is built by `uv venv` and has no pip in it, so this binary is
# the only way anything can be installed later. A Python process cannot work out
# where the bundle is — `sys.prefix` points at the environment, somewhere else
# entirely — so the path is handed down rather than searched for.
export TRANSCRIPT_UV="$RESOURCES/uv"

# Model weights live inside the application, so that dragging it to the Trash
# takes its twenty-odd gigabytes along. Nothing else deletes them together: a
# shared HuggingFace cache outlives every application that ever wrote to it.
export TRANSCRIPT_MODELS="$RESOURCES/models"

# The set below, so that an update installed from inside the window asks for
# exactly what this launcher asked for.
EXTRAS="@EXTRAS@"
export TRANSCRIPT_EXTRAS="$EXTRAS"

mkdir -p "$(dirname "$LOG")"
# No terminal anywhere behind us: without a log every failure looks the same,
# like the icon bounced once and nothing happened.
exec >> "$LOG" 2>&1
echo "--- запуск $(date '+%Y-%m-%d %H:%M:%S') ---"

say() {
  osascript -e "display alert \\"Transcripter\\" message \\"$1\\"" > /dev/null 2>&1
}

note() {
  osascript -e "display notification \\"$1\\" with title \\"Transcripter\\"" > /dev/null 2>&1
}

# Until a person moves a downloaded application in Finder, macOS runs it from a
# randomised read-only copy of itself. Weights are kept inside the bundle, and in
# that copy there is nothing to keep them in — it is thrown away at exit, and
# writing to it fails anyway. Better to say so than to fail later and elsewhere.
case "$RESOURCES" in
  */AppTranslocation/*)
    echo "запущено из AppTranslocation: $RESOURCES"
    say "Move Транскрибатор to Applications in Finder, then open it again."
    exit 1;;
esac

WHEEL="$(ls "$RESOURCES"/transcript-*.whl 2>/dev/null | head -1)"
if [ -z "$WHEEL" ]; then
  say "The application is damaged: the package is missing from it. Download it again."
  exit 1
fi

# What is installed, recorded as version and content. The version decides which
# of the two is newer; the hash catches a rebuild that kept the same number, which
# is every rebuild during development.
WHEEL_VERSION="$(basename "$WHEEL" | cut -d- -f2)"
WANTED="$WHEEL_VERSION|$(shasum -a 256 "$WHEEL" | cut -d' ' -f1)"
STAMP="$RUNTIME/.installed"
HAVE="$(cat "$STAMP" 2>/dev/null || true)"
HAVE_VERSION="${HAVE%%|*}"

# Did the application update itself through the window? Then what is installed is
# newer than what this bundle carries, and reinstalling would be a downgrade —
# done silently, on every launch, undoing what a person asked for.
self_updated() {
  [ -n "$HAVE_VERSION" ] &&
  [ "$HAVE_VERSION" != "$WHEEL_VERSION" ] &&
  [ "$(printf '%s\n%s\n' "$WHEEL_VERSION" "$HAVE_VERSION" | sort -V | tail -1)" = "$HAVE_VERSION" ]
}

install() {
  "$RESOURCES/uv" pip install --python "$RUNTIME/bin/python" "$@" "$WHEEL[$EXTRAS]"
}

if [ ! -x "$RUNTIME/bin/python" ]; then
  # The size is named before anything is fetched, and Cancel is a real answer:
  # an application has no business spending someone's gigabytes unasked.
  osascript > /dev/null 2>&1 <<'ASK' || exit 0
display dialog "Transcripter sets up its environment before the first launch:
about 1.7 GB of packages, five to ten minutes. Nothing else is needed —
ffmpeg and the speaker models come with it.

The speech models are separate, about 3 GB, and the application asks
before downloading those too. They are kept inside the application
itself, so moving it to the Trash takes them with it.

Recordings and transcripts never leave this computer. The only things
sent anywhere are questions about newer versions — to GitHub about the
application, to PyPI about yt-dlp — and they can be switched off." ¬
  with title "Transcripter" ¬
  buttons {"Cancel", "Install"} default button "Install" ¬
  cancel button "Cancel" with icon note
ASK

  note "Setting up the environment, five to ten minutes…"
  rm -rf "$RUNTIME"

  # `only-managed` — uv fetches an interpreter of its own instead of borrowing
  # whatever 3.13 happens to be installed. Fifty megabytes more on the first run,
  # and in exchange the application stops depending on someone else's package
  # manager: a Homebrew python that gets uninstalled would otherwise take the
  # environment with it, and only one of the two paths was ever tested here.
  if ! "$RESOURCES/uv" venv --python 3.13 --python-preference only-managed "$RUNTIME"; then
    say "Could not create the environment. Details: ~/Library/Logs/Транскрибатор.log"
    exit 1
  fi

  if ! install; then
    say "Could not install the packages. Details: ~/Library/Logs/Транскрибатор.log"
    exit 1
  fi

  printf '%s' "$WANTED" > "$STAMP"
  note "Ready, starting up"

elif [ "$HAVE" != "$WANTED" ] && ! self_updated; then
  # A new build over a working environment. Only our own package is replaced —
  # asking again about 1.3 GB would be a lie, since the other 125 packages are
  # already there and a resolve leaves them alone.
  #
  # `--reinstall-package` is belt and braces. The version number stays the same
  # between builds while the contents do not, and uv was measured to notice that
  # by itself — but the noticing is a caching heuristic, and this does not depend
  # on it. Reinstalling one package that is already unpacked costs milliseconds.
  note "Updating"
  if ! install --reinstall-package transcript; then
    say "Could not update. Details: ~/Library/Logs/Транскрибатор.log"
    exit 1
  fi
  printf '%s' "$WANTED" > "$STAMP"
fi

exec "$RUNTIME/bin/python" -m transcriber.cli app
"""

# Two lines of batch and nothing more. Everything a launcher has to do here —
# compare versions, hash a file, ask a question — is a paragraph of PowerShell and
# a minefield in `cmd`, where a mistyped block fails silently and takes the whole
# launch with it. The `.cmd` exists only because a `.ps1` cannot be started by
# double-clicking it.
_LAUNCHER_CMD = """@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher.ps1"
"""

# ASCII only, deliberately. Windows PowerShell reads a script without a byte order
# mark in the system's ANSI code page, and a Cyrillic message in it would arrive
# as rubbish on every machine whose code page is not the author's.
#
# Deliberate and unenforced is how it stayed broken: two em dashes crept into the
# comments, `write_text(encoding="ascii")` refused them, and the Windows archive
# could not be built at all. A dash is not a message and nobody would have looked
# for one — hence the test that now reads this very string.
_LAUNCHER_PS1 = """# Everything this application owns lives beside this file: the environment, the
# model weights, the interpreter uv fetches and uv's own package cache. Deleting
# the folder therefore deletes all of it, which is the whole point of keeping it
# here instead of under AppData, where nothing is ever deleted with anything.

# Not "Stop", and this is not an oversight. With `Stop` in force, a native
# program whose stderr is redirected into the pipeline turns every line it writes
# there into a terminating error, and uv writes its progress to stderr. The
# install would break on the first line of output it printed. Exit codes are
# checked below, one call at a time, which is the thing that actually matters.
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
$runtime = Join-Path $here "runtime"
$python = Join-Path $runtime "Scripts\\python.exe"
$windowed = Join-Path $runtime "Scripts\\pythonw.exe"
$uv = Join-Path $here "uv.exe"
$log = Join-Path $here "transcript.log"
$stamp = Join-Path $runtime ".installed"

# Handed down rather than searched for: a Python process cannot work out where
# this folder is, since `sys.prefix` points at the environment inside it.
$env:TRANSCRIPT_UV = $uv
$env:TRANSCRIPT_MODELS = Join-Path $here "models"
$env:TRANSCRIPT_EXTRAS = "@EXTRAS@"

# uv keeps its interpreters and its downloaded packages under LOCALAPPDATA unless
# told otherwise, and both would outlive the folder they were fetched for.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $here "python"
$env:UV_CACHE_DIR = Join-Path $here "uv-cache"

function Write-Log($text) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $text" | Out-File $log -Append -Encoding utf8
}

function Fail($text) {
    Write-Log "ERROR: $text"
    Write-Host ""
    Write-Host "  $text" -ForegroundColor Red
    Write-Host "  Details: $log"
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

function Invoke-Uv($arguments) {
    Write-Log "uv $($arguments -join ' ')"
    # `Out-Host` at the end, and it is load-bearing. Tee-Object passes what it
    # writes on down the pipeline, and anything a function leaves in the pipeline
    # becomes part of what it returns, and the caller would get several hundred lines
    # of uv output with a boolean at the end of them, and every check below would
    # read as true.
    & $uv @arguments 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    return $LASTEXITCODE -eq 0
}

Write-Log "--- launch ---"

$wheel = Get-ChildItem -Path (Join-Path $here "transcript-*.whl") -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $wheel) {
    Fail "The application is damaged: the package is missing from it. Download it again."
}

# What is installed, recorded as version and content. The version decides which of
# the two is newer; the hash catches a rebuild that kept the same number, which is
# every rebuild during development.
$wheelVersion = $wheel.Name.Split('-')[1]
$wanted = "$wheelVersion|$((Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower())"
$have = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
$haveVersion = $have.Split('|')[0]

# Did the application update itself through the window? Then what is installed is
# newer than what this folder carries, and reinstalling would be a downgrade, done
# silently on every launch, undoing what a person asked for.
$selfUpdated = $false
if ($haveVersion -and $haveVersion -ne $wheelVersion) {
    try { $selfUpdated = [version]$haveVersion -gt [version]$wheelVersion } catch { }
}

# Not `Install-Package`: PowerShell already has a command by that name, and a
# function shadowing it is a trap for whoever reads this next.
function Install-Ours($extra) {
    $arguments = @("pip", "install", "--python", $python) + $extra +
        @("$($wheel.FullName)[$($env:TRANSCRIPT_EXTRAS)]")
    return Invoke-Uv $arguments
}

if (-not (Test-Path $python)) {
    Write-Host ""
    Write-Host "  Transcripter sets up its environment before the first launch:"
    Write-Host "  about half a gigabyte to download, a few minutes to unpack."
    Write-Host "  Nothing else is needed - ffmpeg and the speaker models come with it."
    Write-Host ""
    Write-Host "  The speech models are separate, about 3 GB, and the application"
    Write-Host "  asks before downloading those too. All of it is kept in this"
    Write-Host "  folder, so deleting the folder leaves nothing behind."
    Write-Host ""
    Write-Host "  Recordings and transcripts never leave this computer. The only"
    Write-Host "  things sent anywhere are questions about newer versions - to GitHub"
    Write-Host "  about the application, to PyPI about yt-dlp - and they can be"
    Write-Host "  switched off."
    Write-Host ""
    if ((Read-Host "  Install? [Y/n]") -match '^[nN]') { exit 0 }

    Write-Host ""
    Write-Host "  Setting up the environment. This takes a few minutes."
    Remove-Item $runtime -Recurse -Force -ErrorAction SilentlyContinue

    # `only-managed` - uv fetches an interpreter of its own instead of borrowing
    # whatever 3.13 happens to be on the machine. Fifty megabytes more on the
    # first run, and in exchange the application stops depending on a Python
    # somebody else installed and may uninstall.
    if (-not (Invoke-Uv @("venv", "--python", "3.13", "--python-preference", "only-managed",
                          $runtime))) {
        Fail "Could not create the environment."
    }

    # torch first, and from whichever index matches the card in this machine.
    # `auto` reads the installed NVIDIA driver and picks the CUDA channel for it,
    # falling back to the CPU build where there is no driver. Done before the
    # package rather than after, so that the CPU build is not downloaded and then
    # thrown away - and only torch and torchaudio are asked for, because the CUDA
    # channel has no Windows build of torchcodec at all.
    if (-not (Invoke-Uv @("pip", "install", "--python", $python, "--torch-backend", "auto",
                          "torch", "torchaudio"))) {
        Write-Log "torch from the matching index failed; the package will bring its own"
    }

    if (-not (Install-Ours @())) { Fail "Could not install the packages." }

    $wanted | Out-File $stamp -NoNewline -Encoding ascii
    Write-Host "  Ready, starting up."
}
elseif ($have -ne $wanted -and -not $selfUpdated) {
    # A new build over a working environment. Only our own package is replaced:
    # asking again about two gigabytes would be a lie, since the rest is already
    # there and a resolve leaves it alone.
    Write-Host "  Updating."
    if (-not (Install-Ours @("--reinstall-package", "transcript"))) {
        Fail "Could not update."
    }
    $wanted | Out-File $stamp -NoNewline -Encoding ascii
}

# `pythonw` and not `python`: the same interpreter built as a windowed program,
# which is what stops a black console from standing behind the interface for as
# long as the application runs. Started detached, so this console closes now.
$folder = [Environment]::GetFolderPath("UserProfile")
$run = @("-m", "transcriber.cli", "app")
Start-Process -FilePath $windowed -ArgumentList $run -WorkingDirectory $folder
"""


def version() -> str:
    """The one written in pyproject: the plist, the wheel and the archive share it."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def uv() -> Path:
    """The uv binary to put inside the archive.

    Taken from this machine rather than downloaded: it is the very copy the
    project is built with, so what a person gets is what was tested here. It is
    also why the archive can only be built on the system it is for.
    """
    found = shutil.which("uv")
    if found is None:
        raise SystemExit("uv не найден в PATH — без него нечего класть в бандл")
    return Path(found).resolve()


def build_wheel(into: Path) -> Path:
    """Builds the package wheel — the whole application, minus its dependencies."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(into)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(into.glob("transcript-*.whl"))
    if not wheels:
        raise SystemExit(f"колесо не собралось в {into}")
    return wheels[-1]


def _licences(binary: Path, into: Path) -> None:
    """Carries uv's licence texts along — the whole of what they ask in return."""
    for name in ("LICENSE-APACHE", "LICENSE-MIT"):
        licence = binary.parent.parent / name
        if licence.exists():
            shutil.copy2(licence, into / f"uv-{name}")


def assemble_macos(into: Path, wheel: Path, release: str) -> Path:
    """Lays out the .app: a launcher, an icon, uv and our wheel."""
    bundle = into / BUNDLE_NAME
    contents = bundle / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)

    (contents / "Info.plist").write_bytes(
        plistlib.dumps(_PLIST | {"CFBundleVersion": release, "CFBundleShortVersionString": release})
    )

    launcher = macos / "launcher"
    launcher.write_text(_LAUNCHER.replace(_SLOT, EXTRAS), encoding="utf-8")
    launcher.chmod(0o755)

    shutil.copy2(ASSETS / "icon.icns", resources / "icon.icns")
    shutil.copy2(wheel, resources / wheel.name)

    binary = uv()
    shutil.copy2(binary, resources / "uv")
    (resources / "uv").chmod(0o755)
    _licences(binary, resources)

    return bundle


def assemble_windows(into: Path, wheel: Path) -> Path:
    """Lays out the folder: what to double-click, what it runs, uv and our wheel.

    A plain folder rather than an installer. Everything the application makes —
    the environment, the weights, uv's own downloads — is written inside it, so
    the folder is the whole application: it can be moved to another disk, carried
    on a stick, and deleted without leaving anything behind.
    """
    folder = into / FOLDER_NAME
    folder.mkdir(parents=True)

    (folder / f"{FOLDER_NAME}.cmd").write_text(_LAUNCHER_CMD, encoding="ascii")
    (folder / "launcher.ps1").write_text(_LAUNCHER_PS1.replace(_SLOT, EXTRAS), encoding="ascii")

    shutil.copy2(ASSETS / "icon.ico", folder / "icon.ico")
    shutil.copy2(wheel, folder / wheel.name)

    binary = uv()
    shutil.copy2(binary, folder / "uv.exe")
    _licences(binary, folder)

    return folder


def archive_macos(bundle: Path, target: Path) -> Path:
    """Packs the bundle with `ditto`, not `zip`.

    `zip` loses the executable bit and mangles symlinks, and a bundle unpacked
    from such an archive does not start — the system refuses the launcher it can
    no longer run.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    subprocess.run(
        ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(target)],
        check=True,
    )
    return target


def archive_windows(folder: Path, target: Path) -> Path:
    """Packs the folder into a zip. Nothing here needs a permission bit kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    shutil.make_archive(str(target.with_suffix("")), "zip", folder.parent, folder.name)
    return target


def build(staging: Path, wheel: Path, release: str) -> tuple[Path, str]:
    """Assembles the application for the system this is running on."""
    if sys.platform == "darwin":
        return assemble_macos(staging, wheel, release), "macos"
    if sys.platform == "win32":
        return assemble_windows(staging, wheel), "windows"
    raise SystemExit(f"релиз собирается на macOS и Windows, а не на {sys.platform}")


def _say_it_in_full() -> None:
    """Позволяет напечатать то, что мы собрали, как оно называется.

    Приложение зовут «Транскрибатор», и путь к нему — единственное, что этот
    сценарий обязан сообщить. На Windows же вывод по умолчанию идёт кодовой
    страницей консоли, и в CI это `cp1252`, где кириллицы нет вовсе: сборка
    падала `UnicodeEncodeError` в самом конце — колесо собрано, папка разложена,
    архив готов, — и по такому отказу не заподозришь, что дело в печати.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.reconfigure(encoding="utf-8", errors="replace")


def built_from() -> str:
    """Что именно уехало в архив: коммит, и грязное ли дерево.

    Печатается, потому что архив живёт дольше памяти о нём. Ассеты 0.3.0 были
    собраны, затем родились ещё два коммита, а пересобрать я забыл — и в релизе
    оказалось два разных колеса под одной версией. Отличить их можно было только
    распаковав; строка ниже даёт ответ до публикации, а не после.

    Не отказ, а сообщение: собрать из грязного дерева — обычное дело при
    отладке, и запрещать это значило бы мешать работе. Врать о том, что собрано,
    нельзя, а собирать — можно.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # Сборка из распакованного архива, без гита. Сказать нечего — и не надо.
        return "не из репозитория"
    return f"{head} + несохранённые правки" if dirty else head


def main() -> None:
    _say_it_in_full()
    release = version()
    out = ROOT / "dist"
    staging = out / "staging"

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    wheel = build_wheel(staging / "wheel")
    application, system = build(staging, wheel, release)
    name = ARCHIVE_STEM.format(version=release, system=system)
    packer = archive_macos if system == "macos" else archive_windows
    zipped = packer(application, out / f"{name}.zip")

    # The wheel goes up as an asset of its own. An installed copy updates itself
    # by it — a quarter of a megabyte instead of twenty-one, and no replacing a
    # running application with itself. The archive stays for those arriving for
    # the first time.
    published = out / wheel.name
    shutil.copy2(wheel, published)

    print(f"версия      {release} · собрано из {built_from()}")
    print(f"приложение  {application}")
    for asset in (zipped, published):
        size = asset.stat().st_size / 1024 / 1024
        print(f"ассет   {asset}")
        print(f"        {size:.1f} МБ · sha256 {hashlib.sha256(asset.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
