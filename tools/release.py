"""Builds the macOS application and the zip that is published as a release.

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

# The zip keeps an ASCII name even though the bundle inside does not: it becomes
# a link on a release page, and a Cyrillic file name there turns into percent
# escapes that no one can read or type.
ARCHIVE_STEM = "Transcripter-{version}-macos"

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

# The `app` extra is what brings pywebview in. Without it everything installs,
# the launch reports no error, and no window ever opens.
install() {
  "$RESOURCES/uv" pip install --python "$RUNTIME/bin/python" "$@" "$WHEEL[app]"
}

if [ ! -x "$RUNTIME/bin/python" ]; then
  # The size is named before anything is fetched, and Cancel is a real answer:
  # an application has no business spending someone's gigabytes unasked.
  osascript > /dev/null 2>&1 <<'ASK' || exit 0
display dialog "Transcripter sets up its environment before the first launch:
about 1.7 GB of packages, five to ten minutes. Nothing else is needed —
ffmpeg and the speaker models come with it.

The speech models are separate, about 3 GB, and the application asks
before downloading those too.

Recordings and transcripts never leave this computer. The only thing
sent anywhere is a question to GitHub about newer versions, and that
can be switched off." ¬
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


def version() -> str:
    """The one written in pyproject: the plist, the wheel and the zip share it."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def uv() -> Path:
    """The uv binary to put inside the bundle.

    Taken from this machine rather than downloaded: it is the very copy the
    project is built with, so what a person gets is what was tested here.
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


def assemble(into: Path, wheel: Path, release: str) -> Path:
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
    launcher.write_text(_LAUNCHER, encoding="utf-8")
    launcher.chmod(0o755)

    shutil.copy2(ASSETS / "icon.icns", resources / "icon.icns")
    shutil.copy2(wheel, resources / wheel.name)

    binary = uv()
    shutil.copy2(binary, resources / "uv")
    (resources / "uv").chmod(0o755)
    # uv is redistributed under either of two licences; carrying the texts along
    # is the whole of what both of them ask in return.
    for name in ("LICENSE-APACHE", "LICENSE-MIT"):
        licence = binary.parent.parent / name
        if licence.exists():
            shutil.copy2(licence, resources / f"uv-{name}")

    return bundle


def archive(bundle: Path, target: Path) -> Path:
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


def main() -> None:
    release = version()
    out = ROOT / "dist"
    staging = out / "staging"

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    wheel = build_wheel(staging / "wheel")
    bundle = assemble(staging, wheel, release)
    zipped = archive(bundle, out / f"{ARCHIVE_STEM.format(version=release)}.zip")

    # The wheel goes up as an asset of its own. An installed copy updates itself
    # by it — a quarter of a megabyte instead of twenty-one, and no replacing a
    # running application with itself. The zip stays for those arriving for the
    # first time.
    published = out / wheel.name
    shutil.copy2(wheel, published)

    print(f"бандл   {bundle}")
    for asset in (zipped, published):
        size = asset.stat().st_size / 1024 / 1024
        print(f"ассет   {asset}")
        print(f"        {size:.1f} МБ · sha256 {hashlib.sha256(asset.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
