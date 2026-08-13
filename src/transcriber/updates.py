"""Finding out that a newer version exists, and installing it.

What gets replaced is the package, not the application. Everything of ours lives
in one wheel — the window, the pipeline, the backends — while the bundle around
it holds a launcher and a copy of `uv` that change once a year. Downloading a
quarter of a megabyte instead of twenty-one, and never having to replace a
running program with itself, is worth the one thing it costs: a launcher whose
own wheel may be older than what is installed, so it must know not to overwrite.

This is the only place in the project that reaches the network for its own sake
rather than at a person's request. It can be switched off — `update_check` in the
settings — and it sends nothing but a GET.

Through httpx rather than the standard library, and not for the convenience.
`urllib` trusts the system certificate store, which a Python framework build on
macOS does not have — the request failed verification and the failure looked
exactly like "no update", forever and on every machine like this one. httpx
carries a certificate bundle of its own, and it is already a dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .subproc import interpreter, quiet_flags

LATEST = "https://api.github.com/repos/Krist-Marrakesh/Transcripter/releases/latest"

EXTRAS = "TRANSCRIPT_EXTRAS"
"""The variable in which the launcher names the extras it installed us with.

Named by it rather than repeated here. The two were written out separately once —
`[app]` in this file against `[app,documents]` in the launcher — and nothing broke
only because extras add and never take away: the first dependency added to
`documents` would have gone missing on every machine that updated from the window,
and only there.
"""

TIMEOUT = 10.0
"""A check nobody asked for must not make the window wait."""


class UpdateError(RuntimeError):
    """The update could not be found out about, downloaded or installed."""


@dataclass(frozen=True)
class Release:
    version: str
    wheel: str
    """A link to the package itself. The zip beside it is for new arrivals."""

    notes: str


def parse(value: str) -> tuple[int, ...]:
    """A version as numbers, for comparing.

    Deliberately without `packaging`: it arrives here as somebody else's
    dependency, and it is not worth declaring one of our own for versions shaped
    like `0.1.0`. Anything that is not a number ends the version — `1.2.0rc1`
    counts as `1.2`, which is close enough while pre-releases are not published.
    """
    numbers: list[int] = []
    for part in value.strip().removeprefix("v").split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def newer(candidate: str, than: str) -> bool:
    """Whether the first version is later than the second."""
    return parse(candidate) > parse(than)


def latest(url: str = LATEST) -> Release | None:
    """The published release, or `None` when there is nothing to say.

    `None` covers every ordinary silence: no releases yet, no network, GitHub
    rate-limiting an unauthenticated request. None of that is worth an error in
    the window — a check nobody asked for should fail invisibly.
    """
    import httpx

    try:
        answer = httpx.get(url, headers={"Accept": "application/vnd.github+json"}, timeout=TIMEOUT)
    except httpx.HTTPError:
        return None
    # 404 is the ordinary answer for a repository with no releases yet, and 403
    # is GitHub rate-limiting an unauthenticated request. Neither is news.
    if answer.status_code != 200:
        return None
    try:
        data = answer.json()
    except ValueError:
        return None

    # By suffix rather than by name: the file carries the version, and matching
    # the exact name would break on the first bump.
    wheels = [
        asset.get("browser_download_url", "")
        for asset in data.get("assets") or ()
        if str(asset.get("name", "")).endswith(".whl")
    ]
    tag = str(data.get("tag_name") or "").removeprefix("v")
    if not wheels or not tag:
        # A release without a package: there is something to announce but nothing
        # to install, and offering it would lead to a button that cannot work.
        return None

    return Release(version=tag, wheel=wheels[0], notes=str(data.get("body") or "").strip())


def available(current: str, url: str = LATEST) -> Release | None:
    """The release worth offering, if there is one."""
    found = latest(url)
    return found if found and newer(found.version, current) else None


def uv() -> str:
    """The uv binary that can install into our environment.

    The environment is built by `uv venv` and has no pip in it at all, so this is
    the only way to install anything. Inside a released application uv lies in
    the bundle, whose path a Python process has no way to work out — `sys.prefix`
    points at the environment, which is somewhere else entirely. So the launcher
    passes it down.
    """
    handed_over = os.environ.get("TRANSCRIPT_UV")
    if handed_over and Path(handed_over).exists():
        return handed_over
    # A run from a source checkout: uv is what the project was built with.
    found = shutil.which("uv")
    if found is None:
        raise UpdateError("uv was not found — there is nothing to install with")
    return found


def _asked_for(wheel: Path) -> str:
    """The wheel with the extras the launcher installed, in pip's own notation.

    Bare where nobody handed any down, which means a run from a source checkout:
    guessing a set there would either add packages a developer did not ask for or
    name a smaller one than is already installed.
    """
    extras = os.environ.get(EXTRAS)
    return f"{wheel}[{extras}]" if extras else str(wheel)


def stamp() -> Path:
    """Where the launcher records what it put into the environment.

    It reads the file on every start to decide whether the wheel it carries is
    worth installing. An update that did not write here would be rolled back to
    the bundled version on the very next launch, quietly and every time.
    """
    return Path(sys.prefix) / ".installed"


def install(release: Release, *, timeout: float = 600.0) -> None:
    """Downloads the wheel and puts it in place of the running one.

    Into a temporary file rather than straight from the link: uv would fetch it
    itself, but then a half-arrived package and a network failure become the same
    event, and the environment is left holding whichever it was.
    """
    from hashlib import sha256
    from tempfile import TemporaryDirectory

    import httpx

    with TemporaryDirectory() as folder:
        wheel = Path(folder) / release.wheel.rsplit("/", 1)[-1]
        try:
            # Release assets answer with a redirect to storage, so following one
            # is not optional here.
            answer = httpx.get(release.wheel, timeout=TIMEOUT, follow_redirects=True)
            answer.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpdateError(f"could not download the update: {exc}") from exc
        payload = answer.content
        wheel.write_bytes(payload)

        done = subprocess.run(
            [
                uv(),
                "pip",
                "install",
                # Our environment, not `sys.executable`: pywebview relaunches the
                # application through Python.app, and the update would land in a
                # stranger while ours stayed as it was.
                "--python",
                interpreter(),
                # The version differs, so a resolver would replace the package
                # anyway; being explicit costs nothing and depends on nothing.
                "--reinstall-package",
                "transcript",
                _asked_for(wheel),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **quiet_flags(),
        )

    if done.returncode != 0:
        reason = (done.stderr or "").strip().splitlines()[-3:]
        raise UpdateError("\n".join(["could not install the update", *reason]))

    # Only where the launcher keeps one. In a source checkout there is no such
    # file and inventing one would leave litter in a developer's environment.
    if (record := stamp()).exists():
        record.write_text(f"{release.version}|{sha256(payload).hexdigest()}", encoding="utf-8")
