"""Keeping yt-dlp as new as YouTube demands, and its solver in step with it.

YouTube closes the paths an old yt-dlp takes, and it does not wait for our
releases. Lecture M6QCn9SCaGQ answered "HTTP Error 403: Forbidden" on 2026.7.4
and came down whole on 2026.8.19 — that measurement is the lower bound in
`pyproject.toml`, and a lower bound helps only an installation made after it was
written. So when the window opens PyPI is asked about yt-dlp, and a newer one is
installed at once rather than offered: nobody wants the old one, and a person
looking at a 403 has no way to know that this is the button to press.

Unlike our own update this needs no restart, provided yt-dlp has not been
imported yet — `ingest.youtube` imports it lazily, at the first link. That is
also why the window starts no transcription while the files are being replaced:
an import in the middle would read half of each version.

Two packages move together. The script that solves YouTube's JavaScript
challenge comes as `yt-dlp-ejs`, and yt-dlp takes one only when its major and
minor version match what it was built against. Asked of yt-dlp's own Deno
provider on 25.09.2026: 2026.8.19 beside solver 0.5.0 answered "core script
version 0.5.0 is not supported" and found no lib script at all — its fallback
wants npm packages from the network — while beside 0.8.0 it took both halves
from the package. Not every link asks for the challenge (that day the test
lecture came through a client that did not), but one that does gets nothing but
storyboard pictures without a solver. yt-dlp pins the exact version in its
`default` extra, so the pin is read from the same answer that names the release,
and both are installed in one go. Not the extra itself: it brings requests,
websockets and four more packages this project has never needed.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from .updates import TIMEOUT, newer, parse, pip_install

FEED = "https://pypi.org/rss/project/yt-dlp/releases.xml"
"""The last forty uploads of yt-dlp — the one question asked on an ordinary launch.

1.1 KB over the wire and 567 ms, against 264 KB and 834 ms for the full project
document, which carries the whole release history and the README to answer the
same question (medians of five, 25.09.2026). Forty is enough: in the two years
before that date at most 26 nightly builds came between two releases.
"""

RELEASE = "https://pypi.org/pypi/yt-dlp/{version}/json"
"""One release: which solver it pins, and whether it was withdrawn.

52 KB, fetched only for a version newer than ours. Withdrawn releases do occur —
six so far, all from 2021 — and the feed does not mark them.
"""

PROJECT = "https://pypi.org/pypi/yt-dlp/json"
"""Everything PyPI knows about yt-dlp, asked only when the feed holds no release.

Its `info.version` leaves out nightly builds and withdrawn releases by itself,
so it answers what the feed cannot — at the price of 264 KB.
"""

SOLVER = "yt-dlp-ejs"

# An exact pin of the solver, however the name is spelled: `==` is how yt-dlp
# writes it, and anything looser is not a version one could install.
_PINNED_SOLVER = re.compile(r"^\s*yt[-_.]dlp[-_.]ejs\s*==\s*([\w.]+)", re.IGNORECASE)


@dataclass(frozen=True)
class YtDlpRelease:
    """A published yt-dlp and the solver version it was built against."""

    version: str
    solver: str | None = None
    """None when the release pins none: it then goes in alone, and the solver stays."""

    def requirements(self) -> list[str]:
        """What uv is asked for — both exactly, so that neither drifts from the other."""
        pinned = [f"yt-dlp=={self.version}"]
        if self.solver:
            pinned.append(f"{SOLVER}=={self.solver}")
        return pinned


def installed() -> str:
    """The yt-dlp version in our environment, "0" when there is none to speak of.

    Read from the package metadata rather than from `yt_dlp.version`: importing
    yt-dlp in order to ask would load the very modules about to be replaced, and
    this process would then keep the old ones until a restart.

    Missing and broken both count as "0", so that anything published is newer.
    An installation cut short by a closed window leaves exactly that, and the
    next launch should mend it rather than stumble over it.
    """
    try:
        return version("yt-dlp") or "0"
    except PackageNotFoundError:
        return "0"


def outdated() -> YtDlpRelease | None:
    """The release to install, when PyPI has one newer than ours.

    Newer, not merely different: a nightly build someone put in on purpose is
    later than the release before it, and going back to that release would be a
    downgrade done behind their back.

    An ordinary launch asks the feed and nothing else, since nothing in it is
    newer. A release worth installing is then looked up on its own, for the
    solver it pins and in case it was withdrawn; a withdrawn one gives way to the
    release before it, as long as that is still newer than ours.
    """
    current = installed()
    releases = _releases_in_feed()
    if releases is None:
        return None
    if not releases:
        # Forty nightly builds in a row without a release among them has not
        # happened in two years, but the full document knows the answer anyway.
        return _newest_of_all(current)

    for candidate in releases:
        if not newer(candidate, current):
            return None
        info = _info(RELEASE.format(version=candidate))
        if info is None:
            # Unanswered rather than withdrawn: the releases before it would
            # only wait for the same silence, one timeout each.
            return None
        if not info.get("yanked"):
            return _described(candidate, info)
    return None


def _answer(url: str):
    """PyPI's answer, or None for every ordinary silence.

    No network, a status other than 200, a service having a bad minute — the
    same ground as for our own update. Nobody asked for this check, so its
    failure is not worth a word in the window.
    """
    import httpx

    try:
        answer = httpx.get(url, timeout=TIMEOUT)
    except httpx.HTTPError:
        return None
    return answer if answer.status_code == 200 else None


def _releases_in_feed() -> list[str] | None:
    """The releases the feed lists, newest first; None when it could not be read.

    Nightly builds are told apart by their shape: a release is numbers and dots,
    a nightly ends in `.devN`. Sorted rather than taken in feed order, which is
    the order of uploading.

    The standard parser rather than `defusedxml`, and not for want of trying the
    attacks: it expands no external entities, and Expat has refused billion laughs
    since 2.4.1 ("limit on input amplification factor", tried on 2.6.3 and 2.8.1).
    """
    from xml.etree import ElementTree

    answer = _answer(FEED)
    if answer is None:
        return None
    try:
        items = ElementTree.fromstring(answer.content).iter("item")
        titles = {(item.findtext("title") or "").strip() for item in items}
    except ElementTree.ParseError:
        return None
    releases = [title for title in titles if title and all(p.isdigit() for p in title.split("."))]
    return sorted(releases, key=parse, reverse=True)


def _info(url: str) -> dict | None:
    """The `info` part of a PyPI description, or None when there is none to read."""
    answer = _answer(url)
    if answer is None:
        return None
    try:
        info = answer.json()["info"]
    except (ValueError, KeyError, TypeError):
        return None
    return info if isinstance(info, dict) else None


def _newest_of_all(current: str) -> YtDlpRelease | None:
    """The newest release by the full document, when the feed could not name one."""
    info = _info(PROJECT)
    if info is None:
        return None
    latest = str(info.get("version") or "")
    return _described(latest, info) if newer(latest, current) else None


def _described(version: str, info: dict) -> YtDlpRelease:
    """The release with the solver its requirements pin, if they pin one."""
    for requirement in info.get("requires_dist") or ():
        if pinned := _PINNED_SOLVER.match(str(requirement)):
            return YtDlpRelease(version=version, solver=pinned.group(1))
    return YtDlpRelease(version=version)


def install(release: YtDlpRelease, *, timeout: float = 300.0) -> None:
    """Puts the release in place of the installed one, the solver along with it.

    The timeout is sized for the download: a yt-dlp wheel is 3.2 MB, which a line
    of 256 kbit/s brings in about a hundred seconds.
    """
    pip_install(*release.requirements(), what=f"yt-dlp {release.version}", timeout=timeout)
    # The next import has to find the new files, not a directory listing cached
    # before they existed.
    importlib.invalidate_caches()


def imported() -> bool:
    """Whether this process already runs a yt-dlp — then the new one waits for a restart.

    A module once imported stays what it was, whatever lies on disk afterwards.
    Dropping it from `sys.modules` would force a fresh import, but yt-dlp puts
    finders of its own into `sys.meta_path` on the way in, and unpicking another
    library's state is exactly the kind of patch that misses.
    """
    return "yt_dlp" in sys.modules
