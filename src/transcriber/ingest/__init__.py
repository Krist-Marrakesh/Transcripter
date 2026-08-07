"""Ingest: bringing any source to WAV, 16 kHz, mono."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..cache import ArtifactCache, fingerprint, stable_key
from . import media, youtube

Notify = Callable[[str], None]
Advance = Callable[[float], None]
"""How far the long step has got — here that is the download."""

# How many times to re-fetch a truncated download before working with whatever
# arrived. A break is the network's doing and usually cures itself on a retry; but
# when the source lies about the duration — common with recorded livestreams —
# retrying achieves nothing.
DOWNLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class Source:
    """The pipeline's input, normalised."""

    audio: Path
    """WAV, 16 kHz, mono — what every step after this one works on."""

    origin: str
    """The original path or URL, for metadata and output file names."""

    title: str
    duration: float


def prepare(
    target: str,
    cache: ArtifactCache,
    notify: Notify | None = None,
    advance: Advance | None = None,
) -> Source:
    """Brings a file or a link to the pipeline's format.

    The result is cached by the content of the source: running again on the same
    file does not decode it a second time.

    Progress is reported for the download only. A local file has one step of any
    length — the transcoding — and the only way ffmpeg reports its progress is by
    parsing its own output.
    """
    say = notify or (lambda _: None)
    if youtube.is_url(target):
        return _prepare_url(target, cache, say, advance)
    return _prepare_file(Path(target).expanduser().resolve(), cache, say)


def _prepare_file(path: Path, cache: ArtifactCache, say: Notify) -> Source:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    key = stable_key(fingerprint(path), step="wav16k")
    wav = cache.reserve("audio", key, ".wav")
    if not wav.exists():
        media.extract_audio(path, wav)

    duration = media.probe(wav).duration
    # A local file cannot be fetched again, so this is a warning and nothing more.
    # The check runs even with the WAV ready: the shortfall shows on the source,
    # not on the result.
    if warning := media.truncation_warning(media.probe(path).duration, duration):
        say(warning)

    return Source(audio=wav, origin=str(path), title=path.stem, duration=duration)


def _prepare_url(
    url: str, cache: ArtifactCache, say: Notify, advance: Advance | None = None
) -> Source:
    info = youtube.probe(url)

    # Keyed by URL rather than by content: downloading a file to fingerprint it,
    # in order to find out it was already downloaded, is a circle.
    key = stable_key(url, step="wav16k")
    wav = cache.reserve("audio", key, ".wav")

    # A truncated download may have settled in the cache on an earlier run, so the
    # finished WAV is checked too — otherwise a clipped lecture stays there forever.
    if not wav.exists() or media.truncation_warning(info.duration, media.probe(wav).duration):
        _download_audio(url, wav, info.duration, cache, key, say, advance)

    return Source(
        audio=wav,
        origin=url,
        title=info.title,
        # The actual duration, not the declared one: on a truncated recording the
        # declared one promises what the audio does not contain, and the transcript
        # would come out "two hours long".
        duration=media.probe(wav).duration,
    )


def _download_audio(
    url: str,
    wav: Path,
    declared: float,
    cache: ArtifactCache,
    key: str,
    say: Notify,
    advance: Advance | None = None,
) -> None:
    """Downloads and converts, trying again when the audio arrived incomplete."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        staging = cache.reserve("downloads", key, "")
        # Leftovers of an interrupted download would confuse the choice of file —
        # start from a clean place and clear up afterwards either way.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            raw = youtube.download_audio(url, staging, advance)
            media.extract_audio(raw, wav)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        warning = media.truncation_warning(declared, media.probe(wav).duration)
        if warning is None:
            return
        if attempt < DOWNLOAD_ATTEMPTS:
            say(f"{warning}; downloading again, attempt {attempt + 1} of {DOWNLOAD_ATTEMPTS}")
        else:
            say(warning)
