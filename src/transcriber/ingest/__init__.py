"""Ingest: bringing any source to WAV, 16 kHz, mono."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..cache import ArtifactCache, building, fingerprint, stable_key
from ..report import Report
from . import media, youtube

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
    report: Report | None = None,
    *,
    cookies: youtube.Cookies | None = None,
) -> Source:
    """Brings a file or a link to the pipeline's format.

    The result is cached by the content of the source: running again on the same
    file does not decode it a second time.

    Progress is reported for the download only. A local file has one step of any
    length — the transcoding — and the only way ffmpeg reports its progress is by
    parsing its own output.
    """
    told = report or Report()
    if youtube.is_url(target):
        return _prepare_url(target, cache, told, cookies)
    return _prepare_file(Path(target).expanduser().resolve(), cache, told)


def _prepare_file(path: Path, cache: ArtifactCache, report: Report) -> Source:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    key = stable_key(fingerprint(path), step="wav16k")
    wav = cache.reserve("audio", key, ".wav")
    if not wav.exists():
        # Under a temporary name, because `wav.exists()` above is the only thing
        # standing between the next run and this file. Decoding cut short by a
        # stop or a crash would otherwise leave a shorter recording under the
        # right name, and every run after that would quietly transcribe the part.
        with building(wav) as partial:
            media.extract_audio(path, partial, report)

    duration = media.probe(wav).duration
    # A local file cannot be fetched again, so this is a warning and nothing more.
    # The check runs even with the WAV ready: the shortfall shows on the source,
    # not on the result.
    if warning := media.truncation_warning(media.probe(path).duration, duration):
        report.say(warning)

    return Source(audio=wav, origin=str(path), title=path.stem, duration=duration)


def _prepare_url(
    url: str, cache: ArtifactCache, report: Report, cookies: youtube.Cookies | None
) -> Source:
    info = youtube.probe(url, cookies=cookies)

    # Keyed by what the site calls the recording rather than by content:
    # downloading a file to fingerprint it, in order to find out it was already
    # downloaded, is a circle. And not by the address either — the same battle
    # arrived twice, once as `?t=3116s` and once as `?t=2940s`, and the second
    # time went to fetch 118 MB that were already on the disk.
    key = stable_key(info.identity, step="wav16k")
    wav = cache.reserve("audio", key, ".wav")

    # A truncated download may have settled in the cache on an earlier run, so the
    # finished WAV is checked too — otherwise a clipped lecture stays there forever.
    if not wav.exists() or media.truncation_warning(info.duration, media.probe(wav).duration):
        _download_audio(url, wav, info.duration, cache, key, report, cookies)

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
    report: Report,
    cookies: youtube.Cookies | None,
) -> None:
    """Downloads and converts, trying again when the audio arrived incomplete."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        staging = cache.reserve("downloads", key, "")
        # Leftovers of an interrupted download would confuse the choice of file —
        # start from a clean place and clear up afterwards either way.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            raw = youtube.download_audio(url, staging, report, cookies=cookies)
            with building(wav) as partial:
                media.extract_audio(raw, partial, report)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        warning = media.truncation_warning(declared, media.probe(wav).duration)
        if warning is None:
            return
        if attempt < DOWNLOAD_ATTEMPTS:
            report.say(
                f"{warning}; downloading again, attempt {attempt + 1} of {DOWNLOAD_ATTEMPTS}"
            )
        else:
            report.say(warning)
