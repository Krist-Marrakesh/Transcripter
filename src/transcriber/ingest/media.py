"""Bringing any input to the pipeline's format through ffmpeg.

The only place that deals with containers, codecs and sample rates. Everything
further down the line receives WAV at 16 kHz, mono, and nothing else.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..audio import SAMPLE_RATE
from ..report import Report
from ..subproc import quiet_flags

# How long to wait for the tool before looking at the stop flag again. Short
# enough that a person does not notice the delay, long enough that a transcode
# lasting minutes costs a few hundred idle wake-ups and nothing more.
POLL = 0.2

# Extensions it is not worth even trying beyond.
MEDIA_SUFFIXES = frozenset(
    {
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
        ".aiff",
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".flv",
        ".wmv",
        ".ts",
    }
)


class FFmpegError(RuntimeError):
    """ffmpeg or ffprobe exited with an error."""


# A small disagreement between durations is normal: containers round it and VBR
# embellishes. The threshold is both a fraction and a number of seconds — on an
# hour-long recording a percentage is too generous, on a one-minute one too strict.
DURATION_TOLERANCE = 0.02
DURATION_SLACK = 5.0


def truncation_warning(declared: float, actual: float) -> str | None:
    """Whether all the promised audio arrived. A message, or None if it did.

    A truncated download passes for a sound file: the header promises the full
    duration while there is less inside. ffmpeg decodes what it can and exits with
    a zero code, and so does yt-dlp, so the shortfall is ours to catch. A silent
    loss here is the most expensive kind — what comes out is a coherent transcript,
    just not of the whole recording.
    """
    if declared <= 0 or actual <= 0:
        return None
    if declared - actual <= max(declared * DURATION_TOLERANCE, DURATION_SLACK):
        return None
    return (
        f"warning: {actual / 60:.1f} min of audio instead of the declared {declared / 60:.1f} — "
        "the source is truncated, only this part will be transcribed"
    )


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool


def _require(tool: str) -> str:
    """The path to ffmpeg or ffprobe.

    A system one comes first: it is nearly always newer than ours and already
    suited to the machine. The fallback arrives as a package from PyPI — static
    builds with nothing to link against. Without it the application would demand
    `brew install ffmpeg` before the first recording, and since a window started
    from Finder does not inherit the shell PATH, someone who had installed ffmpeg
    years ago would see that very message.
    """
    if path := shutil.which(tool):
        return path
    if bundled := _bundled(tool):
        return bundled
    raise FFmpegError(f"{tool} was not found: neither in PATH nor in the ffmpeg-binaries package")


def ffmpeg_folder() -> Path | None:
    """The folder holding our ffmpeg, for anything that looks for it itself.

    yt-dlp keeps no fallback and looks only in PATH: on a machine without a system
    ffmpeg it would quietly refuse to merge formats.
    """
    if shutil.which("ffmpeg"):
        # It will find a system one by itself; ours has no business on top.
        return None
    bundled = _bundled("ffmpeg")
    return Path(bundled).parent if bundled else None


def _bundled(tool: str) -> str | None:
    """The binary from `ffmpeg-binaries`, if it is installed and unpacked."""
    try:
        import ffmpeg
    except ImportError:
        return None
    path = {"ffmpeg": ffmpeg.FFMPEG_PATH, "ffprobe": ffmpeg.FFPROBE_PATH}.get(tool)
    return path if path and Path(path).exists() else None


def _run(cmd: list[str], report: Report | None = None) -> str:
    """Runs the tool and hands back its stdout, killing it if a stop is asked for.

    Waiting in slices rather than in one call is what makes the button honest:
    transcoding a four-hour video is minutes of work, and a stop that only takes
    effect once it is over is not a stop. Without a report the loop simply waits.
    """
    told = report or Report()
    # ffmpeg and ffprobe are called for every file, and the application window has
    # no console: without the flag a black rectangle would blink on every call.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **quiet_flags(),
    )
    while True:
        try:
            # Repeated with a timeout rather than polled by hand: `communicate`
            # keeps draining both pipes, and a tool that fills one would otherwise
            # block forever waiting for a reader that never comes.
            out, err = process.communicate(timeout=POLL)
            break
        except subprocess.TimeoutExpired:
            if told.stopping:
                process.kill()

    told.stop_if_asked()
    if process.returncode != 0:
        tail = err.strip().splitlines()[-5:]
        raise FFmpegError("\n".join([f"{cmd[0]} exited with code {process.returncode}", *tail]))
    return out


def probe(path: Path) -> MediaInfo:
    """Reads the duration and the stream layout without decoding the file."""
    raw = _run(
        [
            _require("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(raw)
    codec_types = {stream.get("codec_type") for stream in data.get("streams", [])}
    duration = data.get("format", {}).get("duration")
    return MediaInfo(
        duration=float(duration) if duration else 0.0,
        has_audio="audio" in codec_types,
        has_video="video" in codec_types,
    )


def extract_audio(source: Path, target: Path, report: Report | None = None) -> Path:
    """Takes the audio track out and brings it to WAV, 16 kHz, mono PCM.

    The video stream is dropped (`-vn`) before decoding, so a large mkv is handled
    almost as quickly as an mp3.
    """
    info = probe(source)
    if not info.has_audio:
        raise FFmpegError(f"{source}: the file has no audio track")

    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _require("ffmpeg"),
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map",
            "a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        report,
    )
    # A zero exit code is not proof that anything was written: asked for an output
    # whose format it cannot guess, ffmpeg reports the trouble on stderr and still
    # exits with zero. Without this the failure surfaces much later, as a missing
    # file in a step that has nothing to do with decoding.
    if not target.exists():
        raise FFmpegError(f"ffmpeg produced no output for {source}")
    return target


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_SUFFIXES
