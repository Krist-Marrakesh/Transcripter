"""Downloading from YouTube and the other sites yt-dlp supports.

There is a fast path of its own: when a video already has subtitles published by
its author, they can be taken as they are and ASR skipped altogether.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

Progress = Callable[[float], None]
"""The share downloaded, 0 to 1. The same measure recognition uses — one bar."""


class DownloadError(RuntimeError):
    """yt-dlp could not get the media."""


@dataclass(frozen=True)
class RemoteInfo:
    url: str
    title: str
    duration: float
    # Subtitle languages published by the author. Auto-generated ones are left
    # out: they come from the same kind of ASR as ours and are usually worse.
    subtitle_languages: tuple[str, ...] = field(default=())


def is_url(value: str) -> bool:
    return bool(URL_PATTERN.match(value))


def _ydl(**options: object):
    """Builds a YoutubeDL with the shared settings. Lazy import: yt-dlp is heavy."""
    from yt_dlp import YoutubeDL

    from .media import ffmpeg_folder

    # yt-dlp looks for ffmpeg in PATH and keeps no fallback. The folder is named
    # explicitly: without it, on a machine with no system ffmpeg, it would quietly
    # refuse to merge formats — while we do have ffmpeg, just not where it looked.
    location = {"ffmpeg_location": str(folder)} if (folder := ffmpeg_folder()) else {}
    return YoutubeDL(
        {"quiet": True, "no_warnings": True, "noprogress": True, **location, **options}
    )


def probe(url: str) -> RemoteInfo:
    """Fetches the metadata without downloading anything."""
    try:
        with _ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises a hierarchy of its own
        raise DownloadError(f"could not get the details of {url}: {exc}") from exc

    return RemoteInfo(
        url=url,
        title=info.get("title") or "untitled",
        duration=float(info.get("duration") or 0.0),
        subtitle_languages=tuple(info.get("subtitles") or ()),
    )


class Stopped(RuntimeError):
    """Загрузку остановили по просьбе человека."""


def download_audio(
    url: str,
    target_dir: Path,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Fetches the best available audio track without transcoding it.

    There is nothing to gain by transcoding here: the next step brings the file to
    the pipeline's format through ffmpeg anyway.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    template = str(target_dir / "%(id)s.%(ext)s")
    hooks = [_reporter(progress, cancel)] if progress is not None or cancel is not None else []

    try:
        with _ydl(format="bestaudio/best", outtmpl=template, progress_hooks=hooks) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    except Stopped:
        # Не отказ, а решение человека — пусть поднимается как есть.
        raise
    except Exception as exc:
        raise DownloadError(f"could not download {url}: {exc}") from exc


def _reporter(
    progress: Progress | None, cancel: threading.Event | None = None
) -> Callable[[dict], None]:
    """Turns a yt-dlp report into a share of the work done.

    An exact size may not exist at all: a server does not always give one, and an
    estimate is what is left. While there is neither, no share is reported — a bar
    crawling along a guess is worse than no bar.
    """

    def hook(event: dict) -> None:
        # Единственное место, где yt-dlp отдаёт нам управление по ходу загрузки.
        # Исключение отсюда — способ её прервать: часовое видео иначе пришлось бы
        # дожидаться до конца.
        if cancel is not None and cancel.is_set():
            raise Stopped("download stopped")
        if progress is None:
            return
        status = event.get("status")
        if status == "finished":
            progress(1.0)
            return
        if status != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate") or 0
        if total > 0:
            progress(min(1.0, (event.get("downloaded_bytes") or 0) / total))

    return hook
