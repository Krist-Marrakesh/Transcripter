"""Downloading from YouTube and the other sites yt-dlp supports.

There is a fast path of its own: when a video already has subtitles published by
its author, they can be taken as they are and ASR skipped altogether.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..report import Report, Stopped

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


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
    # What the site calls this recording, prefixed by the site. The address is
    # not that: `?t=2940s` says where a browser should start playing and changes
    # nothing about the media, yet it made the same battle download a second
    # time under a second name. The extractor is part of it because the numbers
    # two sites hand out have no reason to differ.
    media_id: str = ""

    @property
    def identity(self) -> str:
        """What to key a download by. Falls back to the address if the site gave nothing."""
        return self.media_id or self.url


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
        {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            # Без цвета: yt-dlp красит свои сообщения escape-последовательностями,
            # а они попадают в текст исключения и дальше — в окно, где выглядят
            # как «[0;31mERROR:[0m». Гасить их разбором постфактум было бы
            # лечением следствия: проще не просить красить.
            "color": "no_color",
            # Ссылка на видео из плейлиста несёт и `v=`, и `list=`, а yt-dlp по
            # умолчанию понимает такую как весь список. Проверено на лекции №20
            # из курса: без этого он отдавал `YoutubeTab` с 47 записями, без
            # длительности и с заголовком всего курса, а скачивание кончалось
            # «HTTP Error 403: Forbidden». С флагом — `Youtube:OzIGqaizOAo`,
            # 741 секунда, своё название.
            #
            # Хуже 403 то, что тише: ключ кэша строится по этому же
            # идентификатору, и все сорок семь лекций курса делили бы один ключ
            # — вторая открывалась бы звуком первой.
            "noplaylist": True,
            **location,
            **options,
        }
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
        media_id=_identity(info),
    )


def _identity(info: dict) -> str:
    """`youtube:f_KUzNwlMpo` — the site and its own name for the recording."""
    site = info.get("extractor_key") or info.get("extractor") or ""
    name = info.get("id") or ""
    return f"{site}:{name}".strip(":") if name else ""


def download_audio(url: str, target_dir: Path, report: Report | None = None) -> Path:
    """Fetches the best available audio track without transcoding it.

    There is nothing to gain by transcoding here: the next step brings the file to
    the pipeline's format through ffmpeg anyway.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    template = str(target_dir / "%(id)s.%(ext)s")
    hooks = [_reporter(report)] if report is not None else []

    try:
        with _ydl(format="bestaudio/best", outtmpl=template, progress_hooks=hooks) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    except Stopped:
        # Не отказ, а решение человека — пусть поднимается как есть.
        raise
    except Exception as exc:
        raise DownloadError(f"could not download {url}: {exc}") from exc


def _reporter(report: Report) -> Callable[[dict], None]:
    """Turns a yt-dlp report into a share of the work done.

    An exact size may not exist at all: a server does not always give one, and an
    estimate is what is left. While there is neither, no share is reported — a bar
    crawling along a guess is worse than no bar.
    """

    def hook(event: dict) -> None:
        # Единственное место, где yt-dlp отдаёт нам управление по ходу загрузки.
        # Исключение отсюда — способ её прервать: часовое видео иначе пришлось бы
        # дожидаться до конца.
        report.stop_if_asked()
        status = event.get("status")
        if status == "finished":
            report.at(1.0)
            return
        if status != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate") or 0
        if total > 0:
            report.at(min(1.0, (event.get("downloaded_bytes") or 0) / total))

    return hook
