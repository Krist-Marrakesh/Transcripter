"""Загрузка с YouTube и прочих площадок, поддерживаемых yt-dlp.

Отдельный быстрый путь: если у видео уже есть субтитры, выложенные автором,
их можно взять готовыми и не запускать ASR вообще.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

Progress = Callable[[float], None]
"""Доля скачанного, 0..1. Та же мера, что у распознавания, — полоса одна."""


class DownloadError(RuntimeError):
    """yt-dlp не смог получить медиа."""


@dataclass(frozen=True)
class RemoteInfo:
    url: str
    title: str
    duration: float
    # Языки субтитров, выложенных автором. Автогенерённые сюда не попадают:
    # они сделаны тем же Whisper-подобным ASR и обычно хуже нашего локального.
    subtitle_languages: tuple[str, ...] = field(default=())


def is_url(value: str) -> bool:
    return bool(URL_PATTERN.match(value))


def _ydl(**options: object):
    """Создаёт YoutubeDL с общими настройками. Импорт ленивый — yt-dlp тяжёлый."""
    from yt_dlp import YoutubeDL

    from .media import ffmpeg_folder

    # yt-dlp ищет ffmpeg в PATH и своего запасного не имеет. Каталог указываем
    # явно: без него на машине без системного ffmpeg он молча отказался бы
    # склеивать форматы — при том что у нас ffmpeg есть, просто не в PATH.
    location = {"ffmpeg_location": str(folder)} if (folder := ffmpeg_folder()) else {}
    return YoutubeDL(
        {"quiet": True, "no_warnings": True, "noprogress": True, **location, **options}
    )


def probe(url: str) -> RemoteInfo:
    """Достаёт метаданные без скачивания."""
    try:
        with _ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp кидает собственную иерархию исключений
        raise DownloadError(f"не удалось получить сведения о {url}: {exc}") from exc

    return RemoteInfo(
        url=url,
        title=info.get("title") or "untitled",
        duration=float(info.get("duration") or 0.0),
        subtitle_languages=tuple(info.get("subtitles") or ()),
    )


def download_audio(url: str, target_dir: Path, progress: Progress | None = None) -> Path:
    """Качает лучшую доступную аудиодорожку без перекодирования.

    Перекодировать здесь незачем: следующим шагом ffmpeg всё равно приводит
    файл к формату пайплайна.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    template = str(target_dir / "%(id)s.%(ext)s")
    hooks = [_reporter(progress)] if progress is not None else []

    try:
        with _ydl(format="bestaudio/best", outtmpl=template, progress_hooks=hooks) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise DownloadError(f"не удалось скачать {url}: {exc}") from exc


def _reporter(progress: Progress) -> Callable[[dict], None]:
    """Переводит отчёт yt-dlp в долю выполненного.

    Точного размера может не быть вовсе: сервер отдаёт его не всегда, и остаётся
    оценка. Пока нет ни того, ни другого, доля не считается — шкала, ползущая по
    догадке, хуже отсутствующей.
    """

    def hook(event: dict) -> None:
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
