"""Загрузка с YouTube и прочих площадок, поддерживаемых yt-dlp.

Отдельный быстрый путь: если у видео уже есть субтитры, выложенные автором,
их можно взять готовыми и не запускать ASR вообще.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


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

    return YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True, **options})


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


def download_audio(url: str, target_dir: Path) -> Path:
    """Качает лучшую доступную аудиодорожку без перекодирования.

    Перекодировать здесь незачем: следующим шагом ffmpeg всё равно приводит
    файл к формату пайплайна.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    template = str(target_dir / "%(id)s.%(ext)s")

    try:
        with _ydl(format="bestaudio/best", outtmpl=template) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise DownloadError(f"не удалось скачать {url}: {exc}") from exc
