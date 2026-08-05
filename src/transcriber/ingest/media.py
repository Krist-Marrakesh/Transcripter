"""Приведение любого входа к формату пайплайна через ffmpeg.

Единственное место, где мы имеем дело с контейнерами, кодеками и частотами
дискретизации. Всё, что ниже по конвейеру, получает уже WAV 16 кГц моно.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..audio import SAMPLE_RATE
from ..subproc import quiet_flags

# Расширения, по которым не имеет смысла даже пытаться.
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
    """ffmpeg или ffprobe завершились с ошибкой."""


# Небольшое расхождение длительностей — норма: контейнеры её округляют, а VBR
# привирает. Порог берём и в долях, и в секундах: на часовой записи процент
# великоват, на минутной — наоборот, слишком чувствителен.
DURATION_TOLERANCE = 0.02
DURATION_SLACK = 5.0


def truncation_warning(declared: float, actual: float) -> str | None:
    """Проверяет, весь ли обещанный звук доехал. Сообщение или None, если всё на месте.

    Оборванная загрузка притворяется исправным файлом: заголовок обещает полную
    длительность, а данных внутри меньше. ffmpeg декодирует сколько может и
    выходит с нулевым кодом, yt-dlp — тоже, поэтому недостачу мы ловим сами.
    Молчаливая потеря здесь дороже всего: на выходе связный транскрипт, просто
    не всей записи.
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
    """Путь к ffmpeg или ffprobe.

    Системный идёт первым: он почти всегда свежее нашего и уже настроен под
    железо машины. Запасной приезжает пакетом с PyPI — статические сборки без
    внешних зависимостей. Без него приложение требовало бы `brew install ffmpeg`
    ещё до первого распознавания, а окно из Finder не наследует PATH оболочки,
    и человек с установленным ffmpeg видел бы ровно то же сообщение.
    """
    if path := shutil.which(tool):
        return path
    if bundled := _bundled(tool):
        return bundled
    raise FFmpegError(f"{tool} не найден: ни в PATH, ни в пакете ffmpeg-binaries")


def ffmpeg_folder() -> Path | None:
    """Каталог с нашим ffmpeg — для тех, кто ищет его сам.

    yt-dlp своего запасного не держит и смотрит только в PATH: на машине без
    системного ffmpeg он молча отказался бы склеивать форматы.
    """
    if shutil.which("ffmpeg"):
        # Системный он найдёт и сам, а подсовывать наш поверх незачем.
        return None
    bundled = _bundled("ffmpeg")
    return Path(bundled).parent if bundled else None


def _bundled(tool: str) -> str | None:
    """Бинарь из пакета `ffmpeg-binaries`, если тот стоит и распакован."""
    try:
        import ffmpeg
    except ImportError:
        return None
    path = {"ffmpeg": ffmpeg.FFMPEG_PATH, "ffprobe": ffmpeg.FFPROBE_PATH}.get(tool)
    return path if path and Path(path).exists() else None


def _run(cmd: list[str]) -> str:
    # ffmpeg и ffprobe зовутся на каждый файл, а окно приложения консоли не
    # имеет: без флага под Windows на каждый вызов мигал бы чёрный прямоугольник.
    result = subprocess.run(cmd, capture_output=True, text=True, **quiet_flags())
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-5:]
        raise FFmpegError("\n".join([f"{cmd[0]} завершился с кодом {result.returncode}", *tail]))
    return result.stdout


def probe(path: Path) -> MediaInfo:
    """Читает длительность и состав потоков без декодирования файла."""
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


def extract_audio(source: Path, target: Path) -> Path:
    """Достаёт аудиодорожку и приводит её к WAV 16 кГц моно PCM.

    Видеопоток отбрасывается (`-vn`) до декодирования, поэтому большой mkv
    обрабатывается почти так же быстро, как mp3.
    """
    info = probe(source)
    if not info.has_audio:
        raise FFmpegError(f"{source}: в файле нет аудиодорожки")

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
        ]
    )
    return target


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_SUFFIXES
