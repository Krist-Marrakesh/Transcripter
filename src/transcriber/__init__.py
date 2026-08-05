"""Локальный транскрибатор аудио, видео и ссылок с YouTube.

Пайплайн: ingest (ffmpeg / yt-dlp) → VAD → ASR (Whisper) → диаризация →
перевод и саммари локальной LLM → экспорт.

Всё считается на своём железе; сеть нужна один раз, чтобы скачать веса моделей.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import Settings, load_settings
from .models import Diarization, Segment, SpeakerTurn, Summary, Transcript, Translation, Word
from .pipeline import Pipeline

# Asked of the installed distribution rather than written down here. The updater
# compares what is running against what is published, and a literal in the source
# answers a different question than the one being asked: after an update installs
# a new wheel into the environment, the literal would still name the old version
# and the same update would be offered forever.
try:
    __version__ = version("transcript")
except PackageNotFoundError:
    # Running straight from a source tree that was never installed. Nothing to
    # compare against, and saying so beats inventing a number.
    __version__ = "0"

__all__ = [
    "Diarization",
    "Pipeline",
    "Segment",
    "Settings",
    "SpeakerTurn",
    "Summary",
    "Transcript",
    "Translation",
    "Word",
    "load_settings",
]
