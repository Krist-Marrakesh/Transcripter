"""Локальный транскрибатор аудио, видео и ссылок с YouTube.

Пайплайн: ingest (ffmpeg / yt-dlp) → VAD → ASR (Whisper) → диаризация →
перевод и саммари локальной LLM → экспорт.

Всё считается на своём железе; сеть нужна один раз, чтобы скачать веса моделей.
"""

from __future__ import annotations

from .config import Settings, load_settings
from .models import Diarization, Segment, SpeakerTurn, Summary, Transcript, Translation, Word
from .pipeline import Pipeline

__version__ = "0.1.0"

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
