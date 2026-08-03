"""Общий интерфейс ASR-бэкендов.

Бэкенды взаимозаменяемы. По умолчанию работает mlx-whisper: он считает на
Metal и на Apple Silicon заметно быстрее. faster-whisper (CTranslate2) на
macOS умеет только CPU и оставлен как запасной вариант.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..models import Segment

Task = Literal["transcribe", "translate"]
"""`translate` у Whisper означает только X→английский — ограничение обучающих
данных, а не архитектуры. Для EN→RU используется LLM из `nlp`."""


@dataclass(frozen=True)
class ASRResult:
    segments: list[Segment]
    language: str
    """Определённый моделью язык — при `language=None` она распознаёт его сама."""


class ASRBackend(Protocol):
    """Контракт бэкенда: массив сэмплов на входе, сегменты с таймкодами на выходе."""

    repo: str

    device: str
    """Где идёт счёт, в человекочитаемом виде.

    Часть контракта, а не украшение: молчаливый откат на процессор — самый
    дорогой отказ в проекте. Он ничего не ломает, только делает распознавание
    в разы медленнее, и обязан быть виден до начала работы, а не после.
    """

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        task: Task = "transcribe",
        beam_size: int = 5,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> ASRResult: ...
