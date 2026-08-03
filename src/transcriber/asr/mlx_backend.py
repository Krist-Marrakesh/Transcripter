"""ASR на mlx-whisper — Whisper, считающий на Metal.

Важное ограничение бэкенда: в mlx-whisper реализован только жадный декодер,
beam search выбрасывает NotImplementedError. Поэтому `beam_size` здесь
транслируется в `best_of` — mlx сэмплирует несколько траекторий и ранжирует их
по правдоподобию. Работает это только на температурном фолбэке: при t=0 mlx сам
снимает параметр и декодирует жадно. Если нужен настоящий beam search — это
бэкенд faster-whisper, но он на macOS считает на CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Segment, Word
from .base import ASRResult, Task


class MLXWhisperBackend:
    """Обёртка над `mlx_whisper` с приведением вывода к моделям пайплайна."""

    # mlx собирается только под Apple Silicon, поэтому альтернатив Metal здесь нет.
    device = "metal"

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        task: Task = "transcribe",
        beam_size: int = 5,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        # Импорт ленивый: mlx_whisper тянет mlx и при первом вызове веса модели.
        import mlx_whisper

        raw = mlx_whisper.transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            path_or_hf_repo=self.repo,
            language=language,
            task=task,
            # Не beam_size: beam-декодера в mlx нет. См. модульный docstring.
            **({"best_of": beam_size} if beam_size > 1 else {}),
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            # Контекст предыдущего окна улучшает связность, но одна ошибка
            # попадает в него и дальше сама себя поддерживает — то самое
            # зацикливание. На реальных записях отключение надёжнее.
            condition_on_previous_text=False,
            verbose=None,
        )

        return ASRResult(
            segments=[_build_segment(item) for item in raw.get("segments", ())],
            language=raw.get("language") or language or "unknown",
        )


def _build_segment(raw: dict[str, Any]) -> Segment:
    return Segment(
        start=float(raw["start"]),
        end=float(raw["end"]),
        text=raw["text"].strip(),
        words=[
            Word(
                start=float(word["start"]),
                end=float(word["end"]),
                text=word["word"].strip(),
                probability=word.get("probability"),
            )
            for word in raw.get("words") or ()
        ],
    )
