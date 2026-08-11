"""ASR на faster-whisper (CTranslate2).

Основной бэкенд везде, кроме Apple Silicon: на NVIDIA считает через CUDA, на
остальных машинах — на процессоре. На macOS остаётся запасным, потому что Metal
в CTranslate2 не поддерживается и выигрыш там даёт только mlx.
"""

from __future__ import annotations

import numpy as np

from ..device import Device, detect
from ..models import Segment, Word
from .base import ASRResult, Task


def normalize_model_name(name: str) -> str:
    """Приводит имя модели к тому, что понимает faster-whisper.

    mlx-репозитории ему не подходят, но пользователю удобно указывать одно и то
    же имя для обоих бэкендов — снимаем mlx-обёртку с известных названий.
    """
    stripped = name.removeprefix("mlx-community/whisper-").removesuffix("-mlx")
    return stripped if "/" not in stripped else name


def select_device(device: Device | None = None, compute_type: str | None = None) -> tuple[str, str]:
    """Подбирает устройство и точность для CTranslate2.

    Metal он не умеет, поэтому на Apple honest-ответ — процессор. На CUDA
    точность по умолчанию `float16`: она вдвое быстрее `int8` на тензорных ядрах,
    а на процессоре наоборот выигрывает `int8`.
    """
    resolved = device or detect()
    if resolved == "mps":
        resolved = "cpu"
    if compute_type is None:
        compute_type = "float16" if resolved == "cuda" else "int8"
    return resolved, compute_type


class FasterWhisperBackend:
    """Обёртка над `faster_whisper.WhisperModel`."""

    def __init__(
        self, repo: str, *, device: Device | None = None, compute_type: str | None = None
    ) -> None:
        self.repo = normalize_model_name(repo)
        self.target, self.compute_type = select_device(device, compute_type)
        self.device = f"{self.target} · {self.compute_type}"
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.repo, device=self.target, compute_type=self.compute_type
            )
        return self._model

    def release(self) -> None:
        """Отпускает модель. CTranslate2 держит её здесь, и больше нигде."""
        self._model = None

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
        segments, info = self._ensure_model().transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            # Выключен по той же причине, что и у mlx, и там же лежит замер:
            # перенос контекста заводит зацикливание. Ради ровной пунктуации его
            # включать не стоит — она от этого почти не меняется.
            condition_on_previous_text=False,
            # Свой VAD выключен: тишину уже вырезал общий шаг пайплайна.
            vad_filter=False,
        )

        # segments — ленивый генератор, расчёт идёт по мере обхода.
        return ASRResult(
            segments=[_build_segment(item) for item in segments],
            language=info.language or language or "unknown",
        )


def _build_segment(raw) -> Segment:
    return Segment(
        start=float(raw.start),
        end=float(raw.end),
        text=raw.text.strip(),
        words=[
            Word(
                start=float(word.start),
                end=float(word.end),
                text=word.word.strip(),
                probability=getattr(word, "probability", None),
            )
            for word in raw.words or ()
        ],
    )
