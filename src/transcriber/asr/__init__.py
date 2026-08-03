"""Выбор ASR-бэкенда."""

from __future__ import annotations

from typing import Literal

from .base import ASRBackend, ASRResult, Task

__all__ = ["ASRBackend", "ASRResult", "Task", "create_backend"]


def create_backend(kind: Literal["mlx", "faster"], repo: str) -> ASRBackend:
    """Создаёт бэкенд по имени. Импорты внутри — чтобы не тянуть лишний стек."""
    match kind:
        case "mlx":
            from .mlx_backend import MLXWhisperBackend

            return MLXWhisperBackend(repo)
        case "faster":
            try:
                from .faster_backend import FasterWhisperBackend
            except ImportError as exc:
                raise ImportError(
                    'бэкенд faster-whisper не установлен: uv pip install -e ".[faster]"'
                ) from exc

            return FasterWhisperBackend(repo)
        case _:
            raise ValueError(f"неизвестный ASR-бэкенд: {kind}")
