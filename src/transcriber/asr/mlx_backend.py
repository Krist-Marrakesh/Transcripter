"""ASR on mlx-whisper — Whisper computing on Metal.

An important limitation of the backend: only a greedy decoder is implemented in
mlx-whisper, beam search raises NotImplementedError. So `beam_size` is translated
into `best_of` here — mlx samples several trajectories and ranks them by
likelihood. That works only on the temperature fallback: at t=0 mlx drops the
parameter itself and decodes greedily. If real beam search is needed, that is the
faster-whisper backend, but on macOS it computes on the processor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Segment, Word
from .base import ASRResult, Task


class MLXWhisperBackend:
    """A wrapper over `mlx_whisper` that maps the output onto pipeline models."""

    # mlx builds only for Apple Silicon, so there is no alternative to Metal here.
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
        # A lazy import: mlx_whisper pulls in mlx, and the model weights on the
        # first call.
        import mlx_whisper

        raw = mlx_whisper.transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            path_or_hf_repo=self.repo,
            language=language,
            task=task,
            # Not beam_size: mlx has no beam decoder. See the module docstring.
            **({"best_of": beam_size} if beam_size > 1 else {}),
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            # The context of the previous window improves coherence, but a
            # single mistake lands in it and then keeps itself alive — the
            # looping. On real recordings turning it off is safer.
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
