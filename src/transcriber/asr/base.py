"""The common interface of the ASR backends.

The backends are interchangeable. mlx-whisper is the default: it computes on
Metal and is noticeably faster on Apple Silicon. faster-whisper (CTranslate2) can
only use the CPU on macOS and is kept as the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..models import Segment
from ..report import Advance

Task = Literal["transcribe", "translate"]
"""`translate` in Whisper means only X→English — a limit of the training data
rather than of the architecture. EN→RU goes through the LLM in `nlp`."""


@dataclass(frozen=True)
class ASRResult:
    segments: list[Segment]
    language: str
    """The language the model settled on — with `language=None` it decides."""


class ASRBackend(Protocol):
    """The contract: samples in, segments with timestamps out."""

    repo: str

    device: str
    """Where the computing happens, in a form a person can read.

    Part of the contract, not decoration: a silent fall back to the processor is
    the most expensive failure in the project. It breaks nothing, it only makes
    recognition several times slower, and it has to be visible before the work
    starts rather than after.
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
        progress: Advance | None = None,
    ) -> ASRResult:
        """Recognises the samples handed over.

        `progress` is measured against those samples, not the original recording:
        the silence is already cut out by then, and a bar counting the original
        would stop at a quarter on a lecture that is three quarters pauses.
        """
        ...
