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

    Settled once `load` has run; before that it is an intention.
    """

    device_warning: str
    """Why the work is not on the device that was chosen — empty when it is.

    A field rather than a sentence the backend says for itself. Only the backend
    can know that its model refused to come up on the card; only the caller knows
    where a line for a person goes. Splitting it that way is also what keeps
    `Report` out of here — see the note on `transcribe`.
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
    ) -> ASRResult:
        """Recognises the samples handed over: in, and out.

        No progress channel on purpose. The pipeline hands recognition one portion
        at a time and counts the portions, so it already knows how far along the
        recording is — while a backend knows only about the array in front of it.
        Asking every backend to invent a way of reporting is what made us patch
        tqdm inside somebody else's module.
        """
        ...

    def load(self) -> None:
        """Brings the model up, so that `device` stops being a promise.

        The other half of `release`, and the announcement is what it exists for.
        The device is named before recognition starts, because learning afterwards
        that an hour of lecture went to the processor is learning too late. While
        the model came up inside the first call, that line described an intention:
        a backend that then failed over to the CPU had already been announced as
        running on the card.

        A backend with nowhere to fall back to has nothing to do here. Saying so
        in its own docstring is the answer; leaving the method out is not.
        """
        ...

    def release(self) -> None:
        """Lets go of the model: the memory is wanted by the next step.

        Recognition and the LLM are never needed at the same time, and where a
        backend keeps its weights is its own business — one holds them in this
        object, another in a global of its own that outlives it. Only the
        backend can know which, so only the backend can let go.
        """
        ...
