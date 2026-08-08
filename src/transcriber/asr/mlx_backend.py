"""ASR on mlx-whisper — Whisper computing on Metal.

An important limitation of the backend: only a greedy decoder is implemented in
mlx-whisper, beam search raises NotImplementedError. So `beam_size` is translated
into `best_of` here — mlx samples several trajectories and ranks them by
likelihood. That works only on the temperature fallback: at t=0 mlx drops the
parameter itself and decodes greedily. If real beam search is needed, that is the
faster-whisper backend, but on macOS it computes on the processor.

Progress comes through the only seam the library leaves. `transcribe` walks the
recording in thirty-second windows and moves a tqdm bar after each one, but takes
no callback of its own. So for the length of the call the tqdm inside its
namespace is replaced by a counter of ours: what it reports is frames actually
consumed, not an estimate from a stopwatch.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..models import Segment, Word
from ..report import Advance
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
        progress: Advance | None = None,
    ) -> ASRResult:
        # A lazy import: mlx_whisper pulls in mlx, and the model weights on the
        # first call.
        import mlx_whisper

        with _reporting(progress):
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


class _Counter:
    """A stand-in for a tqdm bar: it takes the same calls and reports a fraction.

    The signature is deliberately open. mlx-whisper may add `desc` or `leave`
    tomorrow, and a narrow one would turn into a TypeError in the middle of an
    hour-long recording.

    `disable` is ignored knowingly: with `verbose=None` it arrives as True, and
    obeying it would silence the very reporting this class exists for.
    """

    report: Advance

    def __init__(self, *args: Any, total: int = 0, **kwargs: Any) -> None:
        self._total = total
        self._done = 0

    def __enter__(self) -> _Counter:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def update(self, step: int = 1) -> None:
        self._done += step
        if self._total > 0:
            self.report(min(1.0, self._done / self._total))


@contextmanager
def _reporting(progress: Advance | None) -> Iterator[None]:
    """Puts our counter in place of tqdm for the length of one call.

    Restored on the way out in any case. The substitution is visible to the whole
    process while it stands, and the weight downloader runs in a thread of its
    own — a patch left behind after an exception would follow it into the next
    recording.
    """
    # By name through importlib, not `import mlx_whisper.transcribe as engine`.
    # The package rebinds that name to the function of the same name, so the
    # import statement hands back the function rather than the module the bar
    # lives in — the patch then lands nowhere and reports nothing.
    engine = importlib.import_module("mlx_whisper.transcribe")

    original = getattr(engine, "tqdm", None)
    if progress is None or original is None:
        # No one to report to, or the library has moved its bar elsewhere. Either
        # way, a progress bar must never be the reason a transcription fails.
        yield
        return

    bound = type("BoundCounter", (_Counter,), {"report": staticmethod(progress)})
    # A stand-in for the module: the library calls `tqdm.tqdm(...)`, so what is
    # replaced has to answer to that name too.
    engine.tqdm = type("tqdm", (), {"tqdm": bound})
    try:
        yield
    finally:
        engine.tqdm = original


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
