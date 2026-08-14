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

    device_warning = ""
    """Always empty: with one device there is nothing to have landed instead of."""

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def load(self) -> None:
        """Nothing to bring up early, and that is the answer rather than an omission.

        `load` exists so that `device` is settled before the pipeline announces
        it. Here it is settled at construction and cannot move: mlx builds only
        for Apple Silicon, and Metal has nothing to fall back to. Reading the
        weights now would shift the same seconds earlier and buy nothing — the
        library reads them on the first call and keeps them in a class attribute
        of its own, which is what `release` empties.
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
            # The context of the previous window improves coherence, but a single
            # mistake lands in it and then keeps itself alive — the looping. Off
            # on real recordings, and now measured rather than assumed, because
            # punctuation is the thing people ask this to fix.
            #
            # A 42-minute lecture through this pipeline, with it on: sentences
            # get shorter — 22.9 words against 28.9, commas 7.8 per hundred words
            # against 6.5 — and recognition takes 4.9 minutes instead of 3.5. And
            # the word "generally" repeats forty-two times in a row, in a Russian
            # lecture. Fourteen more places repeat a line at least twice; with it
            # off there is not one.
            #
            # Whoever comes back to this: measure through the pipeline, not with
            # one call to the library. Recognition goes in portions of two
            # minutes, so the context resets at every boundary anyway. Measured
            # in a single ten-minute call the same flag looks like it closes 93%
            # of segments with a full stop against 30% — a number this
            # application can never reach.
            condition_on_previous_text=False,
            verbose=None,
        )

        return ASRResult(
            segments=[_build_segment(item) for item in raw.get("segments", ())],
            language=raw.get("language") or language or "unknown",
        )

    def release(self) -> None:
        """Frees the weights mlx-whisper keeps to itself between calls.

        They live in a class attribute of the library and outlive this object
        entirely, so dropping the backend frees nothing at all. Emptying both
        fields is the class's own contract rather than a patch of it: its
        `get_model` loads again whenever the model is missing or the path has
        changed.
        """
        import gc
        import sys

        # Never brought up, nothing held: a recording with no speech in it at all
        # gives no portions, and then the weights were never read. Asking is a
        # dictionary lookup against the second of import that answering it costs.
        if "mlx_whisper.transcribe" not in sys.modules:
            return

        import mlx.core as mx

        # `from mlx_whisper.transcribe import ...` and not an attribute of the
        # package: `mlx_whisper.transcribe` there is the function of that name,
        # not the module, and reaching through it raises AttributeError. This is
        # the same footgun as the tqdm patch that quietly missed its module —
        # and a release that misses looks exactly like one that worked.
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
        # The weights are unreachable now but not yet returned: Python has to
        # collect the object, and mlx keeps freed buffers in a pool of its own.
        gc.collect()
        mx.clear_cache()


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
