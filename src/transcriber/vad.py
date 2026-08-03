"""VAD — speech detection and silence removal.

Why it is needed: Whisper is autoregressive. On silence and music it has nothing
to lean on, yet it must still emit the next token — hence phantom sign-offs and
loops. Not feeding it such audio at all is cheaper.

A side effect is speed. The decoder runs once per token, so the pauses thrown
away save more time than they themselves last.

How it works: find speech intervals → glue them into one continuous stream →
recognise that → map timestamps back onto the original axis via `Timeline`.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .audio import SAMPLE_RATE, to_samples
from .models import Segment


@dataclass(frozen=True)
class SpeechRegion:
    """A speech interval on the original time axis."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class Timeline:
    """A mapping from the compressed time axis back onto the original one.

    After gluing, ASR sees a recording without pauses and reports timestamps in
    that compressed coordinate system. `to_original` converts them back.
    """

    def __init__(self, regions: Sequence[SpeechRegion] = ()) -> None:
        self.regions = tuple(regions)

        # Where each region starts on the compressed axis — prefix sums of durations.
        starts: list[float] = []
        cursor = 0.0
        for region in self.regions:
            starts.append(cursor)
            cursor += region.duration

        self._starts = starts
        self.compressed_duration = cursor

    @property
    def is_identity(self) -> bool:
        """True when nothing was compressed and no conversion is needed."""
        return not self.regions

    def to_original(self, timestamp: float) -> float:
        """Converts a moment on the compressed axis to the original one. O(log n)."""
        if not self.regions:
            return timestamp

        index = max(0, bisect_right(self._starts, timestamp) - 1)
        region = self.regions[index]
        # Clamped by the right edge: ASR sometimes reports a timestamp past the chunk.
        return min(region.start + (timestamp - self._starts[index]), region.end)

    def remap(self, segment: Segment) -> Segment:
        """Moves a segment onto the original axis together with word timestamps."""
        if self.is_identity:
            return segment
        return segment.model_copy(
            update={
                "start": self.to_original(segment.start),
                "end": self.to_original(segment.end),
                "words": [
                    word.model_copy(
                        update={
                            "start": self.to_original(word.start),
                            "end": self.to_original(word.end),
                        }
                    )
                    for word in segment.words
                ],
            }
        )


@lru_cache(maxsize=1)
def _load_model():
    """The silero model is a few megabytes but is not instant to load — keep one."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def detect(
    samples: np.ndarray,
    *,
    threshold: float = 0.5,
    min_speech: float = 0.25,
    min_silence: float = 0.5,
    padding: float = 0.15,
) -> list[SpeechRegion]:
    """Finds speech intervals.

    Args:
        threshold: confidence threshold; higher means stricter selection.
        min_speech: fragments shorter than this are not speech (cuts off clicks).
        min_silence: a pause shorter than this does not break a line.
        padding: margin around speech so attack and decay are not clipped.
    """
    import torch
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        torch.from_numpy(samples),
        _load_model(),
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=int(min_speech * 1000),
        min_silence_duration_ms=int(min_silence * 1000),
        speech_pad_ms=int(padding * 1000),
        return_seconds=True,
    )
    return [SpeechRegion(start=s["start"], end=s["end"]) for s in stamps]


def compress(samples: np.ndarray, regions: Sequence[SpeechRegion]) -> np.ndarray:
    """Glues the speech pieces into one continuous stream."""
    if not regions:
        return samples
    return np.concatenate([samples[to_samples(r.start) : to_samples(r.end)] for r in regions])


def speech_ratio(regions: Sequence[SpeechRegion], total_duration: float) -> float:
    """Share of the recording taken by speech. Used only for reporting."""
    if total_duration <= 0:
        return 0.0
    return sum(r.duration for r in regions) / total_duration
