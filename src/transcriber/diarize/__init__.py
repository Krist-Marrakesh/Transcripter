"""Diarization: who speaks when.

It has nothing to do with recognising text — this is a separate family of models:
speech segmentation → voice embeddings (which encode timbre, not content) →
clustering. Words and speakers are computed independently and stitched together
by timestamp overlap; this is the step where interruptions produce errors.

Two backends, and the choice between them is not about quality alone. pyannote's
models are gated: the terms have to be accepted by hand on the model page and a
token obtained, which no installer can do on someone's behalf. sherpa-onnx runs
on openly published models and therefore works on a machine where nothing has
been arranged — which is why it is the default.

What that costs was measured rather than guessed, on five recordings and four
and a half hours: see `sherpa_backend`. In short, sherpa invents no speakers who
are not there, and misses rare brief ones who are.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from ..models import Segment, SpeakerTurn

Notify = Callable[[str], None]

Backend = Literal["sherpa", "pyannote"]


class DiarizationError(RuntimeError):
    """The diarization pipeline is unavailable or failed to run."""


def diarize(
    audio: Path,
    *,
    backend: Backend = "sherpa",
    model: str = "",
    token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    notify: Notify | None = None,
) -> list[SpeakerTurn]:
    """Labels a recording by speaker.

    When the speaker count is known in advance, `num_speakers` improves quality
    noticeably on either backend: clustering no longer has to guess it from a
    distance threshold, and guessing it is the harder half of the job.
    """
    say = notify or (lambda _: None)

    if backend == "pyannote":
        from .pyannote_backend import run

        turns = run(
            audio,
            model=model,
            token=token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            notify=say,
        )
    else:
        from .sherpa_backend import run

        turns = run(audio, num_speakers=num_speakers, notify=say)

    return _relabel(turns)


def _relabel(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Renames the backend's own labels into "Спикер 1" by order of appearance.

    Here rather than in the backends: the numbering a person reads should not
    depend on which of the two produced it.
    """
    mapping: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"Спикер {len(mapping) + 1}"
    return [t.model_copy(update={"speaker": mapping[t.speaker]}) for t in turns]


def assign_speakers(segments: Sequence[Segment], turns: Sequence[SpeakerTurn]) -> list[Segment]:
    """Assigns each segment the speaker with the largest overlap in time.

    Both sequences are sorted by start, so one pass with a sliding window is
    enough: the `lo` pointer only ever moves forward.
    """
    result = list(segments)
    if not turns:
        return result

    ordered = sorted(turns, key=lambda t: t.start)
    lo = 0

    for position, segment in enumerate(result):
        # Turns that ended before this segment began are of no use later either:
        # segments come in increasing order of start.
        while lo < len(ordered) and ordered[lo].end <= segment.start:
            lo += 1

        overlaps: defaultdict[str, float] = defaultdict(float)
        for index in range(lo, len(ordered)):
            turn = ordered[index]
            if turn.start >= segment.end:
                break
            span = min(turn.end, segment.end) - max(turn.start, segment.start)
            if span > 0:
                overlaps[turn.speaker] += span

        if overlaps:
            best = max(overlaps, key=overlaps.__getitem__)
            result[position] = segment.model_copy(update={"speaker": best})

    return result
