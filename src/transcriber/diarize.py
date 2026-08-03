"""Diarization: who speaks when.

It has nothing to do with recognising text — this is a separate family of models:
speech segmentation → voice embeddings (which encode timbre, not content) →
clustering. Words and speakers are computed independently and stitched together
by timestamp overlap; this is the step where interruptions produce errors.

pyannote models are gated: a HuggingFace token and an accepted agreement on the
model page are required.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .models import Segment, SpeakerTurn


class DiarizationError(RuntimeError):
    """The diarization pipeline is unavailable or failed to run."""


def diarize(
    audio: Path,
    *,
    model: str,
    token: str | None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerTurn]:
    """Labels a recording by speaker.

    When the speaker count is known in advance, `num_speakers` improves quality
    noticeably: clustering no longer has to guess it from a distance threshold.
    """
    # A lazy import: pyannote pulls in torch and lightning, seconds at startup.
    from pyannote.audio import Pipeline

    if not token:
        raise DiarizationError(
            "a HuggingFace token is required: set TRANSCRIPT_HF_TOKEN in .env.\n"
            f"Token: https://huggingface.co/settings/tokens\n"
            f"The model terms must be accepted by hand: https://huggingface.co/{model}"
        )

    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as exc:
        raise DiarizationError(f"could not load {model}: {exc}") from exc

    if pipeline is None:
        raise DiarizationError(
            f"pyannote returned an empty pipeline for {model} — "
            f"the terms at https://huggingface.co/{model} were most likely not accepted"
        )

    _move_to_device(pipeline)

    constraints = {
        key: value
        for key, value in (
            ("num_speakers", num_speakers),
            ("min_speakers", min_speakers),
            ("max_speakers", max_speakers),
        )
        if value is not None
    }

    # An invariant: what arrives here is the path to the full recording, not the
    # VAD-compressed array. pyannote timings must sit on the original axis — ASR
    # segments are already mapped back by now, and stitching happens in one system.
    try:
        result = pipeline(str(audio), **constraints)
    except Exception as exc:
        # Inference fails in many ways (out of memory, broken audio, MPS gaps), but
        # for the caller it is one case: no speakers. A single error type lets the
        # pipeline tell a failed enrichment from a failed ASR run.
        raise DiarizationError(f"could not label speakers: {exc}") from exc

    turns = [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=speaker)
        for segment, _, speaker in _annotation(result).itertracks(yield_label=True)
    ]
    return _relabel(turns)


def _annotation(result):
    """Extracts the annotation from whatever the pipeline returned.

    In pyannote 4 this is no longer an `Annotation` but a composite object: the
    annotation, its overlap-free variant and speaker embeddings. We take the
    overlap-free one — it is meant for stitching onto a transcript: a line spoken
    over someone else is still one line in the text, and two speakers claiming a
    single ASR segment would only argue with each other.
    """
    for field in ("exclusive_speaker_diarization", "speaker_diarization"):
        if (annotation := getattr(result, field, None)) is not None:
            return annotation
    # pyannote 3 returned the annotation directly.
    return result


def _move_to_device(pipeline) -> None:
    """Moves models onto the GPU when there is one. Failure is fine — CPU then."""
    import torch

    from .device import detect

    device = detect()
    if device == "cpu":
        return
    try:
        pipeline.to(torch.device(device))
    except (RuntimeError, NotImplementedError, AssertionError):
        # Some pyannote operations are not implemented on MPS, and CUDA may simply
        # run out of memory. Either is a reason to stay on the CPU, not to crash.
        pass


def _relabel(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Renames SPEAKER_00 into "Спикер 1" in order of first appearance."""
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
