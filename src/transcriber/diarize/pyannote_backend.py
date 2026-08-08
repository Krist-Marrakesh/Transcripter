"""Diarization through pyannote — the pipeline that needs a token.

The models are gated: the API reports `gated: auto`, so the terms have to be
accepted by hand on the model page and a token obtained. That is the model
owner's licence and nothing an installer can arrange, which is why this backend
is offered rather than assumed.

What it buys is the rarer speaker. Measured against sherpa on the same five
recordings, pyannote found three speakers where sherpa found two, and four where
sherpa found two — the ones it keeps are brief questions from the room.
"""

from __future__ import annotations

from pathlib import Path

from ..models import SpeakerTurn
from ..report import Report
from . import DiarizationError


def run(
    audio: Path,
    *,
    model: str,
    token: str | None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    report: Report,
) -> list[SpeakerTurn]:
    """Labels a recording by speaker."""
    # A lazy import: pyannote pulls in torch and lightning, seconds at startup.
    from pyannote.audio import Pipeline

    if not token:
        raise DiarizationError(
            "a HuggingFace token is required.\n"
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

    _move_to_device(pipeline, report)

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
    return turns


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


def _move_to_device(pipeline, report: Report) -> None:
    """Moves models onto the GPU when there is one. Failure is fine — CPU then.

    Fine, but not silent. On a machine with a card, labelling that quietly slid
    onto the processor takes tens of minutes instead of one, and nothing on
    screen says why — the recording simply "turned out to be slow".
    """
    import torch

    from ..device import detect

    device = detect()
    if device == "cpu":
        return
    try:
        pipeline.to(torch.device(device))
    except Exception as exc:
        # The failures have nothing in common: some pyannote operations are not
        # implemented on MPS, CUDA runs out of memory, a driver that does not
        # match the build raises something of its own. Any of them is a reason to
        # stay on the CPU rather than to give up labelling altogether.
        report.say(f"{device} is unavailable, labelling on the CPU: {exc}")
