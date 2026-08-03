"""Exporting a transcript into text formats.

Pure functions without side effects: a model in, a string out. Writing to disk
is the caller's concern.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from .models import Segment, Summary, Transcript

FORMATS = ("srt", "vtt", "txt", "md", "json")


def _timestamp(seconds: float, *, separator: str = ",") -> str:
    """HH:MM:SS,ms — the subtitle format. SRT uses a comma, WebVTT a dot."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def clock(seconds: float) -> str:
    """A short timestamp for the human-readable formats."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def group_by_speaker(segments: Sequence[Segment]) -> Iterator[tuple[str | None, list[Segment]]]:
    """Merges consecutive segments of one speaker into lines.

    Whisper cuts speech by pauses and window length rather than by speaker, so one
    person's line almost always spans several segments.
    """
    batch: list[Segment] = []
    current: str | None = None

    for segment in segments:
        if batch and segment.speaker != current:
            yield current, batch
            batch = []
        current = segment.speaker
        batch.append(segment)

    if batch:
        yield current, batch


def to_srt(transcript: Transcript, *, with_speakers: bool = True) -> str:
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        prefix = f"[{segment.speaker}] " if with_speakers and segment.speaker else ""
        blocks.append(
            f"{index}\n"
            f"{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n"
            f"{prefix}{segment.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(transcript: Transcript, *, with_speakers: bool = True) -> str:
    blocks = ["WEBVTT\n"]
    for segment in transcript.segments:
        prefix = f"<v {segment.speaker}>" if with_speakers and segment.speaker else ""
        blocks.append(
            f"{_timestamp(segment.start, separator='.')} --> "
            f"{_timestamp(segment.end, separator='.')}\n"
            f"{prefix}{segment.text}\n"
        )
    return "\n".join(blocks)


def to_txt(transcript: Transcript, *, with_speakers: bool = True) -> str:
    """Solid text: as lines when speakers are known, as one stream otherwise."""
    if not with_speakers or not transcript.speakers:
        return transcript.text

    lines = []
    for speaker, batch in group_by_speaker(transcript.segments):
        text = " ".join(s.text.strip() for s in batch if s.text.strip())
        if text:
            lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n\n".join(lines)


def to_markdown(transcript: Transcript, *, with_speakers: bool = True) -> str:
    header = [
        f"# {transcript.source}",
        "",
        f"- Language: `{transcript.language}`",
        f"- Duration: {clock(transcript.duration)}",
        f"- Model: `{transcript.asr_model}`",
    ]
    if transcript.speakers:
        header.append(f"- Speakers: {', '.join(transcript.speakers)}")
    header.append("")

    body = []
    for speaker, batch in group_by_speaker(transcript.segments):
        text = " ".join(s.text.strip() for s in batch if s.text.strip())
        if not text:
            continue
        stamp = clock(batch[0].start)
        label = f"**{speaker}**" if with_speakers and speaker else "**—**"
        body.append(f"`{stamp}` {label}\n\n{text}\n")

    return "\n".join([*header, *body])


def summary_to_markdown(summary: Summary, *, title: str = "Summary") -> str:
    parts = [f"# {title}", "", summary.overview.strip(), ""]

    if summary.key_points:
        parts += ["## Key points", "", *(f"- {p}" for p in summary.key_points), ""]
    if summary.action_items:
        parts += ["## Tasks and decisions", "", *(f"- [ ] {a}" for a in summary.action_items), ""]

    parts.append(f"*Model: `{summary.llm_model}`*")
    return "\n".join(parts)


_RENDERERS: dict[str, Callable[..., str]] = {
    "srt": to_srt,
    "vtt": to_vtt,
    "txt": to_txt,
    "md": to_markdown,
}


def render(transcript: Transcript, fmt: str, *, with_speakers: bool = True) -> str:
    """Renders a transcript into the given format. `json` is served by the model itself."""
    if fmt == "json":
        return transcript.model_dump_json(indent=2)
    try:
        return _RENDERERS[fmt](transcript, with_speakers=with_speakers)
    except KeyError:
        raise ValueError(f"unknown format: {fmt}. Available: {', '.join(FORMATS)}") from None
