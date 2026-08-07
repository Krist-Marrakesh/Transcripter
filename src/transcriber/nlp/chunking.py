"""Cutting a transcript into pieces that fit an LLM's context.

Local models degrade well before the nominal size of their context window, so we
cut the text ourselves rather than trust that it "will all fit". The cuts fall on
segment boundaries — no line is torn in half.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..models import Segment


def chunk_segments(segments: Sequence[Segment], max_chars: int) -> Iterator[list[Segment]]:
    """Cuts segments into groups no longer than `max_chars` characters.

    A segment longer than the limit on its own is handed over as a piece of its
    own: cutting a line in two to honour a limit is worse than exceeding it once.
    """
    batch: list[Segment] = []
    size = 0

    for segment in segments:
        length = len(segment.text)
        if batch and size + length > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(segment)
        size += length

    if batch:
        yield batch


def chunk_text(text: str, max_chars: int) -> Iterator[str]:
    """Cuts continuous text on paragraph boundaries, then on sentences."""
    if len(text) <= max_chars:
        yield text
        return

    buffer: list[str] = []
    size = 0

    for paragraph in text.split("\n\n"):
        if size + len(paragraph) > max_chars and buffer:
            yield "\n\n".join(buffer)
            buffer, size = [], 0
        buffer.append(paragraph)
        size += len(paragraph) + 2

    if buffer:
        yield "\n\n".join(buffer)
