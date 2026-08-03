"""Нарезка транскрипта на куски под контекст LLM.

Локальные модели деградируют задолго до номинального размера контекстного окна,
поэтому режем сами и не полагаемся на то, что «влезет целиком». Границы кусков
проходят по границам сегментов — реплики не разрываются посередине.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..models import Segment


def chunk_segments(segments: Sequence[Segment], max_chars: int) -> Iterator[list[Segment]]:
    """Режет сегменты на группы не длиннее `max_chars` символов.

    Сегмент, который сам по себе длиннее лимита, отдаётся отдельным куском:
    разрезать реплику ради формального соблюдения лимита хуже, чем один раз
    превысить его.
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
    """Режет сплошной текст по границам абзацев, затем предложений."""
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
