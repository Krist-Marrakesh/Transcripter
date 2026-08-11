"""Translating a transcript with a local LLM.

Whisper's built-in `task=translate` only ever goes into English, so it manages
RU→EN and cannot do EN→RU at all. Here the LLM does the translating and works
both ways.

Segment by segment, because the timings have to survive — subtitles fall apart
otherwise. Running the model on one line at a time is far too slow, so segments
go in numbered batches; when the model returns the wrong number of lines, that
batch is redone one line at a time.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models import Segment, Transcript, Translation
from ..report import Report
from .chunking import chunk_segments
from .llm import LLM

LANGUAGE_NAMES = {
    "ru": "русский",
    "en": "английский",
    "de": "немецкий",
    "fr": "французский",
    "es": "испанский",
    "zh": "китайский",
}

_SYSTEM = (
    "Ты профессиональный переводчик расшифровок устной речи. "
    "Переводишь точно и естественно, сохраняя разговорный регистр и termini. "
    "Не добавляешь ничего от себя, не сокращаешь и не комментируешь."
)

_BATCH_PROMPT = """Переведи реплики на {target} язык.

Требования:
- сохрани нумерацию: ровно {count} строк, каждая начинается со своего номера и точки
- одна реплика — одна строка, порядок не менять
- переводи только текст, ничего не добавляй и не объясняй
- обрывы фраз и разговорные обороты сохраняй как есть

Реплики:
{numbered}"""

_SINGLE_PROMPT = """Переведи на {target} язык. В ответе — только перевод, без пояснений.

{text}"""

_NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*)$")


def translate(
    transcript: Transcript,
    llm: LLM,
    *,
    target_language: str,
    chunk_chars: int = 6000,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    report: Report | None = None,
) -> Translation:
    """Translates every segment, keeping the timings.

    Stoppable between batches, and only there: a batch already handed to the
    model comes back or does not, and cutting it in half would throw away the
    seconds already spent on it. An hour-long lecture is minutes of work, which
    is long enough to change one's mind about.
    """
    told = report or Report()
    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    batches = list(chunk_segments(transcript.segments, chunk_chars))

    translated: list[Segment] = []
    for index, batch in enumerate(batches, start=1):
        told.stop_if_asked()
        told.say(f"translating: batch {index}/{len(batches)}")
        translated.extend(_translate_batch(batch, llm, target_name, max_tokens, temperature))
        told.at(index / len(batches))

    return Translation(
        source_language=transcript.language,
        target_language=target_language,
        llm_model=llm.model,
        segments=translated,
    )


def _translate_batch(
    batch: Sequence[Segment],
    llm: LLM,
    target_name: str,
    max_tokens: int,
    temperature: float,
) -> list[Segment]:
    # An empty segment has nothing to translate and breaks the numbering.
    payload = [(position, seg) for position, seg in enumerate(batch) if seg.text.strip()]
    if not payload:
        return list(batch)

    numbered = "\n".join(
        f"{number}. {seg.text.strip()}" for number, (_, seg) in enumerate(payload, 1)
    )
    raw = llm.complete(
        _BATCH_PROMPT.format(target=target_name, count=len(payload), numbered=numbered),
        system=_SYSTEM,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    lines = _parse_numbered(raw, len(payload))
    if lines is None:
        # The model lost the numbering — one at a time is slower but sound.
        lines = [
            llm.complete(
                _SINGLE_PROMPT.format(target=target_name, text=seg.text.strip()),
                system=_SYSTEM,
                max_tokens=max_tokens,
                temperature=temperature,
            ).strip()
            for _, seg in payload
        ]

    result = list(batch)
    for (position, segment), text in zip(payload, lines, strict=True):
        result[position] = segment.model_copy(update={"text": text, "words": []})
    return result


def _parse_numbered(raw: str, expected: int) -> list[str] | None:
    """Parses a numbered answer. None when the line count does not match.

    A translation may run to several lines, so continuations without a number of
    their own are glued to the item above them.
    """
    collected: dict[int, list[str]] = {}
    current: int | None = None

    for line in raw.splitlines():
        match = _NUMBERED_LINE.match(line)
        if match:
            current = int(match.group(1))
            collected[current] = [match.group(2).strip()]
        elif current is not None and line.strip():
            collected[current].append(line.strip())

    if set(collected) != set(range(1, expected + 1)):
        return None
    return [" ".join(collected[number]).strip() for number in range(1, expected + 1)]
