"""Перевод транскрипта локальной LLM.

Встроенный `task=translate` у Whisper закрывает только X→английский, поэтому
RU→EN он ещё вытягивает, а EN→RU — нет. Здесь перевод делает LLM и работает в
обе стороны.

Перевод посегментный: таймкоды обязаны сохраниться, иначе субтитры развалятся.
Гонять модель по одной реплике слишком медленно, поэтому сегменты переводятся
пачками с нумерацией; если модель вернула не то число строк, пачка
переводится поштучно.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ..models import Segment, Transcript, Translation
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
    progress: Callable[[int, int], None] | None = None,
) -> Translation:
    """Переводит все сегменты, сохраняя тайминги."""
    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    batches = list(chunk_segments(transcript.segments, chunk_chars))

    translated: list[Segment] = []
    for index, batch in enumerate(batches, start=1):
        translated.extend(_translate_batch(batch, llm, target_name, max_tokens, temperature))
        if progress:
            progress(index, len(batches))

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
    # Пустые сегменты переводить нечего — они ломают нумерацию.
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
        # Модель сбилась с нумерации — надёжнее, хоть и медленнее, поштучно.
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
    """Разбирает пронумерованный ответ. None — если число строк не сошлось.

    Перевод может занять несколько строк, поэтому продолжения без номера
    приклеиваются к текущему пункту.
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
