"""Summarising a transcript by map-reduce.

An hour of recording is some 25–30 thousand tokens. Handing them over in one
prompt achieves nothing: local models lose the middle well before the formal
limit of the context window. So each piece is compressed first (map), and the
compressions are drawn together into a result (reduce).

The structured answer is marked with headings rather than JSON: small models
break JSON syntax noticeably more often than they forget a heading.

The prompts below are in Russian, and deliberately so — they ask for Russian
prose, which is the language of the recordings this is used on. Changing them
changes what the model writes, so they are functional text rather than comments.
"""

from __future__ import annotations

import re

from ..models import Summary, Transcript
from ..report import Report
from .chunking import chunk_text
from .llm import LLM

_SYSTEM = (
    "Ты аналитик, который делает выжимки из расшифровок встреч, лекций и интервью. "
    "Пишешь по существу, без воды и без вводных оборотов. "
    "Опираешься только на текст расшифровки и ничего не додумываешь."
)

_MAP_PROMPT = """Ниже фрагмент расшифровки. Изложи его содержание сжато, {language}.

Сохрани: кто что предложил, принятые решения, названные числа, имена и сроки.
Выброси: приветствия, отступления, повторы, слова-паразиты.

Фрагмент:
{text}"""

_REDUCE_PROMPT = """Ниже выжимки последовательных фрагментов одной записи.
Сведи их в цельное саммари {language}.

Ответь строго в таком формате:

## ОБЗОР
Два-четыре предложения о том, чему посвящена запись и чем закончилась.

## КЛЮЧЕВЫЕ МОМЕНТЫ
- пункт
- пункт

## ЗАДАЧИ
- конкретные договорённости, задачи и решения; если их нет — напиши «нет»

Выжимки:
{text}"""

_LANGUAGE_HINT = {"ru": "по-русски", "en": "по-английски"}

_HEADING = re.compile(r"^\s*#{1,4}\s*(.+?)\s*$")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")

_SECTION_ALIASES = {
    "обзор": "overview",
    "overview": "overview",
    "ключевые моменты": "key_points",
    "key points": "key_points",
    "задачи": "action_items",
    "action items": "action_items",
}


_TOPIC_PROMPT = """О чём эта запись? Ответь одной строкой на русском, 3–8 слов.

Без кавычек, без точки в конце, без вводных вроде «запись о том, как».
Пример хорошего ответа: лекция о теореме Байеса

Начало расшифровки:
{text}"""


def topic(text: str, llm: LLM, *, sample_chars: int = 2500, temperature: float = 0.2) -> str:
    """One line about what a recording is about, for the history list.

    Хватает начала: тему занятия называют в первые минуты, а гонять по всей
    лекции ради заголовка списка — несоразмерная трата.
    """
    answer = llm.complete(
        _TOPIC_PROMPT.format(text=text[:sample_chars]),
        system=_SYSTEM,
        max_tokens=48,
        temperature=temperature,
    )
    # Now and then the model adds quotation marks or a full stop of its own.
    return answer.strip().strip('"«»').rstrip(".").strip()


def summarize(
    transcript: Transcript,
    llm: LLM,
    *,
    language: str = "ru",
    chunk_chars: int = 6000,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    report: Report | None = None,
) -> Summary:
    """Builds the summary. Short recordings go through the reduce phase only.

    Stoppable between pieces: folding an hour is minutes, and a person who
    changed their mind should not have to close the window to say so.
    """
    told = report or Report()
    hint = _LANGUAGE_HINT.get(language, f"на языке «{language}»")
    chunks = list(chunk_text(transcript.text, chunk_chars))

    if len(chunks) > 1:
        chunk_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            told.stop_if_asked()
            told.say(f"summarizing: step {index}/{len(chunks) + 1}")
            chunk_summaries.append(
                llm.complete(
                    _MAP_PROMPT.format(language=hint, text=chunk),
                    system=_SYSTEM,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            told.at(index / (len(chunks) + 1))
    else:
        # With a single piece the map phase compresses nothing and loses detail.
        chunk_summaries = chunks

    told.stop_if_asked()
    told.say("summarizing: drawing the pieces together")
    reduced = llm.complete(
        _REDUCE_PROMPT.format(language=hint, text="\n\n---\n\n".join(chunk_summaries)),
        system=_SYSTEM,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    told.at(1.0)

    sections = _parse_sections(reduced)
    return Summary(
        language=language,
        llm_model=llm.model,
        overview=sections.get("overview") or reduced.strip(),
        key_points=_bullets(sections.get("key_points", "")),
        action_items=_bullets(sections.get("action_items", "")),
        chunk_summaries=chunk_summaries if len(chunks) > 1 else [],
    )


def _parse_sections(raw: str) -> dict[str, str]:
    """Parses the answer by heading. Sections we do not know are ignored."""
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in raw.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = _SECTION_ALIASES.get(heading.group(1).strip().lower())
            if current:
                sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _bullets(block: str) -> list[str]:
    items = [match.group(1) for line in block.splitlines() if (match := _BULLET.match(line))]
    # The model may honestly answer "no" — that is not a list item.
    return (
        [] if len(items) == 1 and items[0].lower().rstrip(".") in {"нет", "none", "no"} else items
    )
