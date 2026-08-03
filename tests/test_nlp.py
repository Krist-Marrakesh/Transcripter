"""Разбор ответов LLM и нарезка на куски.

Модель может ответить не тем, что просили, поэтому разбор проверяется отдельно
от самой модели — на фиктивной LLM, без загрузки весов.
"""

from __future__ import annotations

from transcriber.models import Segment, Transcript
from transcriber.nlp.chunking import chunk_segments, chunk_text
from transcriber.nlp.llm import strip_thinking
from transcriber.nlp.summarize import _bullets, _parse_sections, summarize
from transcriber.nlp.translate import _parse_numbered, translate


class FakeLLM:
    """Фиктивная модель: отдаёт заранее заданные ответы и считает вызовы."""

    model = "fake"

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0) if self._responses else ""


def make_transcript(*texts: str) -> Transcript:
    return Transcript(
        source="t.mp3",
        language="ru",
        duration=float(len(texts)),
        asr_model="m",
        segments=[Segment(start=i, end=i + 1, text=t) for i, t in enumerate(texts)],
    )


# --- разбор нумерованного перевода ---


def test_parse_numbered_reads_all_lines():
    assert _parse_numbered("1. Hello\n2. World", 2) == ["Hello", "World"]


def test_parse_numbered_accepts_paren_style():
    assert _parse_numbered("1) Hello\n2) World", 2) == ["Hello", "World"]


def test_parse_numbered_joins_wrapped_continuation():
    assert _parse_numbered("1. Hello\n   there\n2. World", 2) == ["Hello there", "World"]


def test_parse_numbered_rejects_wrong_count():
    """Модель потеряла строку — пачку надо переводить поштучно."""
    assert _parse_numbered("1. Hello", 2) is None


def test_parse_numbered_ignores_preamble():
    assert _parse_numbered("Вот перевод:\n\n1. Hello\n2. World", 2) == ["Hello", "World"]


# --- перевод целиком ---


def test_translate_preserves_timings_and_count():
    transcript = make_transcript("Привет", "Как дела")
    translation = translate(transcript, FakeLLM("1. Hello\n2. How are you"), target_language="en")

    assert [s.text for s in translation.segments] == ["Hello", "How are you"]
    assert [(s.start, s.end) for s in translation.segments] == [(0.0, 1.0), (1.0, 2.0)]
    assert translation.target_language == "en"


def test_translate_falls_back_to_one_by_one_on_bad_numbering():
    transcript = make_transcript("Привет", "Как дела")
    # Первый ответ сбит, дальше идут поштучные переводы.
    llm = FakeLLM("мусор без нумерации", "Hello", "How are you")

    translation = translate(transcript, llm, target_language="en")

    assert [s.text for s in translation.segments] == ["Hello", "How are you"]
    assert len(llm.calls) == 3


def test_translate_passes_empty_segments_through():
    transcript = make_transcript("", "Привет")
    translation = translate(transcript, FakeLLM("1. Hello"), target_language="en")

    assert translation.segments[0].text == ""
    assert translation.segments[1].text == "Hello"


# --- разбор саммари ---


def test_parse_sections_by_headings():
    raw = "## ОБЗОР\nПро погоду.\n\n## КЛЮЧЕВЫЕ МОМЕНТЫ\n- дождь\n- ветер"
    sections = _parse_sections(raw)

    assert sections["overview"] == "Про погоду."
    assert "- дождь" in sections["key_points"]


def test_parse_sections_ignores_unknown_headings():
    assert "overview" not in _parse_sections("## ЧТО-ТО ЕЩЁ\nтекст")


def test_bullets_treats_explicit_no_as_empty():
    assert _bullets("- нет") == []
    assert _bullets("- дождь\n- ветер") == ["дождь", "ветер"]


def test_summarize_short_transcript_skips_map_phase():
    transcript = make_transcript("Короткая запись про погоду.")
    llm = FakeLLM("## ОБЗОР\nПро погоду.\n\n## КЛЮЧЕВЫЕ МОМЕНТЫ\n- дождь")

    summary = summarize(transcript, llm)

    # Только reduce — фаза map на одном куске лишь теряет детали.
    assert len(llm.calls) == 1
    assert summary.overview == "Про погоду."
    assert summary.key_points == ["дождь"]


def test_summarize_falls_back_to_raw_answer_without_headings():
    llm = FakeLLM("Просто текст без разметки.")
    summary = summarize(make_transcript("что-то"), llm)

    assert summary.overview == "Просто текст без разметки."


# --- нарезка ---


def test_chunk_segments_respects_limit():
    segments = [Segment(start=i, end=i + 1, text="x" * 30) for i in range(5)]
    chunks = list(chunk_segments(segments, max_chars=60))

    assert [len(c) for c in chunks] == [2, 2, 1]


def test_oversized_segment_becomes_its_own_chunk():
    segments = [Segment(start=0, end=1, text="x" * 100)]
    assert len(list(chunk_segments(segments, max_chars=10))) == 1


def test_chunk_text_splits_on_paragraphs():
    text = "\n\n".join(["a" * 40] * 4)
    assert all(len(c) <= 100 for c in chunk_text(text, max_chars=100))


def test_strip_thinking_removes_reasoning_trace():
    assert strip_thinking("<think>рассуждаю</think>Ответ") == "Ответ"


def test_strip_thinking_removes_dangling_trace():
    """Открывающий тег дописывает шаблон чата, поэтому в ответе его нет."""
    assert strip_thinking("Сначала разберу задачу.\n</think>\n\nОтвет") == "Ответ"


def test_strip_thinking_keeps_answer_without_reasoning():
    assert strip_thinking("Просто ответ") == "Просто ответ"
