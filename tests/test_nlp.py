"""Разбор ответов LLM и нарезка на куски.

Модель может ответить не тем, что просили, поэтому разбор проверяется отдельно
от самой модели — на фиктивной LLM, без загрузки весов.
"""

from __future__ import annotations

import threading

import pytest

from transcriber.models import Segment, Transcript
from transcriber.nlp.chunking import chunk_segments, chunk_text
from transcriber.nlp.formulas import _is_formula, read_formulas
from transcriber.nlp.llm import strip_thinking
from transcriber.nlp.names import name_speakers
from transcriber.nlp.summarize import _bullets, _parse_sections, summarize
from transcriber.nlp.translate import _parse_numbered, translate
from transcriber.report import Report, Stopped


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


# --- имена спикеров из речи ---


def labelled(*pairs: tuple[str, str]) -> Transcript:
    """Транскрипт с подписанными репликами: (спикер, текст)."""
    return Transcript(
        source="t.mp3",
        language="ru",
        duration=float(len(pairs)),
        asr_model="m",
        segments=[
            Segment(start=i, end=i + 1, text=text, speaker=who)
            for i, (who, text) in enumerate(pairs)
        ],
    )


def test_names_are_read_from_the_speech():
    llm = FakeLLM("Спикер 1 = Юрий Хованский\nСпикер 2 = Ларин")
    transcript = labelled(("Спикер 1", "Меня зовут Юрий."), ("Спикер 2", "А я Ларин."))

    assert name_speakers(transcript, llm) == {"Спикер 1": "Юрий Хованский", "Спикер 2": "Ларин"}


def test_the_prompt_carries_the_labels():
    """Без подписей вопрос «кто из них Ларин» нечем ответить."""
    llm = FakeLLM("")
    name_speakers(labelled(("Спикер 1", "Меня зовут Юрий.")), llm)

    assert "Спикер 1: Меня зовут Юрий." in llm.calls[0]


def test_an_unnamed_speaker_is_left_alone():
    """Молчание честнее выдумки: ярлык говорит меньше имени, но больше ложного."""
    llm = FakeLLM("Спикер 1 = Юрий")
    transcript = labelled(("Спикер 1", "Я Юрий."), ("Спикер 2", "Угу."))

    assert name_speakers(transcript, llm) == {"Спикер 1": "Юрий"}


def test_a_label_that_is_not_in_the_recording_is_dropped():
    """Ярлык из ниоткуда — признак, что модель сочиняет; строка идёт целиком вон."""
    llm = FakeLLM("Спикер 1 = Юрий\nСпикер 7 = Пушкин")

    assert name_speakers(labelled(("Спикер 1", "Я Юрий.")), llm) == {"Спикер 1": "Юрий"}


def test_the_format_line_is_not_a_name():
    """Модель нет-нет да повторит образец ответа вместо ответа."""
    llm = FakeLLM("Спикер 1 = Имя\nСпикер 2 = неизвестно")
    transcript = labelled(("Спикер 1", "раз"), ("Спикер 2", "два"))

    assert name_speakers(transcript, llm) == {}


def test_a_dash_instead_of_the_equals_sign_is_understood():
    llm = FakeLLM("Спикер 1 — Юрий Хованский")

    assert name_speakers(labelled(("Спикер 1", "раз")), llm) == {"Спикер 1": "Юрий Хованский"}


def test_a_recording_without_speakers_asks_nothing():
    """Размечать нечего — и модель ради этого поднимать не за что."""
    llm = FakeLLM("Спикер 1 = Юрий")

    assert name_speakers(make_transcript("раз", "два"), llm) == {}
    assert llm.calls == []


# --- формулы, произнесённые словами ---


def spoken(*texts: str) -> list[Segment]:
    return [Segment(start=i, end=i + 1, text=t) for i, t in enumerate(texts)]


def test_a_spoken_formula_is_written_out():
    llm = FakeLLM("0: \\lim_{x \\to 0} \\frac{\\sin x}{x} = 1")
    segments = spoken("Предел при икс стремящемся к нулю от синус икс делить на икс равен единице.")

    assert read_formulas(segments, llm) == {0: ["\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1"]}


def test_talk_about_a_formula_is_not_a_formula():
    """«Разберёмся, что такое производная» — разговор, а не запись."""
    llm = FakeLLM("")

    assert read_formulas(spoken("Давайте разберёмся, что такое производная."), llm) == {}


def test_an_answer_without_any_sign_of_maths_is_dropped():
    """Модель иногда отвечает фразой. Фраза — не формула, как её ни пронумеруй."""
    llm = FakeLLM("0: производная константы")

    assert read_formulas(spoken("Производная константы равна нулю."), llm) == {}


def test_a_number_outside_the_chunk_is_dropped():
    """Номер не из присланного куска значит, что модель считает не то, что дали."""
    llm = FakeLLM("7: x^2")

    assert read_formulas(spoken("раз", "два"), llm) == {}


def test_two_formulas_in_one_line_are_separated():
    llm = FakeLLM("0: E = mc^2; F = ma")

    assert read_formulas(spoken("Энергия равна эм цэ квадрат, сила равна эм а."), llm) == {
        0: ["E = mc^2", "F = ma"]
    }


def test_numbering_continues_across_chunks():
    """Куски идут подряд, и нумерация в промте обязана идти сквозной.

    Иначе формула из второго куска приписывается сегменту из первого — то есть
    встаёт под чужими словами.
    """
    # Записи, а не `a^2`: степень украшает одну величину и формулой не считается,
    # а проверяем мы здесь нумерацию, и подпирать её негодными данными нельзя.
    llm = FakeLLM("0: a = b + c", "2: d = e + f")
    segments = spoken("первый кусок целиком", "и ещё немного", "второй кусок")

    # Куски по 25 символов: первые две реплики в один, третья во второй.
    assert read_formulas(segments, llm, chunk_chars=25) == {0: ["a = b + c"], 2: ["d = e + f"]}


# --- остановка длинной работы ---


class CountingLLM(FakeLLM):
    """Считает, сколько кусков успели уйти в модель до остановки."""

    def __init__(self, *responses: str) -> None:
        super().__init__(*responses)
        self.asked = 0

    def complete(self, prompt: str, **kw: object) -> str:
        self.asked += 1
        return super().complete(prompt, **kw)


def test_translation_stops_between_batches():
    """Регрессия по замыслу: перевод часовой лекции — минуты, и передумать нельзя.

    Останавливается между кусками, а не внутри: кусок, уже отданный модели,
    вернётся или нет, и рвать его — выбросить потраченные на него секунды.
    """
    stop = threading.Event()
    stop.set()
    llm = CountingLLM("1. раз", "2. два")
    transcript = make_transcript("первая реплика", "вторая реплика")

    with pytest.raises(Stopped):
        translate(transcript, llm, target_language="en", chunk_chars=10, report=Report(cancel=stop))

    assert llm.asked == 0, "остановились до того, как отдали работу модели"


def test_translation_without_a_stop_goes_through():
    """Страховка: проверка остановки не должна ломать обычный путь."""
    llm = CountingLLM("1. one", "2. two")

    result = translate(make_transcript("раз", "два"), llm, target_language="en", chunk_chars=10)

    assert len(result.segments) == 2


def test_summary_stops_between_pieces():
    stop = threading.Event()
    stop.set()
    llm = CountingLLM("выжимка", "## ОБЗОР\nитог")

    with pytest.raises(Stopped):
        summarize(
            make_transcript("а" * 40, "б" * 40), llm, chunk_chars=30, report=Report(cancel=stop)
        )

    assert llm.asked == 0


def test_the_translation_says_where_it_is():
    """Полоса и строки журнала — та же забота, что и остановка, и тот же канал."""
    said: list[str] = []
    seen: list[float] = []
    llm = FakeLLM("1. one", "2. two")

    translate(
        make_transcript("раз", "два"),
        llm,
        target_language="en",
        chunk_chars=10,
        report=Report(say=said.append, at=seen.append),
    )

    assert any("translating: batch" in line for line in said)
    assert seen and seen[-1] == pytest.approx(1.0)


# --- формула или разговор о формуле ---

# Настоящие ответы модели с двух лекций: где формулы диктуют (математика для
# физиков, bco12uPk8f0) и где о них говорят (байесовский классификатор,
# OzIGqaizOAo). Придуманного здесь нет — на придуманном этот дефект не ловился.
DICTATED = [
    r"\forall \varepsilon > 0 \exists \delta > 0",
    r"0 < |x - x_0| < \delta",
    r"\lim (f \pm g) = \lim f \pm \lim g",
    r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    r"\eta_i(x) = -\frac{1}{2}\ln|\mathbf{\Sigma}_i| + \ln P(\omega_i)",
    r"p(x|\omega_i)P(\omega_i)",
    r"\mathbf{\Sigma} = \mathbf{V}\mathbf{S}\mathbf{V}^T",
    r"-\frac{1}{2}(x-\mu_i)^T\mathbf{\Sigma}_i^{-1}(x-\mu_i)",
]

NAMED = [
    r"\text{алгоритм Гауссовского-Баевского классикатора}",
    r"\text{оптимальный баевский классификатор}",
    r"\text{формула}",
    r"\mathbf{M}",
    r"\ln",
    r"VV_1",
    r"\arg\max",
    r"\sigma",
    # Обозначение самого классификатора, четырьмя строками подряд под словами
    # «применим алгоритм»: то же имя, только со скобками.
    r"\eta(x)",
    # «Обратная матрица» — слово уже сказано, степень его не дополняет.
    r"\mathbf{\Sigma}^{-1}",
]


@pytest.mark.parametrize("written", NAMED)
def test_a_named_thing_is_not_a_formula(written):
    """Символ под словом «сигма» читателю не даёт ничего: слово он уже прочёл."""
    assert _is_formula(written) is False


@pytest.mark.parametrize("written", DICTATED)
def test_what_relates_quantities_is_a_formula(written):
    assert _is_formula(written) is True


def test_one_formula_discussed_by_several_remarks_is_written_once():
    """Регрессия: одну формулу с доски модель приписывает каждой реплике о ней.

    «В этой плотности появляется…» и «первая слагаемая вот здесь» — две реплики
    в двух секундах друг от друга, и обе получили `-\frac{1}{2}\ln|\Sigma_i|`.
    Читателю нужна одна запись, у первой из них.
    """
    same = r"-\frac{1}{2}\ln|\Sigma_i|"
    llm = FakeLLM(f"0: {same}\n1: {same}\n2: {same}")
    segments = [Segment(start=float(i), end=i + 1.0, text=f"реплика {i}") for i in range(3)]

    found = read_formulas(segments, llm)

    assert found == {0: [same]}
