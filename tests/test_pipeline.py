"""Оркестрация: что переживает падение шага, а что нет."""

from __future__ import annotations

import pytest

from transcriber.config import Settings
from transcriber.diarize import DiarizationError
from transcriber.ingest import Source, media
from transcriber.models import Segment, Transcript
from transcriber.pipeline import LLMSlot, Pipeline
from transcriber.report import Report


@pytest.fixture
def notes() -> list[str]:
    """Сообщения, которые пайплайн адресует интерфейсу."""
    return []


@pytest.fixture
def pipeline(tmp_path, notes) -> Pipeline:
    return Pipeline(Settings(cache_dir=tmp_path), Report(say=notes.append))


@pytest.fixture
def stub_asr(pipeline, tmp_path, monkeypatch) -> Transcript:
    """Подменяет дорогие шаги: ingest и ASR отдают готовый результат."""
    audio = tmp_path / "запись.wav"
    audio.write_bytes(b"")
    source = Source(audio=audio, origin=str(audio), title="запись", duration=1.0)
    transcript = Transcript(
        source=str(audio),
        language="en",
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="Hello")],
        asr_model="stub",
    )

    monkeypatch.setattr(pipeline, "prepare", lambda target: source)
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **kw: transcript)
    return transcript


def test_diarization_failure_keeps_transcript(pipeline, stub_asr, notes, monkeypatch):
    """Диаризация — обогащение: её падение не отменяет результат ASR."""

    def explode(*args, **kwargs):
        raise DiarizationError("условия модели не приняты")

    monkeypatch.setattr(pipeline, "add_speakers", explode)

    _, transcript = pipeline.run("запись.wav", diarize=True)

    assert transcript is stub_asr
    assert transcript.speakers == []
    # Причина не проглатывается молча — она обязана дойти до интерфейса.
    assert any("условия модели не приняты" in note for note in notes)


def test_asr_failure_propagates(pipeline, monkeypatch):
    """Без распознавания возвращать нечего — такая ошибка обязана всплыть."""
    monkeypatch.setattr(pipeline, "prepare", lambda target: None)
    monkeypatch.setattr(
        pipeline, "transcribe", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ffmpeg упал"))
    )

    with pytest.raises(RuntimeError, match="ffmpeg упал"):
        pipeline.run("запись.wav")


def test_speakers_assigned_when_diarization_works(pipeline, stub_asr, monkeypatch):
    labeled = stub_asr.model_copy(
        update={"segments": [Segment(start=0.0, end=1.0, text="Hello", speaker="Спикер 1")]}
    )
    monkeypatch.setattr(pipeline, "add_speakers", lambda *a, **kw: labeled)

    _, transcript = pipeline.run("запись.wav", diarize=True)

    assert transcript.speakers == ["Спикер 1"]


def test_diarization_skipped_when_not_requested(pipeline, stub_asr, monkeypatch):
    """Выключенная диаризация не должна дёргать pyannote вовсе."""

    def explode(*args, **kwargs):
        raise AssertionError("add_speakers не должен вызываться")

    monkeypatch.setattr(pipeline, "add_speakers", explode)

    _, transcript = pipeline.run("запись.wav", diarize=False)

    assert transcript is stub_asr


# --- владение моделью ---


class FakeLLM:
    """Модель, которая ничего не грузит, но честно отвечает, поднята ли она."""

    def __init__(self, loaded: bool = False) -> None:
        self.model = "stub"
        self.loaded = loaded

    def complete(self, prompt: str, **kwargs) -> str:
        self.loaded = True
        return "ответ"


@pytest.fixture
def built(monkeypatch) -> list[FakeLLM]:
    """Все собранные за тест модели. Длина списка — сколько раз читались веса."""
    made: list[FakeLLM] = []

    def create(*args, **kwargs) -> FakeLLM:
        made.append(FakeLLM())
        return made[-1]

    monkeypatch.setattr("transcriber.pipeline.create_llm", create)
    return made


def test_empty_slot_answers_without_building(built, tmp_path):
    """Пустой слот — это тоже ответ, и он не стоит чтения весов с диска."""
    slot = LLMSlot(Settings(cache_dir=tmp_path))

    assert slot.loaded is False
    assert built == []


def test_one_model_for_everyone_sharing_the_slot(built, tmp_path, notes):
    """Регрессия: каждый перевод строил свою модель и читал веса заново.

    На девятнадцатигигабайтной модели это полминуты на нажатие и две копии в
    памяти на время чтения — при том что нужна была одна и та же.
    """
    settings = Settings(cache_dir=tmp_path)
    slot = LLMSlot(settings)

    first = Pipeline(settings, Report(say=notes.append), slot).llm
    second = Pipeline(settings, Report(say=notes.append), slot).llm

    assert first is second
    assert len(built) == 1


def loads_named(notes: list[str]) -> list[str]:
    return [note for note in notes if note.startswith("loading LLM")]


def test_the_second_press_announces_no_load(built, tmp_path, notes):
    """Регрессия: каждое нажатие перевода объявляло загрузку и делало её.

    Второй пайплайн — это второе нажатие: слот тот же, модель уже отвечала,
    и читать с диска нечего.
    """
    settings = Settings(cache_dir=tmp_path)
    slot = LLMSlot(settings)

    Pipeline(settings, Report(say=notes.append), slot).llm.complete("раз")
    Pipeline(settings, Report(say=notes.append), slot).llm.complete("два")

    assert loads_named(notes) == [f"loading LLM {settings.llm_repo}"]


def test_a_model_that_never_answered_still_announces_its_load(built, tmp_path, notes):
    """Собрать модель и прочитать её веса — у mlx разные моменты.

    Поэтому спрашивается `loaded`, а не «есть ли объект»: пока весов в памяти
    нет, загрузка предстоит, и молчать о ней не за что.
    """
    settings = Settings(cache_dir=tmp_path)
    slot = LLMSlot(settings)
    slot.get()

    Pipeline(settings, Report(say=notes.append), slot).llm.complete("раз")

    assert loads_named(notes) == [f"loading LLM {settings.llm_repo}"]


# --- контроль полноты входа ---


def test_truncation_warning_silent_on_matching_duration():
    assert media.truncation_warning(6589.0, 6589.0) is None


def test_truncation_warning_tolerates_rounding():
    """Контейнеры округляют длительность, а VBR её привирает — это не обрыв."""
    assert media.truncation_warning(6589.0, 6560.0) is None
    assert media.truncation_warning(30.0, 27.0) is None


def test_truncation_warning_catches_broken_download():
    """Тот самый случай: заголовок обещает 110 минут, звука доехало 42."""
    warning = media.truncation_warning(6589.0, 2504.0)

    assert warning is not None
    assert "41.7" in warning and "109.8" in warning


def test_truncation_warning_ignores_unknown_duration():
    """Нулевая длительность означает «не знаю», а не «ничего не доехало»."""
    assert media.truncation_warning(0.0, 2504.0) is None
    assert media.truncation_warning(6589.0, 0.0) is None
