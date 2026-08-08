"""Оркестрация: что переживает падение шага, а что нет."""

from __future__ import annotations

import pytest

from transcriber.config import Settings
from transcriber.diarize import DiarizationError
from transcriber.ingest import Source, media
from transcriber.models import Segment, Transcript
from transcriber.pipeline import Pipeline
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
