"""Рендеринг транскрипта в текстовые форматы."""

from __future__ import annotations

import json

import pytest

from transcriber.export import group_by_speaker, render, to_srt, to_txt, to_vtt
from transcriber.models import Segment, Transcript


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        source="test.mp3",
        language="ru",
        duration=10.0,
        asr_model="whisper-large-v3",
        segments=[
            Segment(start=0.0, end=2.5, text="Привет.", speaker="Спикер 1"),
            Segment(start=2.5, end=5.0, text="Как дела?", speaker="Спикер 1"),
            Segment(start=5.0, end=8.0, text="Нормально.", speaker="Спикер 2"),
        ],
    )


def test_srt_numbering_and_timestamps(transcript):
    output = to_srt(transcript)

    assert output.startswith("1\n00:00:00,000 --> 00:00:02,500\n")
    assert "3\n00:00:05,000 --> 00:00:08,000\n[Спикер 2] Нормально." in output


def test_vtt_uses_dot_separator_and_voice_tags(transcript):
    output = to_vtt(transcript)

    assert output.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in output
    assert "<v Спикер 1>Привет." in output


def test_txt_merges_consecutive_segments_of_one_speaker(transcript):
    output = to_txt(transcript)

    assert output == "Спикер 1: Привет. Как дела?\n\nСпикер 2: Нормально."


def test_txt_without_speakers_is_plain_text(transcript):
    assert to_txt(transcript, with_speakers=False) == "Привет. Как дела? Нормально."


def test_group_by_speaker_preserves_order(transcript):
    groups = list(group_by_speaker(transcript.segments))

    assert [speaker for speaker, _ in groups] == ["Спикер 1", "Спикер 2"]
    assert len(groups[0][1]) == 2


def test_render_rejects_unknown_format(transcript):
    with pytest.raises(ValueError, match="unknown format"):
        render(transcript, "docx")


def test_render_json_roundtrips(transcript):
    restored = Transcript.model_validate_json(render(transcript, "json"))
    assert restored.segments == transcript.segments


def test_hours_are_rendered(transcript):
    long_form = transcript.model_copy(
        update={"segments": [Segment(start=3661.5, end=3662.0, text="час прошёл")]}
    )
    assert "01:01:01,500 --> 01:01:02,000" in to_srt(long_form)


# --- имена вместо ярлыков ---


def named_transcript() -> Transcript:
    return Transcript(
        source="баттл.mp4",
        language="ru",
        duration=2.0,
        asr_model="stub",
        segments=[
            Segment(start=0.0, end=1.0, text="Раунд первый.", speaker="Спикер 1"),
            Segment(start=1.0, end=2.0, text="Поехали.", speaker="Спикер 2"),
        ],
        names={"Спикер 1": "Юрий"},
    )


def test_names_reach_the_rendered_text():
    """Отображение хранится отдельно от подписей, свести их — забота отрисовки.

    Ярлык, за которым имени не нашлось, остаётся ярлыком.
    """
    text = render(named_transcript(), "txt")

    assert "Юрий: Раунд первый." in text
    assert "Спикер 2: Поехали." in text


def test_json_keeps_both_the_labels_and_the_names():
    """Архивная форма: из неё должно быть видно и что разметила диаризация,
    и как это отобразили на людей. Свести их можно всегда, разделить — нет."""
    data = json.loads(render(named_transcript(), "json"))

    assert data["segments"][0]["speaker"] == "Спикер 1"
    assert data["names"] == {"Спикер 1": "Юрий"}
