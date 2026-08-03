"""Обратное отображение таймкодов после VAD.

Самая коварная часть пайплайна: ASR видит склейку речевых кусков и выдаёт
таймкоды в сжатой системе координат, а пользователю нужны исходные.
"""

from __future__ import annotations

import pytest

from transcriber.models import Segment, Word
from transcriber.vad import SpeechRegion, Timeline


def test_identity_timeline_passes_through():
    timeline = Timeline()
    assert timeline.is_identity
    assert timeline.to_original(12.5) == 12.5


def test_maps_compressed_time_back_to_original():
    # Речь: 0–10 и 30–40. Пауза 10–30 выброшена.
    timeline = Timeline([SpeechRegion(0.0, 10.0), SpeechRegion(30.0, 40.0)])

    assert timeline.compressed_duration == 20.0
    assert timeline.to_original(0.0) == 0.0
    assert timeline.to_original(5.0) == 5.0
    # Начало второго куска на сжатой оси — это 30 с на исходной.
    assert timeline.to_original(10.0) == 30.0
    assert timeline.to_original(15.0) == 35.0


def test_clamps_overshoot_to_region_end():
    """ASR иногда выдаёт таймкод за границей куска — он не должен утечь в паузу."""
    timeline = Timeline([SpeechRegion(0.0, 10.0), SpeechRegion(30.0, 40.0)])
    assert timeline.to_original(25.0) == 40.0


def test_remap_shifts_segment_and_words():
    timeline = Timeline([SpeechRegion(0.0, 10.0), SpeechRegion(30.0, 40.0)])
    segment = Segment(
        start=9.0,
        end=11.0,
        text="через границу",
        words=[Word(start=9.0, end=10.0, text="через"), Word(start=10.0, end=11.0, text="границу")],
    )

    remapped = timeline.remap(segment)

    assert remapped.start == 9.0
    assert remapped.end == 31.0
    assert remapped.words[1].start == 30.0
    # Исходный сегмент не тронут.
    assert segment.end == 11.0


@pytest.mark.parametrize("timestamp", [0.0, 3.3, 9.99, 10.0, 19.5])
def test_monotonic(timestamp):
    """Отображение не должно идти назад по времени."""
    timeline = Timeline([SpeechRegion(0.0, 10.0), SpeechRegion(30.0, 40.0)])
    assert timeline.to_original(timestamp) <= timeline.to_original(timestamp + 0.01)
