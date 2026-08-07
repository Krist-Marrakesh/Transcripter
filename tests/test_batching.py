"""Разбиение записи на порции.

Ради этого память перестаёт зависеть от длины записи: в неё попадает одна
порция, а не восемь часов звука. Ошибка здесь дорогая вдвойне — либо потолок
возвращается, либо порции режут речь посередине слова.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber import audio, vad
from transcriber.vad import SpeechRegion, batches

CACHE = Path.home() / ".cache/transcript/audio/53d426fae351ad3a643a43c9840e693f.wav"


def speech(portion) -> float:
    return sum(region.duration for region in portion)


def test_a_portion_never_exceeds_the_limit():
    regions = [SpeechRegion(0, 10), SpeechRegion(20, 35), SpeechRegion(60, 64)]

    portions = batches(regions, 20)

    assert all(speech(p) <= 20 for p in portions)
    assert sum(speech(p) for p in portions) == pytest.approx(29)


def test_cuts_fall_in_the_pauses():
    """Границы порций совпадают с границами, которые нашёл VAD.

    Иначе рез пришёлся бы на середину слова, и обе половины распознались бы
    неверно — по одному обрывку с каждой стороны.
    """
    regions = [SpeechRegion(0, 10), SpeechRegion(20, 30), SpeechRegion(60, 70)]

    portions = batches(regions, 20)

    edges = {(r.start, r.end) for p in portions for r in p}
    assert edges == {(0, 10), (20, 30), (60, 70)}


def test_speech_longer_than_a_portion_is_split():
    """Страховка: непрерывная речь длиннее лимита должна как-то поместиться."""
    portions = batches([SpeechRegion(0, 50)], 20)

    assert [speech(p) for p in portions] == [20, 20, 10]
    assert portions[0][0].end == portions[1][0].start


def test_without_regions_the_whole_recording_is_divided():
    """VAD выключен или ничего не услышал — путь у вызывающего остаётся один."""
    portions = batches([], 20, total=45)

    assert [(r.start, r.end) for p in portions for r in p] == [(0, 20), (20, 40), (40, 45)]


def test_nothing_at_all_gives_nothing():
    assert batches([], 20, total=0) == []


@pytest.mark.skipif(not CACHE.exists(), reason="нужна запись в кэше")
def test_streaming_detection_matches_reading_it_whole():
    """Регрессия по замыслу: шов между блоками не должен рождать лишних границ.

    `scan` читает файл кусками, `detect` — целиком. Речь от этого меняться не
    должна, иначе каждые десять минут в транскрипте появлялся бы разрыв фразы.
    """
    whole = vad.detect(audio.load(CACHE))
    streamed = vad.scan(CACHE)

    assert len(streamed) == len(whole)
    for a, b in zip(streamed, whole, strict=True):
        assert a.start == pytest.approx(b.start, abs=0.05)
        assert a.end == pytest.approx(b.end, abs=0.05)


@pytest.mark.skipif(not CACHE.exists(), reason="нужна запись в кэше")
def test_only_the_named_stretches_are_read():
    """Паузы в память не попадают — на лекции это три четверти файла."""
    spans = [(10.0, 20.0), (100.0, 105.0)]

    samples = audio.read_spans(CACHE, spans)

    assert len(samples) == pytest.approx(audio.to_samples(15), abs=2)
